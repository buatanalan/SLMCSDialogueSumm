"""
torch 2.0.1
transformers 4.31.0
peft 0.5.0.dev0
"""
import json
import math
import os
import time
import datasets

import torch
import numpy as np
from tqdm import tqdm

from typing import *

from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, PretrainedConfig, TrainerState, Seq2SeqTrainer, \
    is_datasets_available
from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_NAME, is_apex_available, is_sagemaker_dp_enabled, is_sagemaker_mp_enabled
)
from transformers.trainer_pt_utils import nested_numpify, nested_concat, find_batch_size, IterableDatasetShard, \
    nested_truncate
from transformers.trainer_utils import has_length, EvalLoopOutput, EvalPrediction, denumpify_detensorize, speed_metrics
from transformers.utils import logging
from accelerate.utils import pad_across_processes


if is_sagemaker_dp_enabled():
    import smdistributed.dataparallel.torch.distributed as dist
else:
    import torch.distributed as dist

if is_apex_available():
    from apex import amp

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


logger = logging.get_logger(__name__)


def print_gpu_utilization(place):
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"{place} | Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

class SeqUnlikelihoodTrainer(Seq2SeqTrainer):
    def __init__(
            self,
            model = None,
            args = None,
            data_collator = None,
            data_collator_for_val_test=None,
            train_dataset = None,
            eval_dataset = None,
            ori_validation_dataset = None,
            test_dataset = None,
            tokenizer = None,
            model_init = None,
            compute_metrics = None,
            callbacks = None,
            optimizers = (None, None),
            ori_dev_test_batch_size=None
    ):
        # ============================================================
        # FIX TOKENIZER: gunakan processing_class untuk transformers >= 4.46.
        # Pakai keyword args (bukan positional) untuk hindari mismatch posisi.
        # ============================================================
        import inspect
        super_init_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
        super_kwargs = dict(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        # transformers >= 4.46 menggunakan processing_class, sebelumnya tokenizer
        if "processing_class" in super_init_params:
            super_kwargs["processing_class"] = tokenizer
        else:
            super_kwargs["tokenizer"] = tokenizer

        super().__init__(**super_kwargs)

        self._max_length = 128
        self._num_beams = 8

        # Simpan tokenizer SEKALI sebagai atribut publik (untuk akses internal)
        self.tokenizer = tokenizer

        self.ori_validation_dataset = ori_validation_dataset
        self.data_collator_for_val_test = data_collator_for_val_test
        self.test_dataset = test_dataset
        self.ori_dev_test_batch_size = ori_dev_test_batch_size

    def _remove_unused_columns(self, dataset, description: Optional[str] = None):
        pass

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        total_loss, lm_loss, margin_loss, model_output = self._compute_loss(model, inputs)

        # self.log({
        #     "main_loss": lm_loss.detach().item(),
        #     "margin_loss": margin_loss.detach().item(),
        #     "total_loss": total_loss.detach().item(),
        # })
        
        return (total_loss, model_output) if return_outputs else total_loss

    def _compute_loss(self, model, inputs):
        # 1. Pisahkan custom keys agar tidak dikirim ke forward model
        custom_keys = ['ref_summ_idx', 'pos_neg', 'src_select_indices']
        model_inputs = {k: v for k, v in inputs.items() if k not in custom_keys}
        
        B, K, N = model_inputs['input_ids'].shape
    
        # Reshape untuk forward pass: (B, K, N) -> (B*K, N)
        for k in ['input_ids', 'attention_mask', 'labels', 'decoder_input_ids']:
            if k in model_inputs:
                model_inputs[k] = model_inputs[k].reshape(-1, model_inputs[k].size(-1))
    
        model_output = model(**model_inputs, use_cache=False)
        lm_logits = model_output.logits  # (B*K, N, V)
        labels = model_inputs['labels']   # (B*K, N)
    
        # ============================================================
        # FIX #3: shift untuk Causal LM, TIDAK shift untuk Seq2Seq.
        # ============================================================
        # Deteksi tipe model dari config.is_encoder_decoder.
        # Aman untuk PEFT (model.config tetap mewarisi dari base).
        underlying = model.module if hasattr(model, "module") else model
        is_encoder_decoder = getattr(underlying.config, "is_encoder_decoder", False)
        # Fallback: cek via base model untuk PEFT-wrapped model
        if not is_encoder_decoder and hasattr(underlying, "base_model"):
            inner = underlying.base_model
            is_encoder_decoder = getattr(getattr(inner, "config", inner), "is_encoder_decoder", False)
    
        if is_encoder_decoder:
            # Seq2Seq (BART/T5/Pegasus): decoder sudah aligned dengan labels,
            # tidak perlu shift manual.
            shift_logits = lm_logits.contiguous()
            shift_labels = labels.contiguous()
        else:
            # Causal LM (Qwen/Llama): token N memprediksi token N+1.
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
    
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        token_loss = token_loss.view(B * K, -1)  # (B*K, L)
        
        del shift_logits, lm_logits  # Hapus untuk membebaskan VRAM
        # Masking untuk menghitung rata-rata log_prob per sample
        mask = (shift_labels != -100).float()

        log_probs = -(token_loss * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        log_probs = log_probs.view(B, K)  # (B, K)
    
        # 3. Margin Loss (Vectorized)
        pos_neg = inputs['pos_neg'].view(B, K)  # 1 untuk pos, 0 untuk neg
        
        # Ambil score untuk pos dan neg
        pos_scores = torch.where(pos_neg == 1, log_probs, torch.tensor(float('inf'), device=log_probs.device))
        neg_scores = torch.where(pos_neg == 0, log_probs, torch.tensor(float('-inf'), device=log_probs.device))

        min_pos_score = pos_scores.min(dim=1).values
        max_neg_score = neg_scores.max(dim=1).values
        theta = getattr(self.args, "theta", 1.0)
        alpha = getattr(self.args, "alpha", 1.0)

        # Hitung margin loss per batch: max(0, theta + max_neg - min_pos)
        margin_loss_per_sample = torch.clamp(theta + max_neg_score - min_pos_score, min=0)
        # Hindari margin loss pada batch yang tidak punya pos atau neg
        valid_margin_mask = (pos_neg.sum(1) > 0) & ((pos_neg == 0).sum(1) > 0)
        margin_loss = (
            margin_loss_per_sample[valid_margin_mask].mean()
            if valid_margin_mask.any()
            else torch.tensor(0.0, device=log_probs.device)
        )

        self.state.batch_pos_tgt_scores = pos_scores
        self.state.batch_neg_tgt_scores = neg_scores
        self.state.batch_min_pos_score = min_pos_score
        self.state.batch_max_neg_score = max_neg_score
        self.state.margin_loss = margin_loss.detach().clone()
    
        # 4. Cross Entropy (Likelihood) pada Reference
        ref_mask = inputs["ref_summ_idx"].view(B, K) == 1
        if ref_mask.any():
            # log_probs negatif -> NLL positif
            lm_loss = -log_probs[ref_mask].mean()
        else:
            lm_loss = torch.tensor(0.0, device=log_probs.device, requires_grad=True)
    
        # Total Loss
        total_loss = lm_loss + alpha * margin_loss
        self.state.model_loss = total_loss.detach().clone()

        if not model.training:
            self.log_dict = {
                "margin_loss": margin_loss.item(),
                "lm_loss": lm_loss.item(),
                "avg_pos_score": log_probs[pos_neg == 1].mean().item()
                if (pos_neg == 1).any() else 0.0,
            }
    
        return total_loss, lm_loss, margin_loss, model_output

    # def _compute_loss(self, model, inputs):
    #     # TODO: What is "inputs"? Is it a batch? -> yes

    #     # uncomment this block to monitor gpu usage/cpu memory/cpu time/gpu time, .etc. of all function calls
    #     # can be very slow. unblock when finished debugging
    #     # with torch.profiler.profile(profile_memory=True, record_shapes=True) as prof:
    #     #     model_output = model(**inputs, use_cache=False)  # some heavy computations potentially causing OOM
    #     # print(">>> profiler details:")
    #     # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

    #     model_output = model(**inputs, use_cache=False)

    #     # shape: N x seq_len x vocab_size
    #     # N is the total number of positive and negative target summaries in a batch
    #     # usually N is larger than batch_size since there can be multiple positive
    #     # and negative target summaries for each sample (dialogue) in the batch
    #     lm_logits = model_output.logits

    #     # calculate the log likelihood for each target sequence (which is the negative cross entropy loss)
    #     loss = None
    #     lm_loss = None
    #     batch_size = inputs['input_ids'].size(0)
    #     margin_loss = torch.zeros(batch_size, device=inputs['input_ids'].device)
    #     theta = self.args.theta  # the margin in margin loss
    #     batch_pos_tgt_scores = []
    #     batch_neg_tgt_scores = []
    #     batch_min_pos_score = []
    #     batch_max_neg_score = []
    #     if 'labels' in inputs:
    #         logits_permuted = lm_logits.permute(0, 2, 1)

    #         # print(f"logits_permuted size: {logits_permuted.size()}")
    #         loss_fct = CrossEntropyLoss(reduction='none')
    #         log_probs = -loss_fct(logits_permuted, inputs['labels']).mean(dim=1)
    #         # ==============
    #         #   margin loss
    #         # ==============
    #         for i in range(batch_size):
    #             # print(log_probs)
    #             target_scores_for_sample = log_probs[inputs['src_select_indices'] == i]
    #             pos_neg_for_sample = inputs['pos_neg'][inputs['src_select_indices'] == i]
    #             pos_tgt_scores = target_scores_for_sample[pos_neg_for_sample == 1]
    #             neg_tgt_scores = target_scores_for_sample[pos_neg_for_sample == 0]
    #             if len(pos_tgt_scores) > 0 and len(neg_tgt_scores) > 0:

    #                 min_pos_score = pos_tgt_scores.min()
    #                 max_neg_score = neg_tgt_scores.max()
    #                 # margin loss
    #                 sample_margin_loss = torch.clamp(torch.zeros(1, device=inputs['input_ids'].device), theta + max_neg_score - min_pos_score)
    #                 margin_loss[i] = sample_margin_loss
    
    #                 batch_min_pos_score.append(min_pos_score.detach())
    #                 batch_max_neg_score.append(max_neg_score.detach())
    #                 batch_pos_tgt_scores.append(pos_tgt_scores.detach())
    #                 batch_neg_tgt_scores.append(neg_tgt_scores.detach())

    #         # add likelihood information in trainer state
    #         self.state.batch_pos_tgt_scores = batch_pos_tgt_scores
    #         self.state.batch_neg_tgt_scores = batch_neg_tgt_scores
    #         self.state.batch_min_pos_score = batch_min_pos_score
    #         self.state.batch_max_neg_score = batch_max_neg_score
    #         self.state.margin_loss = margin_loss.detach().clone()

    #         # =================
    #         #   cross entropy
    #         # =================
    #         ref_summs_pred = lm_logits[inputs["ref_summ_idx"] == 1]
    #         ref_labels = inputs["labels"][inputs["ref_summ_idx"] == 1]
    #         lm_loss_fct = CrossEntropyLoss(reduction='mean')
    #         lm_loss = lm_loss_fct(ref_summs_pred.view(-1, lm_logits.size(-1)), ref_labels.view(-1))

    #         loss = lm_loss + self.args.alpha * margin_loss.mean()
    #         self.state.model_loss = loss.detach().clone()

    #     return loss, lm_loss, margin_loss, model_output

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, logits and labels (each being optional).
        """
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        # Priority (handled in generate):
        # gen_kwargs > model.generation_config > default GenerationConfig()
        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = 128
        gen_kwargs["num_beams"] = 4
        default_synced_gpus = False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
                "labels" in inputs
                and "decoder_input_ids" in inputs
                and inputs["labels"].shape == inputs["decoder_input_ids"].shape
        ):
            # inputs_for_gen = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}
            inputs_for_gen = {}
            inputs_for_gen['input_ids'] = inputs['input_ids']
            inputs_for_gen['attention_mask'] = inputs['attention_mask']
            if len(inputs['input_ids'].shape) == 3:
                inputs_for_gen['input_ids'] = inputs_for_gen['input_ids'].view(-1, inputs['input_ids'].size(2))
                inputs_for_gen['attention_mask'] = inputs_for_gen['attention_mask'].view(-1, inputs['input_ids'].size(2))
        generated_tokens = self.model.generate(**inputs_for_gen, **gen_kwargs)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config

        # 1. Tentukan target_len dengan sistem prioritas (Fallback Chain)
        # Prioritas: Argumen Trainer > Config Generasi > Config Model > Default
        target_len = getattr(self, "_max_length", None) or getattr(gen_config, "max_length", None)
        
        if target_len is None:
            if getattr(gen_config, "max_new_tokens", None) is not None:
                # Jika menggunakan max_new_tokens, pad ke panjang tensor saat ini 
                # (karena sudah mencakup input + generated tokens)
                target_len = generated_tokens.shape[-1]
            else:
                # Fallback terakhir ke config model atau nilai standar
                target_len = getattr(self.model.config, "max_length", 512)
        
        # 2. Eksekusi Padding secara Aman
        if generated_tokens.shape[-1] < target_len:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, target_len)


        if 'pos_neg' in inputs:
            with torch.no_grad():
                # compute loss on predict data
                loss, lm_loss, margin_loss, model_output = self._compute_loss(model, inputs)
        else:
            with torch.no_grad():
                # 1. Flatten 3D ke 2D: [B, N, S] -> [B*N, S]
                # Ini krusial karena model (Llama/Qwen) mengharapkan input 2D
                input_ids = inputs['input_ids'].view(-1, inputs['input_ids'].size(-1))
                attention_mask = inputs['attention_mask'].view(-1, inputs['attention_mask'].size(-1))
                labels = inputs['labels'].view(-1, inputs['labels'].size(-1))

                # 2. Forward pass manual
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True
                )

                # 3. Ekstraksi Loss
                # Jika labels diberikan, model Hugging Face otomatis menghitung CrossEntropyLoss
                loss = outputs.loss
                # print(f"lm loss: {lm_loss}, margin loss: {margin_loss}")
        labels = inputs['labels']
        return (loss, generated_tokens, labels)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        args = self.args
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only
    
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
    
        # Prepare model for distributed eval in Transformers 5.x
        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            if self.is_fsdp_enabled:
                self.model = model
            if model is not self.model:
                self.model_wrapped = model
    
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
    
        batch_size = self.args.eval_batch_size
        model.eval()
    
        self.callback_handler.eval_dataloader = dataloader
        eval_dataset = getattr(dataloader, "dataset", None)
    
        # Final CPU containers
        all_losses, all_preds, all_labels, all_inputs = None, None, None, None
        # Intermediate host containers
        losses_host, preds_host, labels_host, inputs_host = None, None, None, None
    
        observed_num_examples = 0
        for step, inputs in enumerate(dataloader):
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                if batch_size is None:
                    batch_size = observed_batch_size
    
            # 1. Jalankan Prediction Step Utama
            # Di Seq2SeqTrainer, jika predict_with_generate=True, 
            # logits di sini akan berisi generated_ids

            loss, logits, labels = self.prediction_step(model, inputs, False, ignore_keys=ignore_keys)
            # 2. FIX: Pastikan logits tidak None jika kita ingin menghitung metrik (Summarization)
            if logits is None and not prediction_loss_only and self.args.predict_with_generate:
                 # Jika prediction_step gagal generate, kita generate manual di sini
                 gen_kwargs = {
                    "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
                    "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
                 }
                 inputs = self._prepare_inputs(inputs)
                 logits = self.model.generate(**inputs, **gen_kwargs)
    
            inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_for_metrics else None
    
            # 3. Gathering menggunakan Accelerator (Versi 5.x)
            if loss is not None:
                # Ulangi loss sebanyak batch_size agar sinkron saat gather
                loss_t = loss.repeat(observed_batch_size) if loss.dim() == 0 else loss
                losses = self.accelerator.gather_for_metrics(loss_t)
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
    
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits)
                logits = self.accelerator.gather_for_metrics(logits)
                # Simpan hasil ke preds_host agar bisa di-numpify nanti
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels)
                labels = self.accelerator.gather_for_metrics(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
    
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode)
                inputs_decode = self.accelerator.gather_for_metrics(inputs_decode)
                inputs_host = inputs_decode if inputs_host is None else nested_concat(inputs_host, inputs_decode, padding_index=-100)
    
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)
    
            # Accumulation ke CPU jika memory GPU penuh
            # print(losses_host)
            # print(labels_host)
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    all_losses = nested_numpify(losses_host) if all_losses is None else np.concatenate((all_losses, nested_numpify(losses_host)), axis=0)
                if preds_host is not None:
                    all_preds = nested_numpify(preds_host) if all_preds is None else nested_concat(all_preds, nested_numpify(preds_host), padding_index=-100)
                if labels_host is not None:
                    all_labels = nested_numpify(labels_host) if all_labels is None else nested_concat(all_labels, nested_numpify(labels_host), padding_index=-100)
                if inputs_host is not None:
                    all_inputs = nested_numpify(inputs_host) if all_inputs is None else nested_concat(all_inputs, nested_numpify(inputs_host), padding_index=-100)
                
                losses_host, preds_host, labels_host, inputs_host = None, None, None, None

        # 4. Pindahkan sisa host containers ke CPU (Final Gathering)
        if losses_host is not None:
            all_losses = nested_numpify(losses_host) if all_losses is None else np.concatenate((all_losses, nested_numpify(losses_host)), axis=0)
        if preds_host is not None:
            all_preds = nested_numpify(preds_host) if all_preds is None else nested_concat(all_preds, nested_numpify(preds_host), padding_index=-100)
        if labels_host is not None:
            all_labels = nested_numpify(labels_host) if all_labels is None else nested_concat(all_labels, nested_numpify(labels_host), padding_index=-100)
        if inputs_host is not None:
            all_inputs = nested_numpify(inputs_host) if all_inputs is None else nested_concat(all_inputs, nested_numpify(inputs_host), padding_index=-100)
    
        # Truncate untuk membuang padding DDP
        num_samples = observed_num_examples
        if all_losses is not None: all_losses = all_losses[:num_samples]
        if all_preds is not None: all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None: all_labels = nested_truncate(all_labels, num_samples)
        if all_inputs is not None: all_inputs = nested_truncate(all_inputs, num_samples)
    
        # 5. Hitung Metrik
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_for_metrics:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs))
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}
    
        metrics = denumpify_detensorize(metrics)
        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
    
        # Prefixing
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        
        self.log_metrics(description, metrics)
        self.save_metrics(description, metrics)
    
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
        
    # def evaluation_loop(
    #     self,
    #     dataloader: DataLoader,
    #     description: str,
    #     prediction_loss_only: Optional[bool] = None,
    #     ignore_keys: Optional[List[str]] = None,
    #     metric_key_prefix: str = "eval",
    # ) -> EvalLoopOutput:
    #     """
    #     Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

    #     Works both with or without labels.
    #     """
    #     args = self.args

    #     prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

    #     # if eval is called w/o train, handle model prep here
    #     # if self.is_deepspeed_enabled and self.model_wrapped is self.model:
    #     #     _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

    #     model = self._wrap_model(self.model, training=False, dataloader=dataloader)

    #     if len(self.accelerator._models) == 0 and model is self.model:
    #         model = (
    #             self.accelerator.prepare(model)
    #             if self.is_deepspeed_enabled
    #             else self.accelerator.prepare_model(model, evaluation_mode=True)
    #         )

    #         if self.is_fsdp_enabled:
    #             self.model = model

    #         # for the rest of this function `model` is the outside model, whether it was wrapped or not
    #         if model is not self.model:
    #             self.model_wrapped = model

    #         # backward compatibility
    #         if self.is_deepspeed_enabled:
    #             self.deepspeed = self.model_wrapped

    #     # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
    #     # while ``train`` is running, cast it to the right dtype first and then put on device
    #     if not self.is_in_train:
    #         if args.fp16_full_eval:
    #             model = model.to(dtype=torch.float16, device=args.device)
    #         elif args.bf16_full_eval:
    #             model = model.to(dtype=torch.bfloat16, device=args.device)

    #     batch_size = self.args.eval_batch_size

    #     logger.info(f"***** Running {description} *****")
    #     if has_length(dataloader):
    #         logger.info(f"  Num examples = {self.num_examples(dataloader)}")
    #     else:
    #         logger.info("  Num examples: Unknown")
    #     logger.info(f"  Batch size = {batch_size}")

    #     model.eval()

    #     self.callback_handler.eval_dataloader = dataloader
    #     # Do this before wrapping.
    #     eval_dataset = getattr(dataloader, "dataset", None)

    #     # if is_torch_tpu_available():
    #     #     dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

    #     # if args.past_index >= 0:
    #     #     self._past = None

    #     # Initialize containers
    #     # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
    #     losses_host = None
    #     preds_host = None
    #     labels_host = None
    #     inputs_host = None

    #     # losses/preds/labels on CPU (final containers)
    #     all_losses = None
    #     all_preds = None
    #     all_labels = None
    #     all_inputs = None
    #     # Will be useful when we have an iterable dataset so don't know its length.

    #     observed_num_examples = 0
    #     # Main evaluation loop
    #     for step, inputs in enumerate(dataloader):
    #         # Update the observed num examples
    #         observed_batch_size = find_batch_size(inputs)
    #         if observed_batch_size is not None:
    #             observed_num_examples += observed_batch_size
    #             # For batch samplers, batch_size is not known by the dataloader in advance.
    #             if batch_size is None:
    #                 batch_size = observed_batch_size

    #         # Prediction step
    #         loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
    #         inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_for_metrics else None

    #         # show the input documents and generated summaries, to make sure the training process is correct
    #         if step == 0:
    #             # XXX: adapt synced_gpus for fairscale as well
    #             gen_kwargs = {
    #                 "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
    #                 "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
    #                 "synced_gpus": False,
    #             }

    #             if self.tokenizer is not None:
    #                 # some fileds got filtered out here
    #                 generation_inputs = {k: v for k, v in inputs.items() if k in self.tokenizer.model_input_names}
    #                 if len(inputs['input_ids'].shape) == 3:
    #                     generation_inputs['input_ids'] = generation_inputs['input_ids'].view(-1, inputs['input_ids'].size(2))
    #                     generation_inputs['attention_mask'] = generation_inputs['attention_mask'].view(-1, inputs['input_ids'].size(2))
    #                 # very ugly hack to make it work
    #                 generation_inputs["input_ids"] = generation_inputs.pop(self.tokenizer.model_input_names[0])
    #             else:
    #                 generation_inputs = {"input_ids": inputs["input_ids"]}

    #             if torch.cuda.is_available():
    #                 generation_inputs["input_ids"] = generation_inputs["input_ids"].to('cuda')
    #                 generation_inputs["attention_mask"] = generation_inputs["attention_mask"].to('cuda')
    #             print(generation_inputs['input_ids'])
    #             print(gen_kwargs)
    #             generated_tokens = self.model.generate(
    #                 **generation_inputs,
    #                 **gen_kwargs,
    #             )
    #             print(generated_tokens)
    #             decoded_generations = self.tokenizer.batch_decode(generated_tokens)
    #             print(decoded_generations)
    #         # if is_torch_tpu_available():
    #         #     xm.mark_step()

    #         # Update containers on host
    #         if loss is not None:
    #             losses = self.accelerator.gather_for_metrics(loss.repeat(batch_size))
    #             losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
    #         if labels is not None:
    #             labels = self._pad_across_processes(labels)
    #         if inputs_decode is not None:
    #             inputs_decode = self._pad_across_processes(inputs_decode)
    #             inputs_decode = self.accelerator.gather_for_metrics(inputs_decode)
    #             inputs_host = (
    #                 inputs_decode
    #                 if inputs_host is None
    #                 else nested_concat(inputs_host, inputs_decode, padding_index=-100)
    #             )
    #         if logits is not None:
    #             # logits = self._pad_across_processes(logits)
    #             logits = self.accelerator.pad_across_processes(logits)
    #             if self.preprocess_logits_for_metrics is not None:
    #                 logits = self.preprocess_logits_for_metrics(logits, labels)
    #             logits = self.accelerator.gather_for_metrics(logits)
    #             preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
    #         if labels is not None:
    #             labels = self.accelerator.gather_for_metrics(labels)
    #             labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
    #         self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

    #         # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
    #         if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
    #             if losses_host is not None:
    #                 losses = nested_numpify(losses_host)
    #                 all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
    #             if preds_host is not None:
    #                 logits = nested_numpify(preds_host)
    #                 all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
    #             if inputs_host is not None:
    #                 inputs_decode = nested_numpify(inputs_host)
    #                 all_inputs = (
    #                     inputs_decode
    #                     if all_inputs is None
    #                     else nested_concat(all_inputs, inputs_decode, padding_index=-100)
    #                 )
    #             if labels_host is not None:
    #                 labels = nested_numpify(labels_host)
    #                 all_labels = (
    #                     labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
    #                 )

    #             # Set back to None to begin a new accumulation
    #             losses_host, preds_host, inputs_host, labels_host = None, None, None, None

    #     # if args.past_index and hasattr(self, "_past"):
    #     #     # Clean the state at the end of the evaluation loop
    #     #     delattr(self, "_past")

    #     # Gather all remaining tensors and put them back on the CPU
    #     print(1)
    #     print(all_preds)
    #     if losses_host is not None:
    #         losses = nested_numpify(losses_host)
    #         all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
    #     if preds_host is not None:
    #         logits = nested_numpify(preds_host)
    #         all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
    #     if inputs_host is not None:
    #         inputs_decode = nested_numpify(inputs_host)
    #         all_inputs = (
    #             inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
    #         )
    #     if labels_host is not None:
    #         labels = nested_numpify(labels_host)
    #         all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

    #     # Number of samples
    #     if has_length(eval_dataset):
    #         num_samples = len(eval_dataset)
    #     # The instance check is weird and does not actually check for the type, but whether the dataset has the right
    #     # methods. Therefore we need to make sure it also has the attribute.
    #     elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
    #         num_samples = eval_dataset.num_examples
    #     else:
    #         if has_length(dataloader):
    #             num_samples = self.num_examples(dataloader)
    #         else:  # both len(dataloader.dataset) and len(dataloader) fail
    #             num_samples = observed_num_examples
    #     if num_samples == 0 and observed_num_examples > 0:
    #         num_samples = observed_num_examples

    #     # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
    #     # samplers has been rounded to a multiple of batch_size, so we truncate.
    #     print(all_preds)
    #     if all_losses is not None:
    #         all_losses = all_losses[:num_samples]
    #     if all_preds is not None:
    #         all_preds = nested_truncate(all_preds, num_samples)
    #     if all_labels is not None:
    #         all_labels = nested_truncate(all_labels, num_samples)
    #     if all_inputs is not None:
    #         all_inputs = nested_truncate(all_inputs, num_samples)

    #     # Metrics!
    #     if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
    #         if args.include_for_metrics:
    #             metrics = self.compute_metrics(
    #                 EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
    #             )
    #         else:
    #             metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
    #     else:
    #         metrics = {}

    #     # To be JSON-serializable, we need to remove numpy types or zero-d tensors
    #     metrics = denumpify_detensorize(metrics)

    #     if all_losses is not None:
    #         metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
    #     if hasattr(self, "jit_compilation_time"):
    #         metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

    #     # Prefix all keys with metric_key_prefix + '_'
    #     for key in list(metrics.keys()):
    #         if not key.startswith(f"{metric_key_prefix}_"):
    #             metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

    #     self.log_metrics(description, metrics)
    #     self.save_metrics(description, metrics)

    #     return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        print("=> Begin Evaluation Loop")
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        # eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        eval_loop = self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        # if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
        #     # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
        #     xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        # save generated summaries to disk
        summary_file = os.path.join(self.args.output_dir, f"STEP-{self.state.global_step}_generated_summaries.json")
        # Replace -100s used for padding as we can't decode them
        preds = output.predictions
        tokenizer = self.tokenizer
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # post process
        decoded_preds = [pred.strip().replace("\n", " ") for pred in decoded_preds]  # ensure no newline char in output
        # preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
        with open(summary_file, "w") as file:
            json.dump(decoded_preds, file, indent=4)

        logger.info("=== Generate Summaries on Test Set ===")
        # predict summaries on test set
        test_dataloader = self.get_dataloader_for_val_test(self.test_dataset)
        test_output = eval_loop(
            test_dataloader,
            description="Test",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=None,
            ignore_keys=ignore_keys,
            metric_key_prefix="test",
        )
        # save generated summaries to disk
        test_summary_file = os.path.join(self.args.output_dir, f"STEP-{self.state.global_step}_test_generated_summaries.json")
        # Replace -100s used for padding as we can't decode them
        test_preds = test_output.predictions
        tokenizer = self.tokenizer
        test_preds = np.where(test_preds != -100, test_preds, tokenizer.pad_token_id)
        test_decoded_preds = tokenizer.batch_decode(test_preds, skip_special_tokens=True)

        # post process
        test_decoded_preds = [pred.strip().replace("\n", " ") for pred in test_decoded_preds]  # ensure no newline char in output
        # preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
        with open(test_summary_file, "w") as file:
            json.dump(test_decoded_preds, file, indent=4)

        # =======================================
        #  predict summaries on original dev set
        # =======================================
        logger.info("=== Generate Summaries on Ori Dev Set")
        ori_val_dataloader = self.get_dataloader_for_val_test(self.ori_validation_dataset)
        output = eval_loop(
            ori_val_dataloader,
            description="Original Validation Set",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=None,
            ignore_keys=ignore_keys,
            metric_key_prefix="ori_val",
        )
        # save generated summaries to disk
        summary_file = os.path.join(
            self.args.output_dir,
            f"STEP-{self.state.global_step}_oridev_generated_summaries.json"
        )
        # Replace -100s used for padding as we can't decode them
        preds = output.predictions
        tokenizer = self.tokenizer
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # post process
        decoded_preds = [pred.strip().replace("\n", " ") for pred in decoded_preds]  # ensure no newline char in output
        # preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
        with open(summary_file, "w") as file:
            json.dump(decoded_preds, file, indent=4)

        return output.metrics

    def get_dataloader_for_val_test(self, dataset):
        if dataset is None:
            return None
            
        # Filter kolom yang tidak perlu
        cols_to_remove = [c for c in ['dialogue', 'summary', 'summaries'] if c in dataset.column_names]
        eval_dataset = dataset.remove_columns(cols_to_remove)

        return DataLoader(
            eval_dataset,
            sampler=self._get_eval_sampler(eval_dataset),
            batch_size=self.ori_dev_test_batch_size or self.args.eval_batch_size,
            collate_fn=self.data_collator_for_val_test,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

class CausalUnlikelihoodTrainer(Trainer):
    def __init__(
            self,
            model=None,
            args=None,
            data_collator=None,
            data_collator_for_val_test=None,
            train_dataset=None,
            eval_dataset=None,
            ori_validation_dataset=None,
            test_dataset=None,
            tokenizer=None,
            model_init=None,
            compute_metrics=None,
            callbacks=None,
            optimizers=(None, None),
            ori_dev_test_batch_size=None,
    ):
        # ============================================================
        # FIX TOKENIZER: gunakan processing_class untuk transformers >= 4.46.
        # Pakai keyword args (bukan positional) supaya tidak mismatch posisi.
        # ============================================================
        import inspect
        super_init_params = inspect.signature(Trainer.__init__).parameters
        super_kwargs = dict(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        if "processing_class" in super_init_params:
            super_kwargs["processing_class"] = tokenizer
        else:
            super_kwargs["tokenizer"] = tokenizer

        super().__init__(**super_kwargs)

        # Llama biasanya butuh max_length lebih besar (prompt + gen)
        self._num_beams = 2 # Beams tinggi pada Llama sangat memakan VRAM

        # Simpan tokenizer sebagai atribut publik
        self.tokenizer = tokenizer

        self.ori_validation_dataset = ori_validation_dataset
        self.data_collator_for_val_test = data_collator_for_val_test
        self.test_dataset = test_dataset
        self.ori_dev_test_batch_size = ori_dev_test_batch_size
        
        self._gen_kwargs = {
            "max_new_tokens": 128,
            "do_sample": False,
            "num_beams" : self._num_beams,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "num_return_sequences" : 1
        }

    def _remove_unused_columns(self, dataset, description: Optional[str] = None):
        pass
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        total_loss, lm_loss, margin_loss, model_output, min_pos, max_neg = self._compute_loss(model, inputs)

        self.log({
            "main_loss": lm_loss.detach().item(),
            "margin_loss": margin_loss.detach().item(),
            "total_loss": total_loss.detach().item(),
            "min_pos": min_pos.detach().item(),
            "max_neg": max_neg.detach().item()
        })
        
        return (total_loss, model_output) if return_outputs else total_loss

    # def _compute_loss(self, model, inputs):
    #     # Llama Causal LM tidak menggunakan decoder_input_ids
    #     B, K, N = inputs['input_ids'].shape
    #     print(B, K, N)

    #     inputs['input_ids'] = inputs['input_ids'].view(B * K, N)
    #     inputs['attention_mask'] = inputs['attention_mask'].view(B * K, N)
    #     inputs['labels'] = inputs['labels'].view(B * K, N)
    #     model_output = model(**inputs, use_cache=False)
    #     lm_logits = model_output.logits

    #     loss = None
    #     lm_loss = None        
    #     # Ambil hyperparameter dari args
    #     theta = getattr(self.args, "theta", 1.0)
    #     alpha = getattr(self.args, "alpha", 1.0)
        

    #     batch_pos_tgt_scores = []
    #     batch_neg_tgt_scores = []
    #     batch_min_pos_score = []
    #     batch_max_neg_score = []
    #     if 'labels' in inputs:
    #         # Shift logits untuk Causal LM: token n memprediksi n+1
    #         # Namun jika labels sudah diproses di collator (di-shift), gunakan standar
    #         shift_logits = lm_logits[..., :-1, :].contiguous()
    #         shift_labels = inputs['labels'][..., 1:].contiguous()
           
    #         loss_fct = nn.CrossEntropyLoss(reduction='none')
    #         log_probs = -loss_fct(
    #             shift_logits.view(-1, shift_logits.size(-1)),
    #             shift_labels.view(-1)
    #         )
    #         log_probs = log_probs.view(shift_logits.size(0), -1)
            
    #         # Hitung jumlah token yang tidak di-mask (setelah shift)
    #         num_unmasked = (shift_labels != -100).sum(dim=1)
    #         log_probs = log_probs.sum(dim=1) / (num_unmasked + 1e-9)
    #         log_probs = log_probs.view(B,K, -1)
    #         lm_logits = lm_logits.view(B,K,-1)
    #         # ==============
    #         #  Margin Loss
    #         # ==============
    #         batch_size = B
    #         margin_loss = torch.zeros(batch_size, device=inputs['input_ids'].device)
            
    #         for i in range(batch_size):
    #             mask = inputs['src_select_indices'][i] == i
    #             target_scores_for_sample = log_probs[i][mask]
    #             pos_neg_for_sample = inputs['pos_neg'][i][mask]
                
    #             pos_tgt_scores = target_scores_for_sample[pos_neg_for_sample == 1]
    #             neg_tgt_scores = target_scores_for_sample[pos_neg_for_sample == 0]
                
    #             if len(pos_tgt_scores) > 0 and len(neg_tgt_scores) > 0:
    #                 min_pos_score = pos_tgt_scores.min()
    #                 max_neg_score = neg_tgt_scores.max()
    #                 sample_margin_loss = torch.clamp(theta + max_neg_score - min_pos_score, min=0)
    #                 margin_loss[i] = sample_margin_loss
    #                 batch_min_pos_score.append(min_pos_score.detach())
    #                 batch_max_neg_score.append(max_neg_score.detach())
    #                 batch_pos_tgt_scores.append(pos_tgt_scores.detach())
    #                 batch_neg_tgt_scores.append(neg_tgt_scores.detach())

    #         # add likelihood information in trainer state
    #         self.state.batch_pos_tgt_scores = batch_pos_tgt_scores
    #         self.state.batch_neg_tgt_scores = batch_neg_tgt_scores
    #         self.state.batch_min_pos_score = batch_min_pos_score
    #         self.state.batch_max_neg_score = batch_max_neg_score
    #         self.state.margin_loss = margin_loss.detach().clone()
    #         # =================
    #         # Cross Entropy (Likelihood)
    #         # =================
    #         # Hanya hitung LM loss pada referensi asli (ref_summ_idx == 1)
    #         ref_mask = inputs["ref_summ_idx"] == 1
    #         if ref_mask.any():
    #             print(ref_mask)
    #             print(log_probs.shape)
    #             # ref_logits = lm_logits[ref_mask]
    #             # ref_labels = inputs["labels"][ref_mask]


                
    #             # lm_loss_fct = nn.CrossEntropyLoss(reduction='mean')
    #             # lm_loss = lm_loss_fct(ref_logits.view(-1, lm_logits.size(-1)), ref_labels.view(-1))
    #             lm_loss = log_probs[ref_mask]
    #             lm_loss = lm_loss.view(-1).unsqueeze()
    #         else:
    #             lm_loss = torch.tensor(0.0, device=lm_logits.device)
    #         loss = lm_loss + alpha * margin_loss.mean().unsqueeze
    #         self.state.model_loss = loss.detach().clone()


    #     return loss, lm_loss, margin_loss, model_output

    def _compute_loss(self, model, inputs):
        # ===== Toggle debug =====
        DEBUG = False   # set False saat training normal
        DEBUG_FIRST_N_BATCHES = 2   # batasi agar log tidak meledak
    
        # ===== Ambil tokenizer untuk decode =====
        tok = getattr(self, 'tokenizer', None) or getattr(self, 'processing_class', None)
        PAD = tok.pad_token_id if tok is not None else None
    
        # 1. Pisahkan custom keys agar tidak dikirim ke forward model
        custom_keys = ['ref_summ_idx', 'pos_neg', 'src_select_indices']
        model_inputs = {k: v for k, v in inputs.items() if k not in custom_keys}
        B, K, N = model_inputs['input_ids'].shape
    
        # Simpan referensi 3D dulu sebelum reshape (untuk debug)
        ids3d  = model_inputs['input_ids'].clone() if DEBUG else None
        lbl3d  = model_inputs['labels'].clone()    if DEBUG else None
        attn3d = model_inputs['attention_mask'].clone() if (DEBUG and 'attention_mask' in model_inputs) else None
        pn     = inputs['pos_neg'].view(B, K)
    
        # ==================== DEBUG: STRUKTUR & ISI INPUT ====================
        step = getattr(self.state, 'global_step', 0)
        if DEBUG and step < DEBUG_FIRST_N_BATCHES:
            print("\n" + "="*72)
            print(f"[DEBUG step={step}] B={B}, K={K}, N={N}")
            print(f"  pos_neg shape={tuple(inputs['pos_neg'].shape)}, "
                  f"dtype={inputs['pos_neg'].dtype}, unique={inputs['pos_neg'].unique().tolist()}")
    
            for b in range(B):
                print(f"\n--- batch {b}  pos_neg={pn[b].tolist()} ---")
                for k_ in range(K):
                    ids  = ids3d[b, k_]
                    lbl  = lbl3d[b, k_]
                    attn = attn3d[b, k_] if attn3d is not None else None
                    tag  = "POS" if pn[b, k_].item() == 1 else "NEG"
    
                    # statistik token
                    n_pad    = (ids == PAD).sum().item() if PAD is not None else -1
                    n_nonpad = N - n_pad
                    n_attn1  = attn.sum().item() if attn is not None else None
                    n_valid  = (lbl != -100).sum().item()
    
                    print(f"\n  [k={k_} {tag}]")
                    print(f"    non_pad={n_nonpad}  pad={n_pad}  attn=1: {n_attn1}  valid_labels={n_valid}")
    
                    # raw ids di awal & akhir
                    print(f"    input_ids[:50]  = {ids[:50].tolist()}")
                    print(f"    input_ids = {ids.tolist()}")
                    print(lbl)

                    print(f"    labels[-15:]    = {lbl[-15:].tolist()}")
    
                    # posisi & isi token label yang valid
                    valid_pos = (lbl != -100).nonzero(as_tuple=True)[0].tolist()
                    print(f"    posisi label valid: {valid_pos[:20]}{'...' if len(valid_pos)>20 else ''}")
                    if valid_pos and tok is not None:
                        vids = lbl[valid_pos].tolist()
                        print(f"    decode label valid: {tok.decode(vids)!r}")
    
                    # decode input
                    if tok is not None:
                        ids_nopad = ids[ids != PAD] if PAD is not None else ids
                        txt = tok.decode(ids_nopad.tolist(), skip_special_tokens=False)
                        print(f"    decode input head (200ch): {txt[:200]!r}")
                        print(f"    decode input tail (200ch): {txt[-200:]!r}")
    
                # bandingkan langsung antar kandidat (kalau ada pos & neg)
                pos_idx = (pn[b] == 1).nonzero(as_tuple=True)[0].tolist()
                neg_idx = (pn[b] == 0).nonzero(as_tuple=True)[0].tolist()
                if pos_idx and neg_idx:
                    p, n = pos_idx[0], neg_idx[0]
                    same_ids = torch.equal(ids3d[b, p], ids3d[b, n])
                    same_lbl = torch.equal(lbl3d[b, p], lbl3d[b, n])
                    print(f"\n  perbandingan pos[k={p}] vs neg[k={n}]: "
                          f"input_ids_eq={same_ids}, labels_eq={same_lbl}")
        # ==================== END DEBUG INPUT ====================
    
        # 2. Reshape untuk forward pass
        for k in ['input_ids', 'attention_mask', 'labels']:
            if k in model_inputs:
                model_inputs[k] = model_inputs[k].reshape(-1, model_inputs[k].size(-1))
    
        model_output = model(**model_inputs, use_cache=False)
        lm_logits = model_output.logits  # (B*K, N, V)
    
        # 3. Hitung log-prob per sample
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = model_inputs['labels'][..., 1:].contiguous()
    
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(B * K, -1)
        del shift_logits, lm_logits
    
        mask = (shift_labels != -100).float()
        log_probs = -(token_loss * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        log_probs = log_probs.view(B, K)
    
        # ==================== DEBUG: LOG_PROBS ====================
        if DEBUG and step < DEBUG_FIRST_N_BATCHES:
            print("\n[DEBUG] LOG_PROBS & VALID TOKEN PER KANDIDAT")
            mask3d = mask.view(B, K, -1)
            for b in range(B):
                for k_ in range(K):
                    tag = "POS" if pn[b, k_].item() == 1 else "NEG"
                    print(f"  b={b} k={k_} {tag}  log_prob={log_probs[b, k_].item():+.6f}  "
                          f"valid_tok={mask3d[b, k_].sum().item():.0f}")
        # ==================== END DEBUG LOG_PROBS ====================
    
        # 4. Margin loss (vectorized)
        pos_neg = inputs['pos_neg'].view(B, K)
        pos_scores = torch.where(pos_neg == 1, log_probs,
                                 torch.tensor(float('inf'),  device=log_probs.device))
        neg_scores = torch.where(pos_neg == 0, log_probs,
                                 torch.tensor(float('-inf'), device=log_probs.device))
    
        min_pos_score = pos_scores.min(dim=1).values
        max_neg_score = neg_scores.max(dim=1).values
    
        theta = getattr(self.args, "theta", 1.0)
        alpha = getattr(self.args, "alpha", 1.0)
    
        margin_loss_per_sample = torch.clamp(theta + max_neg_score - min_pos_score, min=0)
        valid_margin_mask = (pos_neg.sum(1) > 0) & ((pos_neg == 0).sum(1) > 0)
        margin_loss = (margin_loss_per_sample[valid_margin_mask].mean()
                       if valid_margin_mask.any()
                       else torch.tensor(0.0, device=log_probs.device))
    
        # Simpan untuk logging
        self.state.batch_pos_tgt_scores = pos_scores.detach().cpu()
        self.state.batch_neg_tgt_scores = neg_scores.detach().cpu()
        self.state.batch_min_pos_score  = min_pos_score.detach().cpu()
        self.state.batch_max_neg_score  = max_neg_score.detach().cpu()
        self.state.margin_loss          = margin_loss.detach().cpu()
    
        # ==================== DEBUG: SKOR AKHIR ====================
        if DEBUG and step < DEBUG_FIRST_N_BATCHES:
            print("\n[DEBUG] SKOR AKHIR (per batch)")
            for b in range(B):
                mp = min_pos_score[b].item()
                mn = max_neg_score[b].item()
                print(f"  batch {b}: min_pos={mp:+.6f}  max_neg={mn:+.6f}  "
                      f"diff={mp-mn:+.2e}  margin_loss_raw={max(0.0, theta+mn-mp):+.6f}")
            print("="*72 + "\n")
        # ==================== END DEBUG SKOR ====================
    
        # 5. LM loss pada referensi
        ref_mask = inputs["ref_summ_idx"].view(B, K) == 1
        if ref_mask.any():
            lm_loss = -log_probs[ref_mask].mean()
        else:
            lm_loss = torch.tensor(0.0, device=log_probs.device, requires_grad=True)
    
        # Total loss
        total_loss = lm_loss + alpha * margin_loss
        self.state.model_loss = total_loss.detach().item()
    
        if not model.training:
            self.log_dict = {
                "margin_loss": margin_loss.item(),
                "lm_loss": lm_loss.item(),
                "avg_pos_score": log_probs[pos_neg == 1].mean().item()
            }
    
        return total_loss, lm_loss, margin_loss, model_output, min_pos_score, max_neg_score

    def prediction_step(
        self, 
        model: nn.Module, 
        inputs: Dict[str, Union[torch.Tensor, Any]], 
        prediction_loss_only: bool, 
        ignore_keys: Optional[List[str]] = None
       ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
    
        inputs = self._prepare_inputs(inputs)
        
        # ============================================================
        # JALUR 1: CONTRASTIVE / UNLIKELIHOOD (Ada src_select_indices)
        # ============================================================
        if "src_select_indices" in inputs:
            with torch.no_grad():
                # Menggunakan _compute_loss kustom Anda untuk training/val contrastive
                loss, _, margin_loss, output = self._compute_loss(model, inputs)
                
                # PENTING: Pindahkan ke CPU. Qwen Logits (151k vocab) sangat berat!
                return (
                    loss.detach(),
                    margin_loss.detach(),
                    output.logits.detach().cpu(), 
                    inputs['labels'].detach().cpu()
                )
        
        # ============================================================
        # JALUR 2: STANDARD INFERENCE / TEST (Slicing + Generate + Manual Loss)
        # ============================================================
        if prediction_loss_only:
            # Jika hanya minta loss tanpa generate (hemat VRAM)
            with torch.no_grad():
                outputs = model(**{k: v for k, v in inputs.items() if k in self.tokenizer.model_input_names})
                return (outputs.loss.detach(), None, None)

        # 1. Persiapkan Input
        full_input_ids = inputs["input_ids"]
        full_attention_mask = inputs["attention_mask"]
        labels = inputs.get("labels")

        # Handle 3D shape (B, K, N) -> (B*K, N)
        if full_input_ids.dim() == 3:
            B, K, N = full_input_ids.shape
            full_input_ids = full_input_ids.reshape(B * K, N)
            full_attention_mask = full_attention_mask.reshape(B * K, N)
            if labels is not None:
                labels = labels.reshape(B * K, N)

        # ============================================================
        # FIX #9: SLICE PROMPT supaya model TIDAK lihat ground truth target.
        # ============================================================
        # Format input: [PAD..PAD, src_tokens, tgt_tokens, PAD..PAD]
        # Label:        [-100..-100, -100..-100, tgt_tokens, -100..-100]
        # Posisi pertama label != -100 = awal target = AKHIR prompt.
        # ============================================================
        if labels is not None:
            is_content = (labels != -100)
            # argmax mengembalikan indeks pertama nilai True (=1.0)
            prompt_lengths = is_content.float().argmax(dim=-1)

            # Edge case: row yang semua -100 -> argmax=0; fallback ke attn sum
            all_pad_rows = ~is_content.any(dim=-1)
            if all_pad_rows.any():
                fallback_len = full_attention_mask.sum(dim=-1)
                prompt_lengths = torch.where(
                    all_pad_rows, fallback_len, prompt_lengths
                )

            max_prompt_len = prompt_lengths.max().item()
            # Guard: pastikan minimal 1 token
            max_prompt_len = max(1, max_prompt_len)

            # Potong sampai batas prompt terpanjang
            gen_input_ids = full_input_ids[:, :max_prompt_len].clone()
            gen_attention_mask = full_attention_mask[:, :max_prompt_len].clone()

            # Bersihkan baris yang prompt-nya lebih pendek (cegah kebocoran)
            for i, p_len in enumerate(prompt_lengths):
                p_len_i = p_len.item()
                if p_len_i < max_prompt_len:
                    gen_input_ids[i, p_len_i:] = self.tokenizer.pad_token_id
                    gen_attention_mask[i, p_len_i:] = 0
        else:
            # Tidak ada labels -> pakai full input
            gen_input_ids = full_input_ids
            gen_attention_mask = full_attention_mask
            max_prompt_len = full_input_ids.size(-1)

        # 2. Generate
        gen_kwargs = {
            "max_new_tokens": 128,
            "do_sample": False,
            "num_beams": self._num_beams,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "num_return_sequences": 1,
        }
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=gen_input_ids,            # FIX #9: pakai CROPPED
                attention_mask=gen_attention_mask,   # FIX #9: pakai CROPPED
                **gen_kwargs,
            )

        # 3. Hitung Loss Manual pada bagian yang DIGENERATE (setelah prompt)
        loss = torch.tensor(0.0, device=generated_ids.device)
        if generated_ids.size(1) > max_prompt_len:
            with torch.no_grad():
                outputs = model(
                    input_ids=generated_ids,
                    attention_mask=(generated_ids != self.tokenizer.pad_token_id).long(),
                )
                generated_logits = outputs.logits  # (B, L, V)

                # Causal LM shift, hanya bagian yang baru di-generate
                # Posisi prompt_len-1 memprediksi token prompt_len (awal generation).
                shift_logits = generated_logits[:, max_prompt_len - 1:-1, :].contiguous()
                shift_targets = generated_ids[:, max_prompt_len:].contiguous()

                if shift_logits.size(1) > 0:
                    loss_fct = nn.CrossEntropyLoss(reduction='mean')
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_targets.view(-1),
                    )

        # 5. Return (loss, margin_loss=loss as placeholder, generated_ids, labels)
        # evaluation_loop di CausalUnlikelihoodTrainer unpacking 4 nilai:
        #   loss, margin_loss, logits, labels = self.prediction_step(...)
        # Kita kembalikan generated_ids (token IDs) sebagai "logits" --
        # nanti di evaluation_loop kalau dim==3 akan di-argmax (tidak terjadi
        # karena generated_ids 2D), kalau dim<3 langsung dipakai sebagai pred.
        return (
            loss.detach(),
            loss.detach(),  # margin_loss placeholder
            generated_ids.detach().cpu(),
            inputs.get("labels").detach().cpu() if inputs.get("labels") is not None else None,
        )

    # Tambahkan fungsi pembantu untuk validasi test set seperti versi BART Anda
    def get_dataloader_for_val_test(self, dataset):
        if dataset is None:
            return None
            
        # Filter kolom yang tidak perlu
        cols_to_remove = [c for c in ['dialogue', 'summary', 'summaries'] if c in dataset.column_names]
        eval_dataset = dataset.remove_columns(cols_to_remove)

        return DataLoader(
            eval_dataset,
            sampler=self._get_eval_sampler(eval_dataset),
            batch_size=self.ori_dev_test_batch_size or self.args.eval_batch_size,
            collate_fn=self.data_collator_for_val_test,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        print("=> Begin Evaluation Loop")
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        # eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop

        eval_loop = self.evaluation_loop
        
        print_gpu_utilization("start eval loop")
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        print_gpu_utilization("finish eval loop")

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        # if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
        #     # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
        #     xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        # save generated summaries to disk
        summary_file = os.path.join(self.args.output_dir, f"STEP-{self.state.global_step}_generated_summaries.json")
        # Replace -100s used for padding as we can't decode them
        preds = output.predictions
        tokenizer = self.tokenizer
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # post process
        decoded_preds = [pred.strip().replace("\n", " ") for pred in decoded_preds]  # ensure no newline char in output
        # preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
        with open(summary_file, "w") as file:
            json.dump(decoded_preds, file, indent=4)

        logger.info("=== Generate Summaries on Test Set ===")
        # predict summaries on test set
        print_gpu_utilization("start test loop")
        test_dataloader = self.get_dataloader_for_val_test(self.test_dataset)
        test_output = eval_loop(
            test_dataloader,
            description="Test",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=None,
            ignore_keys=ignore_keys,
            metric_key_prefix="test",
        )
        print_gpu_utilization("finish test loop")
        # save generated summaries to disk
        test_summary_file = os.path.join(self.args.output_dir, f"STEP-{self.state.global_step}_test_generated_summaries.json")
        # Replace -100s used for padding as we can't decode them
        test_preds = test_output.predictions
        tokenizer = self.tokenizer
        test_preds = np.where(test_preds != -100, test_preds, tokenizer.pad_token_id)
        test_decoded_preds = tokenizer.batch_decode(test_preds, skip_special_tokens=True)

        # post process
        test_decoded_preds = [pred.strip().replace("\n", " ") for pred in test_decoded_preds]  # ensure no newline char in output
        # preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
        with open(test_summary_file, "w") as file:
            json.dump(test_decoded_preds, file, indent=4)

        # =======================================
        #  predict summaries on original dev set
        # =======================================
        logger.info("=== Generate Summaries on Ori Dev Set")
        print_gpu_utilization("start ori dev loop")
        ori_val_dataloader = self.get_dataloader_for_val_test(self.ori_validation_dataset)
        output = eval_loop(
            ori_val_dataloader,
            description="Original Validation Set",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=None,
            ignore_keys=ignore_keys,
            metric_key_prefix="ori_val",
        )
        print_gpu_utilization("finish ori dev loop")
        # save generated summaries to disk
        summary_file = os.path.join(
            self.args.output_dir,
            f"STEP-{self.state.global_step}_oridev_generated_summaries.json"
        )
        # Replace -100s used for padding as we can't decode them
        preds = output.predictions
        tokenizer = self.tokenizer
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # post process
        decoded_preds = [pred.strip().replace("\n", " ") for pred in decoded_preds]  # ensure no newline char in output
        # preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
        with open(summary_file, "w") as file:
            json.dump(decoded_preds, file, indent=4)

        return output.metrics

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Versi ringkas evaluation_loop khusus untuk CausalLM.
        Input utama: dataloader
        """
        model = self.model
        model.eval()
    
        # Ganti bagian loop dan penggabungan akhir Anda menjadi seperti ini:
        all_losses = []
        all_margin_losses = []
        final_preds = None  # Gunakan sebagai akumulator
        final_labels = None
        
        print(f"***** Running {description} *****")
        
        for step, inputs in enumerate(tqdm(dataloader)):
            loss, margin_loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only=False)
        
            if loss is not None:
                all_losses.append(loss.item())
                all_margin_losses.append(margin_loss.item())
        
            # Ambil Prediksi
            if logits is not None:
                # Jika logits 3D (B, N, V), ambil argmax. Jika sudah 2D (IDs), gunakan langsung.
                preds = torch.argmax(logits, dim=-1) if logits.dim() == 3 else logits
                
                # Pindahkan ke CPU sebagai tensor
                preds = preds.detach().cpu()
                
                # Akumulasi langsung menggunakan nested_concat
                # Fungsi ini akan otomatis menangani padding jika panjang sekuens antar batch berbeda
                final_preds = preds if final_preds is None else nested_concat(final_preds, preds, padding_index=-100)
        
            # Ambil Labels
            if labels is not None:
                labels = labels.detach().cpu()
                final_labels = labels if final_labels is None else nested_concat(final_labels, labels, padding_index=-100)
        
        # --- Pasca Loop ---
        
        # Hitung rata-rata loss
        avg_loss = np.mean(all_losses) if all_losses else 0.0
        avg_margin_loss = np.mean(all_margin_losses) if all_margin_losses else 0.0
        
        # Konversi hasil akhir ke NumPy agar bisa diproses oleh compute_metrics (ROUGE, dsb)
        if final_preds is not None:
            final_preds = final_preds.numpy()
        if final_labels is not None:
            final_labels = final_labels.numpy()
    
        # 5. Hitung Metrik Custom (Opsional)

        metrics = {"loss": avg_loss, "margin_loss" : avg_margin_loss}
        if self.compute_metrics is not None and final_preds is not None and final_labels is not None:
            # Panggil fungsi compute_metrics milikmu (misal untuk hitung ROUGE/Accuracy)
            custom_metrics = self.compute_metrics((final_preds, final_labels))
            metrics.update(custom_metrics)

        self.log_metrics(description, metrics)
        self.save_metrics(description, metrics)
    
        return EvalLoopOutput(
            predictions=final_preds, 
            label_ids=final_labels, 
            metrics=metrics, 
            num_samples=len(final_labels) if final_labels is not None else 0
        )