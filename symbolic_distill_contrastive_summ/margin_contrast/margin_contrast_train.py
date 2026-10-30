import os
import re
import nltk

import numpy as np
import torch
import pathlib
import json

from dataclasses import dataclass, field

from datasets import load_dataset
import evaluate
from huggingface_hub.utils import logging
from transformers import HfArgumentParser, Seq2SeqTrainingArguments, BartTokenizerFast, PegasusTokenizerFast, \
    T5TokenizerFast, DataCollatorForSeq2Seq, LlamaTokenizerFast, AutoTokenizer, DataCollatorForLanguageModeling, default_data_collator, AutoConfig
# from margin_contrast.callbacks import CustomWandbCallback
from margin_contrast.margin_contrast_data import ContrastiveMultiDataset, DataCollatorForContrastive
from margin_contrast.margin_contrast_models import BartForContrastive, T5ForContrastive, PegasusForContrastive, LlamaForContrastive, QwenForContrastive
from margin_contrast.margin_contrast_trainer import SeqUnlikelihoodTrainer, CausalUnlikelihoodTrainer

logger = logging.get_logger(__name__)

@dataclass
class MySeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    use_original_ref: bool = field(
        default=True, metadata={"help": "Whether to use original human-written reference summaries"}
    )
    use_lora: bool = field(
        default=False, metadata={"help": "Whether to use LoRA for parameter efficient finetuning."}
    )

def main():
    parser = HfArgumentParser((MySeq2SeqTrainingArguments,))
    parser.add_argument('--data_dir', type=str, default="")
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--model_type', type=str, choices=["bart", "pegasus", "t5", "llama", "qwen"])
    parser.add_argument('--theta', type=float, default=0.5, help="the margin in margin loss")
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help="the weight of margin loss in the final loss "
             "(i.e., final loss = standard_crossentropy_loss + alpha * margin_loss)"
    )
    parser.add_argument('--prompt_path', type=str)
    parser.add_argument('--max_source_length', type=int, default=512)
    parser.add_argument('--max_target_length', type=int, default=128)
    parser.add_argument('--ignore_pad_token_for_loss', type=bool, default=True)
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--pad_to_max_length', type=bool, default=False)
    parser.add_argument('--contrast_policy', type=str, default="setwise",
                        choices=["setwise", "pairwise"])
    parser.add_argument('--contrastive_train_file', type=str,
                        help="file for contrastive training set")
    parser.add_argument('--contrastive_dev_file', type=str,
                        help="file for contrastive development set")
    parser.add_argument('--validation_file', type=str, help="file for original development set (instead of contrastive)")
    parser.add_argument('--test_file', type=str, help="file for original test set (instead of contrastive)")
    parser.add_argument('--ori_dev_test_batch_size', type=int, default=8)
    parser.add_argument('--resume_from_ckpt', type=bool, default=False)
    parser.add_argument('--base_adapter', type=str, default=None)
    # parser.add_argument('--use_lora', type=bool, default=False)
    
    training_args, other_args = parser.parse_args_into_dataclasses()

    # very important, otherwise only those columns included in model.forward will be kept,
    # and we can not do further processing in this case
    training_args.remove_unused_columns = False

    training_args.alpha = other_args.alpha
    training_args.theta = other_args.theta

    if other_args.model_type == "bart":
        model = BartForContrastive.from_pretrained(other_args.model_name_or_path)
        tokenizer = BartTokenizerFast.from_pretrained(other_args.model_name_or_path)
    elif other_args.model_type == "pegasus":
        model = PegasusForContrastive.from_pretrained(other_args.model_name_or_path)
        tokenizer = PegasusTokenizerFast.from_pretrained(other_args.model_name_or_path)
    elif other_args.model_type == "t5":
        model = T5ForContrastive.from_pretrained(other_args.model_name_or_path)
        tokenizer = T5TokenizerFast.from_pretrained(other_args.model_name_or_path)
    elif other_args.model_type == "llama":
        model = LlamaForContrastive.from_pretrained(other_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(other_args.model_name_or_path)
    elif other_args.model_type == "qwen":
        # ============================================================
        # FIX: Tokenizer Qwen yang AMAN (tidak corrupt saat di-save)
        # ============================================================
        # BUG SEBELUMNYA:
        #   tokenizer.pad_token = "<|endoftext|>"  (string)
        # bisa menambah token sebagai "added_tokens" baru, lalu saat
        # save mid-training, tokenizer.json jadi corrupt (vocab=1 token).
        #
        # SOLUSI: assign via existing token ID, JANGAN tambah baru.
        # ============================================================
        tokenizer = AutoTokenizer.from_pretrained(
            other_args.model_name_or_path,
            trust_remote_code=True,
            use_fast=True,
        )

        # ----- Set pad_token via existing token (TIDAK menambah token) -----
        if tokenizer.pad_token_id is None:
            # Qwen2.5 base: <|endoftext|> ada di vocab dengan ID 151643.
            # Cek via convert_tokens_to_ids (return unk_token_id kalau tidak ada).
            endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
            if endoftext_id is not None and endoftext_id != tokenizer.unk_token_id:
                # Set via attribute internal -- JANGAN pakai assignment string,
                # supaya tidak memicu add_special_tokens.
                tokenizer.pad_token_id = endoftext_id
            else:
                # Fallback: pakai eos_token_id yang sudah ada
                tokenizer.pad_token_id = tokenizer.eos_token_id
                # Sinkronkan pad_token string juga (aman karena eos_token sudah ada)
                tokenizer.pad_token = tokenizer.eos_token

        # Causal LM dengan format [PAD..PAD, src, tgt] -> left pad source
        tokenizer.padding_side = "left"

        # ----- Verifikasi tokenizer SEHAT sebelum melanjutkan -----
        vocab_size = tokenizer.vocab_size
        tok_len = len(tokenizer)
        assert vocab_size > 1000, (
            f"Tokenizer corrupt! vocab_size={vocab_size}. "
            f"Cek model_name_or_path: {other_args.model_name_or_path}"
        )
        # Sanity: encode-decode round trip
        test_text = "Hello, this is a test."
        test_ids = tokenizer.encode(test_text, add_special_tokens=False)
        test_decoded = tokenizer.decode(test_ids)
        assert len(test_ids) > 1, f"Tokenizer broken: '{test_text}' -> {test_ids}"

        # ----- Load config & model -----
        config = AutoConfig.from_pretrained(
            other_args.model_name_or_path,
            trust_remote_code=True,
        )
        config.pad_token_id = tokenizer.pad_token_id

        model = QwenForContrastive.from_pretrained(
            other_args.model_name_or_path,
            config=config,
            trust_remote_code=True,
        )
        model.config.pad_token_id = tokenizer.pad_token_id

        # Sanity check
        print(f"[Qwen Setup]")
        print(f"  model_path       : {other_args.model_name_or_path}")
        print(f"  tokenizer vocab  : {vocab_size}")
        print(f"  len(tokenizer)   : {tok_len}")
        print(f"  model vocab      : {model.config.vocab_size}")
        print(f"  pad_token        : {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
        print(f"  eos_token        : {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
        print(f"  padding_side     : {tokenizer.padding_side}")
        print(f"  model class      : {type(model).__name__}")
        print(f"  test encode      : {test_ids[:5]}... -> {test_decoded!r}")
        assert "Contrastive" in type(model).__name__, (
            "FATAL: model bukan QwenForContrastive! "
            "Cek inherit class di margin_contrast_models.py"
        )
    else:
        raise ValueError(f"Model type [{other_args.model_type}] not supported.")

    if training_args.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType, PeftModel
        print(training_args)
        if other_args.base_adapter:
            print('using base adapter')
            model = PeftModel.from_pretrained(model, other_args.base_adapter, is_trainable=True)
        else:
            # Apply LoRA
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM if other_args.model_type in ["pegasus", "bart", "t5"] else TaskType.CAUSAL_LM
            )
            model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        model.print_trainable_parameters()

    data_dir = other_args.data_dir

    source_prefix = ''
    if other_args.model_type == "t5":
        source_prefix = "summarize: "

    # FIX: teruskan is_decoder_only ke ContrastiveMultiDataset
    # supaya format prompt training sesuai (causal LM perlu "\nSummary: " suffix)
    is_decoder_only_flag = not model.config.is_encoder_decoder

    train_dataset = ContrastiveMultiDataset(
        os.path.join(data_dir, other_args.contrastive_train_file),
        tokenizer,
        other_args.max_source_length,
        other_args.max_target_length,
        source_prefix=source_prefix,
        policy=other_args.contrast_policy,
        is_decoder_only=is_decoder_only_flag,
        prompt=other_args.prompt_path,
    )

    dev_dataset = ContrastiveMultiDataset(
        os.path.join(data_dir, other_args.contrastive_dev_file),
        tokenizer,
        other_args.max_source_length,
        other_args.max_target_length,
        source_prefix=source_prefix,
        policy=other_args.contrast_policy,
        is_decoder_only=is_decoder_only_flag,
        prompt=other_args.prompt_path,
    )

    # load test dataset
    text_column = "dialogue"
    summary_column = "summary"
    padding = "max_length" if other_args.pad_to_max_length else False

    def preprocess_function(examples):
                
        # 1. Bersihkan data dari None
        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])
        
        is_encoder_decoder = model.config.is_encoder_decoder

        # --- LOGIKA SEQ2SEQ (Encoder-Decoder seperti T5/BART) ---
        if is_encoder_decoder:
            
            inputs = [source_prefix + inp for inp in inputs]
    
            # 2. Tokenize dengan padding FIX (max_length)
            model_inputs = tokenizer(
                inputs, 
                max_length=other_args.max_source_length, 
                padding="max_length", 
                truncation=True
            )
        
            labels = tokenizer(
                text_target=targets, 
                max_length=other_args.max_target_length, 
                padding="max_length", 
                truncation=True
            )
        
            # 3. Handle ignore_pad_token_for_loss (-100)
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] 
                for label in labels["input_ids"]
            ]
        
            # 4. Bungkus menjadi 3D: [1, seq_len] untuk setiap sampel
            model_inputs["input_ids"] = [[seq] for seq in model_inputs["input_ids"]]
            model_inputs["attention_mask"] = [[seq] for seq in model_inputs["attention_mask"]]
            model_inputs["labels"] = [[seq] for seq in labels["input_ids"]]
        
            return model_inputs

        # --- LOGIKA CAUSAL LM (Decoder-only seperti Llama/Qwen) ---
        else:
            # FIX #6: HAPUS setup pad_token/padding_side di sini.
            # Setup sudah dilakukan di awal main() (sebelum dataset dibuat),
            # supaya konsisten antara training dan validation.
            # tokenizer.pad_token = tokenizer.eos_token        <- HAPUS
            # model.config.pad_token_id = model.config.eos_token_id <- HAPUS
            # tokenizer.padding_side = "left"                  <- HAPUS

            all_input_ids = []
            all_labels = []
            all_attention_masks = []
            prompt_path = pathlib.Path(__file__).parent.parent.parent / f'generate_dataset/{other_args.prompt_path}'

            with open(prompt_path, "r") as f:
                try:
                    data = json.loads(f.read())
                    instruct = data["prompt_positive"]
                except Exception as e:
                    print(f"Failed to load prompt !!! {e}")
                    instruct = "Summarize the following dialogue:\n"
            # Tentukan jatah masing-masing (misal total 512 = 384 + 128)
            max_src = other_args.max_source_length  # Jatah Input
            max_tgt = other_args.max_target_length  # Jatah Output
            
            for i in range(len(inputs)):
                # 1. Tokenisasi Input (Tanpa Padding)
                full_prompt = f"{instruct}{inputs[i]}\nSummary: "
                tokenized_src = tokenizer(
                    full_prompt,
                    max_length=max_src,
                    truncation=True,
                    add_special_tokens=True
                )["input_ids"]
            
                # 2. Tokenisasi Output (Tanpa Padding)
                tokenized_tgt = tokenizer(
                    targets[i] + tokenizer.eos_token,
                    max_length=max_tgt,
                    truncation=True,
                    add_special_tokens=False
                )["input_ids"]
            
                # --- PROSES PADDING ---
            
                # A. Input Block: Left Padding
                # [PAD, PAD, ..., Prompt, Dialogue]
                num_pad_src = max_src - len(tokenized_src)
                src_block = ([tokenizer.pad_token_id] * num_pad_src) + tokenized_src
                src_mask = ([0] * num_pad_src) + ([1] * len(tokenized_src))
            
                # B. Output Block: Right Padding
                # [Summary, EOS, PAD, PAD, ...]
                num_pad_tgt = max_tgt - len(tokenized_tgt)
                tgt_block = tokenized_tgt + ([tokenizer.pad_token_id] * num_pad_tgt)
                tgt_mask = ([1] * len(tokenized_tgt)) + ([0] * num_pad_tgt)
            
                # --- PENGGABUNGAN ---
            
                # Gabungkan menjadi satu sequence utuh
                input_ids = src_block + tgt_block
                attention_mask = src_mask + tgt_mask
                
                # Labels: Seluruh blok input adalah -100, 
                # di blok output hanya Summary yang asli, sisanya (pad) adalah -100
                labels = ([-100] * max_src) + tokenized_tgt + ([-100] * num_pad_tgt)

                all_input_ids.append(input_ids)
                all_labels.append(labels)
                all_attention_masks.append(attention_mask)
            
            return {
                "input_ids": all_input_ids,
                "labels": all_labels,
                "attention_mask": all_attention_masks
            }

    data_files = {}
    if other_args.test_file is not None:
        data_files["test"] = os.path.join(other_args.data_dir, other_args.test_file)
        extension = other_args.test_file.split(".")[-1]
    if other_args.validation_file is not None:
        data_files["val"] = os.path.join(other_args.data_dir, other_args.validation_file)
        extension = other_args.validation_file.split(".")[-1]
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=None,
    )

    ori_validation_dataset = raw_datasets["val"]
    with training_args.main_process_first(desc="validation dataset map pre-processing"):
        ori_validation_dataset = ori_validation_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=4,
            remove_columns=None,
            load_from_cache_file=False,
            desc="Running tokenizer on validation dataset",
        )

    test_dataset = raw_datasets["test"]
    with training_args.main_process_first(desc="prediction dataset map pre-processing"):
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=4,
            remove_columns=None,
            load_from_cache_file=False,
            desc="Running tokenizer on test dataset",
        )

    data_collator = DataCollatorForContrastive(tokenizer, model=model, is_decoder_only=(not model.config.is_encoder_decoder), max_length = other_args.max_source_length, max_tgt = other_args.max_target_length)

    label_pad_token_id = -100 if other_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if model.config.is_encoder_decoder:
        data_collator_for_val_test = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
        # data_collator_for_val_test = default_data_collator

    else:
        data_collator_for_val_test = default_data_collator
        

    # custom_wandb_callback = CustomWandbCallback()

    metric = evaluate.load("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
    
        import numpy as np
    
        # Handle tuple output (kadang dari Trainer)
        if isinstance(preds, tuple):
            preds = preds[0]
    
        # =========================
        # 1. HANDLE SHAPE (2D vs 3D)
        # =========================
        # (batch, num_candidates, seq_len) → flatten
        if preds.ndim == 3:
            preds = preds.reshape(-1, preds.shape[-1])
    
        if labels.ndim == 3:
            labels = labels.reshape(-1, labels.shape[-1])
    
        # =========================
        # 2. HANDLE LABEL MASKING
        # =========================
        if other_args.ignore_pad_token_for_loss:
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
        # =========================
        # 3. DECODE
        # =========================

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
        # =========================
        # 4. POST-PROCESS
        # =========================
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    
        # =========================
        # 5. METRIC (ROUGE)
        # =========================
        result = metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
    
        return result

    for name, param in model.named_parameters():
        if "lora" in name:
            print(f"{name}: {param.requires_grad}")

    training_args.looging_strategy = 'steps'
    training_args.logging_steps = 1
    training_args.do_eval = False
    training_args.do_predict = False
    print(training_args)

    if model.config.is_encoder_decoder:
        trainer = SeqUnlikelihoodTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            # eval_dataset=dev_dataset,  # monitor ROUGE
            # ori_validation_dataset=ori_validation_dataset,  # we generate summaries on the original validation set
            # test_dataset=test_dataset,
            data_collator=data_collator,
            data_collator_for_val_test=data_collator_for_val_test,
            tokenizer=tokenizer,
            # callbacks=[custom_wandb_callback],
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
            ori_dev_test_batch_size=other_args.ori_dev_test_batch_size
        )
    else:
        trainer = CausalUnlikelihoodTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            # eval_dataset=dev_dataset,  # monitor ROUGE
            # ori_validation_dataset=ori_validation_dataset,  # we generate summaries on the original validation set
            # test_dataset=test_dataset,
            data_collator=data_collator,
            data_collator_for_val_test=data_collator_for_val_test,
            tokenizer=tokenizer,
            callbacks=[],
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
            ori_dev_test_batch_size=other_args.ori_dev_test_batch_size,
        )

    trainer.train(resume_from_checkpoint=other_args.resume_from_ckpt)

    # Training
    # if training_args.do_train:
    #     checkpoint = None
    #     last_checkpoint = None
    #     if training_args.resume_from_checkpoint is not None:
    #         checkpoint = training_args.resume_from_checkpoint
    #     elif last_checkpoint is not None:
    #         checkpoint = last_checkpoint
        
    #     trainer.save_model()  # Saves the tokenizer too for easy upload

    #     metrics = train_result.metrics

    #     trainer.log_metrics("train", metrics)
    #     trainer.save_metrics("train", metrics)
    #     trainer.save_state()

    # # Evaluation
    # results = {}
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #     metrics = trainer.evaluate(metric_key_prefix="eval")

    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    # if training_args.do_predict:
    #     logger.info("*** Predict ***")

    #     predict_results = trainer.predict(test_dataset, metric_key_prefix="predict")
    #     metrics = predict_results.metrics

    #     trainer.log_metrics("predict", metrics)
    #     trainer.save_metrics("predict", metrics)

    #     if trainer.is_world_process_zero():
    #         if training_args.predict_with_generate:
    #             predictions = predict_results.predictions
    #             predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    #             predictions = tokenizer.batch_decode(
    #                 predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    #             )
    #             predictions = [pred.strip() for pred in predictions]
    #             output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
    #             with open(output_prediction_file, "w", encoding="utf-8") as writer:
    #                 writer.write("\n".join(predictions))

    # kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    # if data_args.dataset_name is not None:
    #     kwargs["dataset_tags"] = data_args.dataset_name
    #     if data_args.dataset_config_name is not None:
    #         kwargs["dataset_args"] = data_args.dataset_config_name
    #         kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
    #     else:
    #         kwargs["dataset"] = data_args.dataset_name

    # if data_args.lang is not None:
    #     kwargs["language"] = data_args.lang

    # if training_args.push_to_hub:
    #     trainer.push_to_hub(**kwargs)
    # else:
    #     trainer.create_model_card(**kwargs)

    


if __name__ == '__main__':
    main()