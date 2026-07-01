from dataclasses import dataclass

import torch
from transformers import PegasusForConditionalGeneration, T5ForConditionalGeneration, BartForConditionalGeneration
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutputWithPast

from typing import *
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
from transformers import LlamaForCausalLM, LlamaConfig, AutoModelForCausalLM

# FIX #1: Import Qwen2ForCausalLM agar bisa di-inherit (bukan AutoModelForCausalLM)
# Butuh transformers >= 4.37. Kalau versi lebih lama, lihat catatan di bawah class.
try:
    from transformers import Qwen2ForCausalLM
except ImportError:
    # Fallback untuk transformers lama: gunakan AutoModel + trust_remote_code.
    # Tapi STRONGLY RECOMMENDED upgrade ke transformers >= 4.45.
    Qwen2ForCausalLM = None
    print("WARNING: Qwen2ForCausalLM tidak tersedia di transformers versi Anda. "
          "Upgrade: pip install --upgrade transformers>=4.45")


@dataclass
class Seq2SeqContrastiveOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


class BartForContrastive(BartForConditionalGeneration):
    def forward(
            self,
            input_ids=None,  # batch_size x seq_len
            pos_neg=None,  # 1 x N binary tensor, 1 means positive target, 0 means negative target
            ref_summ_idx=None,  # 1 x N binary tensor, 1 means reference summary, 0 means augmented targets (positive or negative)
            src_select_indices=None,  # 1 x N, source index of targets
            attention_mask=None,  # batch_size x seq_len
            decoder_input_ids=None,  # N x seq_len
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # ============
        #  evaluation
        # ============
        # print("decoder")
        # print(input_ids.shape)
        # print(decoder_input_ids.shape)
        
        if src_select_indices is None:
            # print("BART - [Evaluation] forward mode.")
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

        # print("BART - [Training] forward mode.")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # some boolean values
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        ###################
        #     Encoder     #
        ###################
    
        if encoder_outputs is None:
            # encode the dialogues
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=False,
            )

            # we need to record the index of dialogue in this batch, for each target summary (either positive or negative)
            # for example, if we have 4 dialogues in a batch (batch_size=4), 16 target summaries (for each dialogue we have
            # 2 positive and 2 negative summaries.
            # dlg_idx: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3] means the first 4 target summaries are matching the
            # first dialogue in this batch
            # pos_neg: [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0] means the first 2 target summaries for the first
            # dialogue are positive summaries (1 for positive, 0 for negative)

            # we replicate encoded dialogues multiple times, to have a 1-1 match between dialogues and target summaries
            # in this case we only encode each dialogue once, to save some computational overhead
            encoder_outputs = tuple([ele.index_select(0, src_select_indices) for ele in encoder_outputs])
            input_ids = input_ids.index_select(0, src_select_indices)
            attention_mask = attention_mask.index_select(0, src_select_indices)

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        ###################
        #     Decoder     #
        ###################
        
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # shared.weight is from Embedding layer
        decoder_last_hidden_state = outputs[0]  # N x seq_len x hidden_dim (e.g., 1024 for bart large)
        # N x seq_len x vocab_size
        lm_logits = F.linear(decoder_last_hidden_state, self.model.shared.weight, bias=self.final_logits_bias)

        return Seq2SeqContrastiveOutput(
            loss=torch.FloatTensor([-1]),  # a placeholder to make sure prediction step doesn't break
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions
        )


class PegasusForContrastive(PegasusForConditionalGeneration):
    def forward(
            self,
            input_ids=None,  # batch_size x seq_len
            pos_neg=None,  # 1 x N binary tensor, 1 means positive target, 0 means negative target
            ref_summ_idx=None,  # 1 x N binary tensor, 1 means reference summary, 0 means augmented targets (positive or negative)
            src_select_indices=None,  # 1 x N, source index of targets
            attention_mask=None,  # batch_size x seq_len
            decoder_input_ids=None,  # N x seq_len
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # ============
        #  evaluation
        # ============
        if src_select_indices is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # some boolean values
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        ###################
        #     Encoder     #
        ###################
        if encoder_outputs is None:
            # encode the dialogues
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=False,
            )

            # we need to record the index of dialogue in this batch, for each target summary (either positive or negative)
            # for example, if we have 4 dialogues in a batch (batch_size=4), 16 target summaries (for each dialogue we have
            # 2 positive and 2 negative summaries.
            # dlg_idx: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3] means the first 4 target summaries are matching the
            # first dialogue in this batch
            # pos_neg: [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0] means the first 2 target summaries for the first
            # dialogue are positive summaries (1 for positive, 0 for negative)

            # we replicate encoded dialogues multiple times, to have a 1-1 match between dialogues and target summaries
            # in this case we only encode each dialogue once, to save some computational overhead
            encoder_outputs = tuple([ele.index_select(0, src_select_indices) for ele in encoder_outputs])
            input_ids = input_ids.index_select(0, src_select_indices)
            attention_mask = attention_mask.index_select(0, src_select_indices)

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        ###################
        #     Decoder     #
        ###################
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # shared.weight is from Embedding layer
        decoder_last_hidden_state = outputs[0]  # N x seq_len x hidden_dim (e.g., 1024 for bart large)
        # N x seq_len x vocab_size
        lm_logits = F.linear(decoder_last_hidden_state, self.model.shared.weight, bias=self.final_logits_bias)

        return Seq2SeqContrastiveOutput(
            loss=torch.FloatTensor([-1]),  # a placeholder to make sure prediction step doesn't break
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions
        )


class T5ForContrastive(T5ForConditionalGeneration):
    def forward(
            self,
            input_ids=None,  # batch_size x seq_len
            pos_neg=None,  # 1 x N binary tensor, 1 means positive target, 0 means negative target
            ref_summ_idx=None,  # 1 x N binary tensor, 1 means reference summary, 0 means augmented targets (positive or negative)
            src_select_indices=None,  # 1 x N, source index of targets
            attention_mask=None,  # batch_size x seq_len
            decoder_input_ids=None,  # N x seq_len
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # ============
        #  evaluation
        # ============
        if src_select_indices is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                head_mask=head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

        # ==========
        #  training
        # ==========
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # some boolean values
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        ###################
        #     Encoder     #
        ###################
        # print_gpu_usage("GPU Memory (before encoding the source)")
        if encoder_outputs is None:
            # encode the dialogues
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=False,
            )

            # we need to record the index of dialogue in this batch, for each target summary (either positive or negative)
            # for example, if we have 4 dialogues in a batch (batch_size=4), 16 target summaries (for each dialogue we have
            # 2 positive and 2 negative summaries.
            # dlg_idx: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3] means the first 4 target summaries are matching the
            # first dialogue in this batch
            # pos_neg: [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0] means the first 2 target summaries for the first
            # dialogue are positive summaries (1 for positive, 0 for negative)

            # we replicate encoded dialogues multiple times, to have a 1-1 match between dialogues and target summaries
            # in this case we only encode each dialogue once, to save some computational overhead
            encoder_outputs = tuple([ele.index_select(0, src_select_indices) for ele in encoder_outputs])
            input_ids = input_ids.index_select(0, src_select_indices)
            attention_mask = attention_mask.index_select(0, src_select_indices)

        # we wrap the encoder output in a BaseModelOutput
        if not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        if labels is not None and decoder_input_ids is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        ###################
        #     Decoder     #
        ###################
        # print("=== size of [decoder input ids]", decoder_input_ids.size())
        # print_gpu_usage("GPU Memory (after encoding the source, before decoding)")
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        # shared.weight is from Embedding layer
        decoder_last_hidden_state = decoder_outputs[0]  # N x seq_len x hidden_dim (e.g., 1024 for bart large)
        # N x seq_len x vocab_size
        # lm_logits = F.linear(decoder_last_hidden_state, self.model.shared.weight, bias=self.final_logits_bias)
        lm_logits = self.lm_head(decoder_last_hidden_state)
        # print_gpu_usage("GPU Memory (after decoding)")
        return Seq2SeqContrastiveOutput(
            loss=torch.FloatTensor([-1]),  # a placeholder to make sure prediction step doesn't break
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions
        )

class LlamaForContrastive(LlamaForCausalLM):
    def forward(
        self,
        input_ids: torch.LongTensor = None, # Context/Source batch
        pos_neg: torch.Tensor = None, # 1 x N binary
        ref_summ_idx: torch.Tensor = None, # 1 x N binary
        src_select_indices: torch.LongTensor = None, # Index pemetaan context ke target
        attention_mask: Optional[torch.Tensor] = None,
        # Di Llama, labels biasanya digunakan untuk menghitung loss causal
        labels: Optional[torch.LongTensor] = None, 
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if src_select_indices is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Forward pass ke model Llama dasar
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        # Llama menggunakan lm_head (Linear layer) untuk proyeksi ke vocab
        lm_logits = self.lm_head(hidden_states)

        # Output disesuaikan dengan struktur CausalLM
        return CausalLMOutputWithPast(
            loss=torch.tensor(-1.0, device=lm_logits.device), # Placeholder
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# ============================================================
# FIX #1: QwenForContrastive harus inherit dari Qwen2ForCausalLM
#         (bukan AutoModelForCausalLM yang adalah factory class!)
# ============================================================
# Catatan: kalau Qwen2ForCausalLM None (transformers lama),
# fallback ke LlamaForCausalLM yang struktur internalnya mirip.
_QwenBase = Qwen2ForCausalLM if Qwen2ForCausalLM is not None else LlamaForCausalLM


class QwenForContrastive(_QwenBase):
    """
    Qwen2 untuk contrastive learning.
    Mendukung dua mode:
      1. src_select_indices=None  -> forward standar (untuk generate/eval)
      2. src_select_indices ada    -> forward contrastive (untuk training)
    """

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pos_neg: torch.Tensor = None,
        ref_summ_idx: torch.Tensor = None,
        src_select_indices: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # Sanity check (sekali waktu) -- bisa dihapus setelah training stabil
        if not hasattr(self, "_custom_forward_called"):
            print(">>> [QwenForContrastive] custom forward DIPANGGIL <<<")
            self._custom_forward_called = True

        # ============================================================
        # JALUR 1: EVALUASI / GENERATE  (no custom contrastive params)
        # ============================================================
        if src_select_indices is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # ============================================================
        # JALUR 2: TRAINING CONTRASTIVE
        # Loss dihitung di trainer (_compute_loss), di sini hanya logits.
        # ============================================================
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        lm_logits = self.lm_head(hidden_states)

        # loss=None lebih aman daripada tensor(-1.0):
        #   - Trainer.compute_loss bisa mendeteksi dan skip
        #   - Tidak ada risiko nilai loss bocor ke logger
        return CausalLMOutputWithPast(
            loss=None,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )