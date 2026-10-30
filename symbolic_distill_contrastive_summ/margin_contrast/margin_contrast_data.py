"""
margin_contrast_data.py  (FIXED)

Perubahan utama:
- FIX #4: Format prompt training konsisten dengan validation
          (tambahkan "\nSummary: " di akhir source untuk decoder-only)
- FIX #5: DataCollatorForContrastive dirombak total:
          * Tidak panggil tokenizer.pad() lagi (manual pad semua)
          * max_tgt TANPA +1 (konsisten dengan preprocess_function)
          * Attention mask, input_ids, labels selalu sinkron panjangnya
          * Helper terpisah untuk seq2seq vs causal LM
"""

import json
import torch
import pathlib
from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import Any, List, Dict


def load_jsonl(filename):
    records = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            records.append(json.loads(line))
    return records


class ContrastiveMultiDataset(Dataset):
    """
    Dataset untuk training contrastive (positive + negative summaries).
    Mendukung BOTH seq2seq (BART/T5/Pegasus) dan causal LM (Qwen/Llama).
    """

    def __init__(
        self,
        filename,
        tokenizer,
        max_src_len=512,
        max_tgt_len=128,
        source_prefix='',
        policy="setwise",
        is_decoder_only=False,
        prompt="",
    ):
        if filename.endswith('.jsonl'):
            self.data = load_jsonl(filename)
        else:
            self.data = json.load(open(filename, 'r', encoding='utf-8'))

        self.tokenizer = tokenizer
        self.policy = policy
        self.is_decoder_only = is_decoder_only
        self.dataset = []

        # ====================================================
        # Load prompt template
        # ====================================================
        prompt_path = pathlib.Path(__file__).parent.parent.parent / f'generate_dataset/{prompt}'
        try:
            with open(prompt_path, "r") as f:
                data = json.loads(f.read())
                instruct = data["prompt_positive"]
        except Exception as e:
            print(f"Failed to load prompt !!! {e}")
            instruct = "Summarize the following dialogue:\n"

        # ====================================================
        # FIX #4: Format prompt source konsisten dengan validation
        # ====================================================
        # Untuk decoder-only (Qwen/Llama), format input adalah:
        #   "{instruct}{dialogue}\nSummary: "
        # Format INI HARUS SAMA dengan preprocess_function di train.py.
        # Untuk seq2seq (BART/T5/Pegasus), gunakan source_prefix biasa.
        # ====================================================

        print(f"vocab_size       : {tokenizer.vocab_size}")
        print(f"len(tokenizer)   : {len(tokenizer)}")
        print(f"tokenizer class  : {type(tokenizer).__name__}")
        print(f"name_or_path     : {tokenizer.name_or_path}")
        print(f"is_fast          : {tokenizer.is_fast}")
        print(f"is_decoder_only  : {is_decoder_only}")
        print(f"max_src_len      : {max_src_len}")
        print(f"max_tgt_len      : {max_tgt_len}")

        for record in self.data:
            # ===== 1. Tokenisasi Source =====
            if is_decoder_only:
                # Format konsisten dengan validation
                source_text = f"{instruct}{record['dialogue']}\nSummary: "
            else:
                source_text = source_prefix + record['dialogue']

            tokenized_source = tokenizer(
                source_text,
                truncation=True,
                max_length=max_src_len,
                add_special_tokens=True,
            )
            source_data = {
                'input_ids': tokenized_source.input_ids,
                'attention_mask': tokenized_source.attention_mask,
            }

            # ===== 2. Tokenisasi Target =====
            # add_special_tokens=False agar tidak ada BOS di tengah sequence
            def tokenize_target(text):
                return tokenizer(
                    text,
                    truncation=True,
                    max_length=max_tgt_len,
                    add_special_tokens=False,
                ).input_ids

            ref_summ_ids = tokenize_target(record['summary'])
            augmented = record.get('summaries', [])
            pos_ids_list = [tokenize_target(x['pos']) for x in augmented if 'pos' in x]
            neg_ids_list = [tokenize_target(x['neg']) for x in augmented if 'neg' in x]

            # ===== 3. Build records sesuai policy =====
            if self.policy == "setwise":
                # Reference dimasukkan sebagai positif terakhir
                positive_pool = pos_ids_list + [ref_summ_ids]
                self.dataset.append({
                    'source': source_data,
                    'positive_summaries': positive_pool,
                    'negative_summaries': neg_ids_list,
                    'index_of_ref_summ': len(positive_pool) - 1,
                })

            elif self.policy == "pairwise":
                for x in augmented:
                    if 'pos' in x and 'neg' in x:
                        p_ids = tokenize_target(x['pos'])
                        n_ids = tokenize_target(x['neg'])
                        self.dataset.append({
                            'source': source_data,
                            'positive_summaries': [p_ids, ref_summ_ids],
                            'negative_summaries': [n_ids],
                            'index_of_ref_summ': 1,
                        })

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


# ============================================================
# FIX #5: DataCollatorForContrastive — ditulis ulang total
# ============================================================
@dataclass
class DataCollatorForContrastive:
    """
    Data collator untuk contrastive learning.

    Untuk causal LM:
      * Source di-LEFT-pad ke `max_length` (max_src)
      * Target di-RIGHT-pad ke `max_tgt`
      * Sequence final: [PAD..PAD, src_tokens, tgt_tokens, PAD..PAD]
      * Label: [-100..-100, -100..-100, tgt_tokens, -100..-100]
      * TIDAK panggil tokenizer.pad() (manual pad ke panjang seragam)

    Untuk seq2seq:
      * Source dipakai apa adanya (dynamic pad via tokenizer.pad)
      * Labels dipad manual untuk handle batch
    """
    tokenizer: Any
    model: Any
    is_decoder_only: bool = True
    label_pad_token_id: int = -100
    padding: bool = True
    max_length: int = None    # = max_src
    max_tgt: int = None       # = max_tgt (TANPA +1)

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        if self.is_decoder_only:
            return self._collate_causal(features)
        else:
            return self._collate_seq2seq(features)

    # ============================================================
    # CAUSAL LM (Qwen / Llama)
    # ============================================================
    def _collate_causal(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        tokenizer = self.tokenizer
        pad_id = tokenizer.pad_token_id
        eos_id = tokenizer.eos_token_id

        max_src = self.max_length
        max_tgt = self.max_tgt   # tanpa +1

        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        src_select_indices = []
        pos_neg_labels = []
        ref_summ_marker = []

        for i, record in enumerate(features):
            src_ids = list(record['source']['input_ids'])
            print(src_ids)

            # Defensive: kalau src kepanjangan, truncate
            if len(src_ids) > max_src:
                src_ids = src_ids[:max_src]

            summary_groups = [
                (record['positive_summaries'], 1),
                (record['negative_summaries'], 0),
            ]

            for summaries, is_pos in summary_groups:
                for j, tgt_ids in enumerate(summaries):
                    tgt_ids = list(tgt_ids)

                    # Tambah EOS jika belum ada
                    if not tgt_ids or tgt_ids[-1] != eos_id:
                        tgt_ids.append(eos_id)

                    # Defensive: kalau tgt kepanjangan, truncate (pastikan diakhiri EOS)
                    if len(tgt_ids) > max_tgt:
                        tgt_ids = tgt_ids[:max_tgt]
                        tgt_ids[-1] = eos_id

                    # ----- A. Source block: LEFT-pad -----
                    num_pad_src = max_src - len(src_ids)
                    src_block = [pad_id] * num_pad_src + src_ids
                    src_mask = [0] * num_pad_src + [1] * len(src_ids)

                    # ----- B. Target block: RIGHT-pad -----
                    num_pad_tgt = max_tgt - len(tgt_ids)
                    tgt_block = tgt_ids + [pad_id] * num_pad_tgt
                    tgt_mask = [1] * len(tgt_ids) + [0] * num_pad_tgt

                    # ----- Concat -----
                    input_ids = src_block + tgt_block
                    attention_mask = src_mask + tgt_mask
                    label = (
                        [self.label_pad_token_id] * max_src
                        + tgt_ids
                        + [self.label_pad_token_id] * num_pad_tgt
                    )

                    # Sanity check — semua harus sama panjang
                    seq_len = max_src + max_tgt
                    assert (
                        len(input_ids) == len(attention_mask) == len(label) == seq_len
                    ), (
                        f"Length mismatch: input={len(input_ids)}, "
                        f"mask={len(attention_mask)}, label={len(label)}, "
                        f"expected={seq_len}"
                    )

                    all_input_ids.append(input_ids)
                    all_attention_masks.append(attention_mask)
                    all_labels.append(label)
                    src_select_indices.append(i)
                    pos_neg_labels.append(is_pos)

                    is_ref = (is_pos == 1 and j == record['index_of_ref_summ'])
                    ref_summ_marker.append(1 if is_ref else 0)

        # ----- Konversi langsung ke tensor (TIDAK panggil tokenizer.pad) -----
        batch_size = len(features)
        num_candidates = len(all_input_ids) // batch_size
        seq_len = max_src + max_tgt

        input_ids_t = torch.tensor(all_input_ids, dtype=torch.long)
        attn_t = torch.tensor(all_attention_masks, dtype=torch.long)
        labels_t = torch.tensor(all_labels, dtype=torch.long)

        batch = {
            'input_ids': input_ids_t.view(batch_size, num_candidates, seq_len),
            'attention_mask': attn_t.view(batch_size, num_candidates, seq_len),
            'labels': labels_t.view(batch_size, num_candidates, seq_len),
            'src_select_indices': torch.tensor(
                src_select_indices, dtype=torch.long
            ).view(batch_size, num_candidates),
            'pos_neg': torch.tensor(
                pos_neg_labels, dtype=torch.long
            ).view(batch_size, num_candidates),
            'ref_summ_idx': torch.tensor(
                ref_summ_marker, dtype=torch.long
            ).view(batch_size, num_candidates),
        }
        return batch

    # ============================================================
    # SEQ2SEQ (BART / T5 / Pegasus)
    # ============================================================
    def _collate_seq2seq(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        tokenizer = self.tokenizer

        all_src = []
        all_labels = []
        src_select_indices = []
        pos_neg_labels = []
        ref_summ_marker = []

        for i, record in enumerate(features):
            src_ids = record['source']['input_ids']
            src_mask = record['source']['attention_mask']

            summary_groups = [
                (record['positive_summaries'], 1),
                (record['negative_summaries'], 0),
            ]

            for summaries, is_pos in summary_groups:
                for j, tgt_ids in enumerate(summaries):
                    tgt_ids = list(tgt_ids)
                    if not tgt_ids or tgt_ids[-1] != tokenizer.eos_token_id:
                        tgt_ids.append(tokenizer.eos_token_id)

                    all_src.append({
                        "input_ids": src_ids,
                        "attention_mask": src_mask,
                    })
                    all_labels.append(tgt_ids)
                    src_select_indices.append(i)
                    pos_neg_labels.append(is_pos)
                    ref_summ_marker.append(
                        1 if (is_pos == 1 and j == record['index_of_ref_summ']) else 0
                    )

        # Pad source via tokenizer (dynamic)
        batch_src = tokenizer.pad(
            all_src,
            padding=True,
            return_tensors="pt",
        )

        # Pad labels manual
        max_label_len = max(len(l) for l in all_labels)
        padding_side = getattr(tokenizer, 'padding_side', 'right')
        padded_labels = []
        for l in all_labels:
            remainder = [self.label_pad_token_id] * (max_label_len - len(l))
            if padding_side == "right":
                padded_labels.append(l + remainder)
            else:
                padded_labels.append(remainder + l)

        labels = torch.tensor(padded_labels, dtype=torch.long)

        batch_size = len(features)
        num_candidates = len(all_labels) // batch_size

        decoder_input_ids = None
        if self.model is not None and hasattr(
            self.model, "prepare_decoder_input_ids_from_labels"
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=labels
            )

        batch = {
            'input_ids': batch_src['input_ids'].view(batch_size, num_candidates, -1),
            'attention_mask': batch_src['attention_mask'].view(batch_size, num_candidates, -1),
            'labels': labels.view(batch_size, num_candidates, -1),
            'src_select_indices': torch.tensor(
                src_select_indices, dtype=torch.long
            ).view(batch_size, num_candidates),
            'pos_neg': torch.tensor(
                pos_neg_labels, dtype=torch.long
            ).view(batch_size, num_candidates),
            'ref_summ_idx': torch.tensor(
                ref_summ_marker, dtype=torch.long
            ).view(batch_size, num_candidates),
        }
        if decoder_input_ids is not None:
            batch['decoder_input_ids'] = decoder_input_ids.view(
                batch_size, num_candidates, -1
            )
        return batch