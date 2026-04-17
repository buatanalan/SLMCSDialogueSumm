import json
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import Any, Optional, List, Dict
from transformers import PegasusTokenizerFast, PegasusForConditionalGeneration


def load_jsonl(filename):
    records = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            records.append(json.loads(line))
    return records

class ContrastiveMultiDataset(Dataset):
    def __init__(self, filename, tokenizer, max_src_len=512, max_tgt_len=128,
                 source_prefix='', policy="setwise"):
        if filename.endswith('.jsonl'):
            self.data = load_jsonl(filename)
        else:
            self.data = json.load(open(filename, 'r', encoding='utf-8'))
            
        self.tokenizer = tokenizer
        self.policy = policy
        self.dataset = []


        for record in self.data:
            # 1. Tokenisasi Source (Dialogue saja)
            dialogue = source_prefix + record['dialogue']
            tokenized_source = tokenizer(
                dialogue, 
                truncation=True, 
                max_length=max_src_len, 
                add_special_tokens=True # Memberikan BOS token di awal dialogue
            )
            
            source_data = {
                'input_ids': tokenized_source.input_ids,
                'attention_mask': tokenized_source.attention_mask
            }

            # 2. Tokenisasi Target (Summary saja - tanpa dialogue)
            # add_special_tokens=False agar tidak ada BOS token di tengah sequence
            def tokenize_target(text):
                return tokenizer(
                    text, 
                    truncation=True, 
                    max_length=max_tgt_len, 
                    add_special_tokens=False 
                ).input_ids

            ref_summ_ids = tokenize_target(record['summary'])
            augmented = record.get('summaries', [])
            
            pos_ids_list = [tokenize_target(x['pos']) for x in augmented if 'pos' in x]
            neg_ids_list = [tokenize_target(x['neg']) for x in augmented if 'neg' in x]

            if self.policy == "setwise":
                # Masukkan original reference ke pool positif
                positive_pool = pos_ids_list + [ref_summ_ids]
                
                # Kita simpan satu record per original dialogue
                # DataCollator akan bertugas meratakan (flatten) semua kombinasi ini
                self.dataset.append({
                    'source': source_data,
                    'positive_summaries': positive_pool,
                    'negative_summaries': neg_ids_list,
                    'index_of_ref_summ': len(positive_pool) - 1 # Indeks ref_summ asli
                })
            
            elif self.policy == "pairwise":
                # Untuk pairwise, kita pastikan ada pasangan pos-neg yang valid
                for x in augmented:
                    if 'pos' in x and 'neg' in x:
                        p_ids = tokenize_target(x['pos'])
                        n_ids = tokenize_target(x['neg'])
                        
                        self.dataset.append({
                            'source': source_data,
                            'positive_summaries': [p_ids, ref_summ_ids],
                            'negative_summaries': [n_ids],
                            'index_of_ref_summ': 1
                        })

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

@dataclass
class DataCollatorForContrastive:
    tokenizer: Any
    model: Any
    is_decoder_only: bool = True
    label_pad_token_id: int = -100
    padding: bool = True

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        all_input_ids = []
        all_labels = []
        all_attention_masks = []
        
        src_select_indices = []
        pos_neg_labels = []
        ref_summ_marker = []

        # Token separator (misal: "\nSummary: ")
        sep_ids = self.tokenizer.encode("\nSummary: ", add_special_tokens=False)
        print(len(features))

        for i, record in enumerate(features):
            source = record['source']
            src_ids = source['input_ids']
            src_mask = source['attention_mask']

            # Kelompokkan semua kandidat ringkasan
            # (list_ids, is_positive_flag)
            summary_groups = [
                (record['positive_summaries'], 1),
                (record['negative_summaries'], 0)
            ]

            for summaries, is_pos in summary_groups:
                for j, tgt_ids in enumerate(summaries):
                    # Tambahkan EOS token jika belum ada
                    if not tgt_ids or tgt_ids[-1] != self.tokenizer.eos_token_id:
                        tgt_ids = tgt_ids + [self.tokenizer.eos_token_id]

                    if self.is_decoder_only:
                        # Logika Decoder-Only (Llama/Qwen): Gabungkan Source + Sep + Target
                        input_ids = src_ids + sep_ids + tgt_ids
                        
                        # Labels: Masking source dan separator dengan -100
                        mask_len = len(src_ids) + len(sep_ids)
                        label = [self.label_pad_token_id] * mask_len + tgt_ids
                        
                        attention_mask = src_mask + [1] * (len(sep_ids) + len(tgt_ids))
                    else:
                        # Logika Seq2Seq (BART/Pegasus): Input hanya source, Label hanya target
                        input_ids = src_ids
                        label = tgt_ids
                        attention_mask = src_mask

                    all_input_ids.append({"input_ids": input_ids})
                    all_labels.append(label)
                    all_attention_masks.append(attention_mask)
                    
                    src_select_indices.append(i)
                    pos_neg_labels.append(is_pos)
                    
                    # Tandai jika ini adalah reference summary asli
                    if is_pos == 1 and j == record['index_of_ref_summ']:
                        ref_summ_marker.append(1)
                    else:
                        ref_summ_marker.append(0)

        # Padding Input IDs dan Attention Mask
        batch = self.tokenizer.pad(
            all_input_ids,
            padding=self.padding,
            return_tensors="pt"
        )        
        batch_size = len(features)
        num_candidates = len(all_input_ids) // batch_size
        # Manual Padding untuk Labels (karena tokenizer.pad tidak mendukung list of labels)
        if self.is_decoder_only:
            max_len = batch['input_ids'].size(1)
        else:
            max_len = max([len(label) for label in all_labels])
        padded_labels = []
        for l in all_labels:
            remainder = [self.label_pad_token_id] * (max_len - len(l))
            if self.tokenizer.padding_side == "right":
                padded_labels.append(l + remainder)
            else:
                padded_labels.append(remainder + l)
                
        labels = torch.tensor(padded_labels, dtype=torch.long)


        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            batch['decoder_input_ids'] = self.model.prepare_decoder_input_ids_from_labels(labels=labels)
            batch['decoder_input_ids'] = batch['decoder_input_ids'].view(batch_size, num_candidates, -1)

        batch['labels'] = torch.tensor(padded_labels, dtype=torch.long)
        batch['src_select_indices'] = torch.tensor(src_select_indices, dtype=torch.long)
        batch['pos_neg'] = torch.tensor(pos_neg_labels, dtype=torch.long)
        batch['ref_summ_idx'] = torch.tensor(ref_summ_marker, dtype=torch.long)

        # === reshape input_ids & attention_mask ===
        batch["input_ids"] = batch["input_ids"].view(batch_size, num_candidates, -1)
        batch["attention_mask"] = batch["attention_mask"].view(batch_size, num_candidates, -1)
        
        # === reshape labels ===
        labels = torch.tensor(padded_labels, dtype=torch.long)
        batch["labels"] = labels.view(batch_size, num_candidates, -1)
        
        # === reshape metadata ===
        batch["src_select_indices"] = torch.tensor(src_select_indices, dtype=torch.long).view(batch_size, num_candidates)
        batch["pos_neg"] = torch.tensor(pos_neg_labels, dtype=torch.long).view(batch_size, num_candidates)
        batch["ref_summ_idx"] = torch.tensor(ref_summ_marker, dtype=torch.long).view(batch_size, num_candidates)

        return batch