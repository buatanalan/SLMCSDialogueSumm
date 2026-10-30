import os
import nltk
import numpy as np
import pathlib
import json

from dataclasses import dataclass, field

from datasets import load_dataset
import evaluate

from pair_contrast.callbacks import ContrastWandbCallback
from pair_contrast.pair_contrast_data import ContrastiveDataset, DataCollatorForContrastive
from pair_contrast.pair_contrast_model import PegasusForContrastive, BartForContrastive, T5ForContrastive, LlamaForContrastive
from pair_contrast.pair_contrast_trainer import ContrastiveTrainer, LlamaContrastiveTrainer
from transformers import PegasusTokenizerFast, Seq2SeqTrainingArguments, HfArgumentParser, BartTokenizerFast, \
    T5TokenizerFast, DataCollatorForSeq2Seq, LlamaTokenizerFast, AutoTokenizer, DataCollatorForLanguageModeling, default_data_collator


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
    parser.add_argument('--data_dir', default='')
    parser.add_argument('--model_type', type=str, choices=["bart", "pegasus", "t5", "llama", "phi", "qwen"])
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument(
        '--n_contrast_pair',
        type=int, default=1,
        help="number of positive and negative samples for contrastive loss. The pos/neg number are the same")
    parser.add_argument('--max_source_length', type=int)
    parser.add_argument('--max_target_length', type=int)
    parser.add_argument('--ignore_pad_token_for_loss', type=bool, default=True)
    parser.add_argument('--pad_to_max_length', action="store_true")
    parser.add_argument('--theta', type=float, help="Final Loss = LM loss + theta * contrast loss")
    parser.add_argument('--contrastive_train_file', type=str,
                        help="file for contrastive training set")
    parser.add_argument('--contrastive_dev_file', type=str,
                        help="file for contrastive development set")
    parser.add_argument('--validation_file', type=str,
                        help="file for original development set (instead of contrastive)")
    parser.add_argument('--test_file', type=str, help="file for original test set (instead of contrastive)")
    parser.add_argument('--ori_dev_test_batch_size', type=int, default=8)
    parser.add_argument('--prompt_path', type=str, default="prompt_formatted.json")
    parser.add_argument('--alpha', type=float, default=1)


    training_args, other_args = parser.parse_args_into_dataclasses()

    training_args.remove_unused_columns = False

    if other_args.model_type == "pegasus":
        model = PegasusForContrastive.from_pretrained(other_args.model_name_or_path)
        tokenizer = PegasusTokenizerFast.from_pretrained(other_args.model_name_or_path)
    elif other_args.model_type == "bart":
        model = BartForContrastive.from_pretrained(other_args.model_name_or_path)
        tokenizer = BartTokenizerFast.from_pretrained(other_args.model_name_or_path)
    elif other_args.model_type == "t5":
        model = T5ForContrastive.from_pretrained(other_args.model_name_or_path)
        tokenizer = T5TokenizerFast.from_pretrained(other_args.model_name_or_path)
    elif other_args.model_type == "llama":
        model = LlamaForContrastive.from_pretrained(other_args.model_name_or_path)
        tokenizer = LlamaTokenizerFast.from_pretrained(other_args.model_name_or_path)
    elif other_args.model_type == "qwen":
        model = LlamaForContrastive.from_pretrained(other_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(other_args.model_name_or_path)
    else:
        raise ValueError(f"Model type '{other_args.model_type}' not supported.")

    if training_args.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
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
        model.print_trainable_parameters()


    source_prefix = ''
    if other_args.model_type == "t5":
        source_prefix = "summarize: "

    train_dataset = ContrastiveDataset(
        os.path.join(other_args.data_dir, other_args.contrastive_train_file),
        tokenizer,
        other_args.max_source_length,
        other_args.max_target_length,
        source_prefix=source_prefix,
        use_original_ref=training_args.use_original_ref,
        n_contrast_pair=other_args.n_contrast_pair,
        is_decoder_only=(not model.config.is_encoder_decoder),
        prompt_path=other_args.prompt_path
    )

    valid_dataset = ContrastiveDataset(
        os.path.join(other_args.data_dir, other_args.contrastive_dev_file),
        tokenizer,
        other_args.max_source_length,
        other_args.max_target_length,
        source_prefix=source_prefix,
        use_original_ref=training_args.use_original_ref,
        n_contrast_pair=other_args.n_contrast_pair,
        is_decoder_only=(not model.config.is_encoder_decoder),
        prompt_path=other_args.prompt_path
    )

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
            model_inputs = tokenizer(
                inputs, 
                max_length=other_args.max_source_length, 
                padding=padding, 
                truncation=True
            )

            labels = tokenizer(
                text_target=targets, 
                max_length=other_args.max_target_length, 
                padding=padding, 
                truncation=True
            )

            if padding == "max_length" and other_args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] 
                    for label in labels["input_ids"]
                ]
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        # --- LOGIKA CAUSAL LM (Decoder-only seperti Llama/Qwen) ---
        else:
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

    print('Dataset Loaded.')

    # model = PegasusForContrastive.from_pretrained('google/pegasus-large')
    # tokenizer = PegasusTokenizerFast.from_pretrained('google/pegasus-large')

    data_collator = DataCollatorForContrastive(tokenizer, model=model, is_decoder_only=(not model.config.is_encoder_decoder), max_length=other_args.max_source_length, max_tgt=other_args.max_target_length)

    label_pad_token_id = -100 if other_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if model.config.is_encoder_decoder:
        data_collator_for_val_test = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
    else:
        data_collator_for_val_test = default_data_collator
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

    # log lm_loss, contrast_loss and total_loss in wandb
    wandb_callback = None

    if model.config.is_encoder_decoder:
        trainer = ContrastiveTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            theta=other_args.theta,
            # eval_dataset=valid_dataset,  # monitor ROUGE
            # ori_validation_dataset=ori_validation_dataset,  # we generate summaries on the original validation set
            # test_dataset=test_dataset,
            data_collator=data_collator,
            data_collator_for_val_test=data_collator_for_val_test,
            tokenizer=tokenizer,
            # callbacks=[wandb_callback],
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
            ori_dev_test_batch_size=other_args.ori_dev_test_batch_size
        )
    else:
        trainer = LlamaContrastiveTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            theta=other_args.theta,
            # eval_dataset=valid_dataset,  # monitor ROUGE
            # ori_validation_dataset=ori_validation_dataset,  # we generate summaries on the original validation set
            # test_dataset=test_dataset,
            data_collator=data_collator,
            data_collator_for_val_test=data_collator_for_val_test,
            tokenizer=tokenizer,
            # callbacks=[wandb_callback],
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
            ori_dev_test_batch_size=other_args.ori_dev_test_batch_size
        )

    trainer.train()


if __name__ == '__main__':
    main()