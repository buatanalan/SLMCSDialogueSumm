import argparse
import json
import torch
import gc
import os
import pathlib
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    GenerationConfig
)
from evaluate import load

# Import PeftModel untuk menangani LoRA
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluasi ROUGE & BERTScore untuk Model (Base/LoRA) Encoder-Decoder dan Decoder-only")
    parser.add_argument("--model_path", type=str, required=True, help="Folder model (atau folder adapter LoRA)")
    parser.add_argument("--base_model_path", type=str, default=None, help="Path ke Base Model (diperlukan jika menggunakan LoRA)")
    parser.add_argument("--input_file", type=str, required=True, help="File input data (JSON)")
    parser.add_argument("--metric_output", type=str, default="evaluation_metrics.json", help="File untuk menyimpan hasil skor")
    parser.add_argument("--prediction_output", type=str, default="generated_results.json", help="File untuk menyimpan hasil teks beserta skor per instance")
    parser.add_argument("--input_col", type=str, default="dialogue", help="Nama kolom input di JSON")
    parser.add_argument("--target_col", type=str, default="summary", help="Nama kolom ground truth di JSON")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--merge_lora", action="store_true", help="Merge weights LoRA untuk inference lebih cepat")
    parser.add_argument("--prompt_path", default="prompt_unformatted.json")
    parser.add_argument("--lang", type=str, default="en", help="Bahasa untuk BERTScore (default: en)")
    return parser.parse_args()

def compute_bertscore_manual(predictions, references, device="cpu"):
    model = SentenceTransformer("all-mpnet-base-v2", device=device)
    
    pred_emb = torch.tensor(model.encode(predictions, convert_to_numpy=True))
    ref_emb  = torch.tensor(model.encode(references,  convert_to_numpy=True))
    
    pred_emb = F.normalize(pred_emb, dim=-1)
    ref_emb  = F.normalize(ref_emb,  dim=-1)
    
    # Menghasilkan list of float (skor untuk setiap instance)
    scores = (pred_emb * ref_emb).sum(dim=-1).tolist()
    
    return {
        "precision": scores,
        "recall": scores,
        "f1": scores
    }

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    is_lora = os.path.exists(os.path.join(args.model_path, "adapter_config.json"))
    if is_lora and not args.base_model_path:
        raise ValueError("Terdeteksi adapter LoRA, silakan masukkan --base_model_path")

    load_path = args.base_model_path if is_lora else args.model_path

    if "qwen3" in load_path:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/qwen3-1.7B", trust_remote_code=True, use_fast=False)
    elif "qwen_5" in load_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True, use_fast=False)
        
    print(f"Loading config and tokenizer...")
    config = AutoConfig.from_pretrained(load_path, trust_remote_code=True)
    
    if config.is_encoder_decoder:
        print(f"Architecture: Encoder-Decoder ({config.model_type})")
        model_class = AutoModelForSeq2SeqLM
        tokenizer.padding_side = "right"
    else:
        print(f"Architecture: Decoder-only ({config.model_type})")
        model_class = AutoModelForCausalLM
        tokenizer.padding_side = "left"

    model = model_class.from_pretrained(
        load_path, 
        torch_dtype=torch.float16, 
        device_map="auto", 
        trust_remote_code=True
    )

    if is_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("Library 'peft' tidak ditemukan. Jalankan 'pip install peft'")
        print(f"Loading LoRA adapters from {args.model_path}...")
        model = PeftModel.from_pretrained(model, args.model_path)
        if args.merge_lora:
            print("Merging LoRA weights...")
            model = model.merge_and_unload()
    
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, dict):
            for key in data:
                if isinstance(data[key], list):
                    data = data[key]
                    break

    rouge_metric = load("rouge")
    bertscore_metric = load("bertscore")
    
    all_predictions = []
    all_references = []
    save_data = []

    prompt_path = pathlib.Path(__file__).parent.parent.parent / f'generate_dataset/{args.prompt_path}'
    if model.config.is_encoder_decoder:
        instruct = ""
    else:  
        try:
            with open(prompt_path, "r") as f:
                data_prompt = json.loads(f.read())
                instruct = data_prompt.get("prompt_positive", "Summarize the following dialogue:\n")
        except Exception as e:
            print(f"Warning: Failed to load prompt.json, menggunakan default instruksi. {e}")
            instruct = "Summarize the following dialogue:\n"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left" 

    for i in tqdm(range(0, len(data), args.batch_size), desc="Evaluating"):
        batch = data[i : i + args.batch_size]
        
        if not config.is_encoder_decoder:
            prompts = [f"{instruct}\n{item[args.input_col]}\nSummary: " for item in batch]
        else:
            prompts = [f"{instruct} {item[args.input_col]}" for item in batch]
            
        references = [item[args.target_col] for item in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        input_length = inputs.input_ids.shape[1]
        batch_size, seq_len = inputs["input_ids"].shape
        
        if seq_len == 0:
            raise ValueError("Input teks kosong setelah ditokenisasi.")
            
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        if not config.is_encoder_decoder:
            generated_tokens = outputs[:, input_length:]
        else:
            generated_tokens = outputs

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_preds = [p.strip() for p in decoded_preds]
        
        all_predictions.extend(decoded_preds)
        all_references.extend(references)

        for prompt, pred, ref in zip(prompts, decoded_preds, references):
            save_data.append({"prompt": prompt, "prediction": pred, "reference": ref})
            
        del inputs, outputs
        torch.cuda.empty_cache()

    print("\nCalculating metrics...")
    
    # 1. Hitung ROUGE per instance (matikan agregator)
    rouge_results_raw = rouge_metric.compute(predictions=all_predictions, references=all_references, use_aggregator=False)
    
    # 2. Hitung BERTScore per instance
    try:
        bertscore_results_raw = bertscore_metric.compute(
            predictions=all_predictions, 
            references=all_references, 
            lang=args.lang,
            device=device,
            use_fast_tokenizer=False
        )
    except:
        print("Calculation bert-score manual")
        bertscore_results_raw = compute_bertscore_manual(predictions=all_predictions, references=all_references, device=device)
    
    # 3. Masukkan skor masing-masing ke dalam save_data
    for idx in range(len(save_data)):
        save_data[idx].update({
            "rouge1": rouge_results_raw["rouge1"][idx],
            "rouge2": rouge_results_raw["rouge2"][idx],
            "rougeL": rouge_results_raw["rougeL"][idx],
            "rougeLsum": rouge_results_raw["rougeLsum"][idx],
            "bertscore_precision": bertscore_results_raw["precision"][idx],
            "bertscore_recall": bertscore_results_raw["recall"][idx],
            "bertscore_f1": bertscore_results_raw["f1"][idx]
        })

    # 4. Hitung nilai rata-rata keseluruhan untuk dicetak dan disimpan di final_metrics
    final_metrics = {
        "rouge1": float(np.mean(rouge_results_raw["rouge1"])),
        "rouge2": float(np.mean(rouge_results_raw["rouge2"])),
        "rougeL": float(np.mean(rouge_results_raw["rougeL"])),
        "rougeLsum": float(np.mean(rouge_results_raw["rougeLsum"])),
        "bertscore_precision": float(np.mean(bertscore_results_raw["precision"])),
        "bertscore_recall": float(np.mean(bertscore_results_raw["recall"])),
        "bertscore_f1": float(np.mean(bertscore_results_raw["f1"]))
    }

    print("\n--- Evaluation Results ---")
    for k, v in final_metrics.items():
        print(f"{k}: {v:.4f}")
    
    with open(args.metric_output, 'w') as f:
        json.dump(final_metrics, f, indent=4)
        
    with open(args.prediction_output, 'w') as f:
        json.dump(save_data, f, indent=4)
        
    print(f"\nEvaluasi selesai.")
    print(f"Skor disimpan di: {args.metric_output}")
    print(f"Hasil teks (beserta skor per baris) disimpan di: {args.prediction_output}")

if __name__ == "__main__":
    main()