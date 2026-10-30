"""
fix_checkpoint_tokenizer.py
============================
Copy tokenizer dari Qwen base ke checkpoint Anda yang kehilangan tokenizer.

Checkpoint Anda hanya berisi model.safetensors + config saja, TANPA file
tokenizer. Script ini mengambil tokenizer dari Qwen2.5-0.5B base lalu
copy ke folder checkpoint.

Usage:
    python fix_checkpoint_tokenizer.py \
        --checkpoint method6_outputs/qwen_5_warmup/outputs/checkpoint-570 \
        --base Qwen/Qwen2.5-0.5B

    # Atau pakai path lokal kalau base sudah ter-download:
    python fix_checkpoint_tokenizer.py \
        --checkpoint method6_outputs/qwen_5_warmup/outputs/checkpoint-570 \
        --base /path/to/local/Qwen2.5-0.5B
"""

import argparse
import os
import shutil
import sys
from transformers import AutoTokenizer, AutoConfig


def fix_checkpoint(checkpoint_path: str, base_model: str):
    print(f"\n{'='*70}")
    print(f"Memperbaiki tokenizer di: {checkpoint_path}")
    print(f"Sumber tokenizer        : {base_model}")
    print(f"{'='*70}\n")

    # ===== 1. Validasi checkpoint =====
    if not os.path.isdir(checkpoint_path):
        print(f"❌ Folder checkpoint tidak ada: {checkpoint_path}")
        sys.exit(1)

    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(safetensors_path):
        # Mungkin pakai pytorch_model.bin
        bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if not os.path.exists(bin_path):
            print(f"❌ Tidak ditemukan model.safetensors atau pytorch_model.bin")
            print(f"   di {checkpoint_path}")
            sys.exit(1)

    if not os.path.exists(config_path):
        print(f"❌ config.json tidak ditemukan!")
        sys.exit(1)

    # ===== 2. Cek apakah tokenizer SUDAH ada =====
    existing_tok_files = [
        f for f in os.listdir(checkpoint_path)
        if "tokenizer" in f.lower() or f in ("vocab.json", "merges.txt", "special_tokens_map.json")
    ]
    if existing_tok_files:
        print(f"⚠️  Tokenizer file sudah ada di checkpoint:")
        for f in existing_tok_files:
            size = os.path.getsize(os.path.join(checkpoint_path, f))
            print(f"      {f}  ({size:,} bytes)")
        ans = input("\nApakah tetap timpa? [y/N]: ").strip().lower()
        if ans != "y":
            print("Dibatalkan.")
            sys.exit(0)

    # ===== 3. Cek vocab_size cocok antara config checkpoint dan base =====
    print("[1] Cek kompatibilitas vocab_size...")
    ckpt_config = AutoConfig.from_pretrained(checkpoint_path)
    base_config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)

    ckpt_vocab = getattr(ckpt_config, "vocab_size", None)
    base_vocab = getattr(base_config, "vocab_size", None)
    print(f"    checkpoint vocab_size : {ckpt_vocab}")
    print(f"    base       vocab_size : {base_vocab}")

    if ckpt_vocab != base_vocab:
        print(f"\n⚠️  Vocab size BERBEDA! Checkpoint Anda mungkin punya")
        print(f"   embedding layer yang berbeda dari base. Tokenizer base")
        print(f"   masih bisa dipakai tapi token tambahan di checkpoint")
        print(f"   tidak akan ter-decode dengan benar.")
        ans = input("\nLanjutkan? [y/N]: ").strip().lower()
        if ans != "y":
            sys.exit(0)
    else:
        print(f"    ✅ vocab_size SAMA, aman untuk copy tokenizer.\n")

    # ===== 4. Load tokenizer base & test =====
    print("[2] Load tokenizer base...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        use_fast=True,
    )

    # Setup pad token (cara aman: via ID)
    if tokenizer.pad_token_id is None:
        endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if endoftext_id is not None and endoftext_id != tokenizer.unk_token_id:
            tokenizer.pad_token_id = endoftext_id
        else:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"    vocab_size      : {tokenizer.vocab_size}")
    print(f"    len(tokenizer)  : {len(tokenizer)}")
    print(f"    pad_token       : {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
    print(f"    eos_token       : {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")

    # Test encode-decode
    test_text = "Hello, this is a sanity test."
    ids = tokenizer.encode(test_text, add_special_tokens=False)
    decoded = tokenizer.decode(ids)
    print(f"    test encode     : {ids[:5]}... -> {decoded!r}")

    if tokenizer.vocab_size < 1000:
        print(f"\n❌ Tokenizer base SENDIRI corrupt (vocab_size={tokenizer.vocab_size})")
        print(f"   Pakai base lain, misal: --base Qwen/Qwen2.5-0.5B")
        sys.exit(1)

    # ===== 5. Save ke checkpoint =====
    print(f"\n[3] Save tokenizer ke {checkpoint_path}...")
    tokenizer.save_pretrained(checkpoint_path)

    saved_files = sorted([
        f for f in os.listdir(checkpoint_path)
        if "tokenizer" in f.lower() or f in ("vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json")
    ])
    print(f"    File ter-save:")
    for f in saved_files:
        size_kb = os.path.getsize(os.path.join(checkpoint_path, f)) / 1024
        print(f"      {f}  ({size_kb:.1f} KB)")

    # ===== 6. Reload & verifikasi =====
    print(f"\n[4] Verifikasi reload tokenizer dari checkpoint...")
    reloaded = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    reload_vocab = reloaded.vocab_size
    print(f"    vocab_size      : {reload_vocab}")

    if reload_vocab != tokenizer.vocab_size:
        print(f"\n❌ GAGAL: vocab_size berubah setelah save ({tokenizer.vocab_size} -> {reload_vocab})")
        sys.exit(1)

    reload_ids = reloaded.encode(test_text, add_special_tokens=False)
    if reload_ids != ids:
        print(f"\n❌ GAGAL: encoding berbeda setelah save")
        print(f"   asli      : {ids[:10]}")
        print(f"   reloaded  : {reload_ids[:10]}")
        sys.exit(1)

    print(f"    ✅ Encoding round-trip OK")

    print(f"\n{'='*70}")
    print(f"✅ SUKSES! Checkpoint sekarang bisa di-load lengkap.")
    print(f"{'='*70}\n")
    print(f"Test load model + tokenizer:")
    print(f"  from transformers import AutoTokenizer, AutoModelForCausalLM")
    print(f"  tok = AutoTokenizer.from_pretrained('{checkpoint_path}')")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{checkpoint_path}')")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path ke folder checkpoint yang kehilangan tokenizer")
    parser.add_argument("--base", default="Qwen/Qwen2.5-0.5B",
                        help="Path ke base model untuk source tokenizer "
                             "(default: Qwen/Qwen2.5-0.5B)")
    args = parser.parse_args()

    fix_checkpoint(args.checkpoint, args.base)