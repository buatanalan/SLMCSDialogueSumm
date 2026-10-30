"""
test_tokenizer_save.py
======================
Script standalone untuk memverifikasi tokenizer TIDAK corrupt setelah save.

Jalankan SEBELUM training penuh untuk pastikan fix bug "tokenizer 1 token" sudah
benar.

Usage:
    python test_tokenizer_save.py Qwen/Qwen2.5-0.5B
    # atau path ke checkpoint Anda
    python test_tokenizer_save.py /path/to/checkpoint
"""

import sys
import tempfile
import os
from transformers import AutoTokenizer


def test_tokenizer(model_path: str):
    print(f"\n{'='*60}")
    print(f"Testing tokenizer dari: {model_path}")
    print(f"{'='*60}")

    # ===== 1. Load tokenizer awal =====
    print("\n[1] Load tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    initial_vocab_size = tokenizer.vocab_size
    initial_len = len(tokenizer)
    print(f"    vocab_size      = {initial_vocab_size}")
    print(f"    len(tokenizer)  = {initial_len}")
    print(f"    pad_token       = {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
    print(f"    eos_token       = {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")

    assert initial_vocab_size > 1000, (
        f"FATAL: vocab_size hanya {initial_vocab_size}, "
        f"tokenizer corrupt sejak awal!"
    )

    # ===== 2. Setup pad_token via ID (cara AMAN) =====
    print("\n[2] Setup pad_token & padding_side (cara aman)...")
    if tokenizer.pad_token_id is None:
        endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if endoftext_id is not None and endoftext_id != tokenizer.unk_token_id:
            tokenizer.pad_token_id = endoftext_id
            print(f"    pad_token_id set ke endoftext_id={endoftext_id}")
        else:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token
            print(f"    pad_token_id fallback ke eos_token_id={tokenizer.eos_token_id}")
    else:
        print(f"    pad_token sudah ada: {tokenizer.pad_token!r}")

    tokenizer.padding_side = "left"

    after_setup_len = len(tokenizer)
    assert after_setup_len == initial_len, (
        f"FATAL: jumlah token berubah dari {initial_len} -> {after_setup_len} "
        "setelah set pad_token! Berarti pad_token DITAMBAHKAN sebagai added_token. "
        "Periksa kode: pastikan pakai pad_token_id (int), bukan pad_token (string)."
    )
    print(f"    OK: jumlah token TIDAK berubah ({after_setup_len})")

    # ===== 3. Encode-decode round trip =====
    print("\n[3] Test encode-decode...")
    test_text = "Hello world, this is a test of the Qwen tokenizer."
    ids = tokenizer.encode(test_text, add_special_tokens=False)
    decoded = tokenizer.decode(ids)
    print(f"    Input  : {test_text!r}")
    print(f"    Ids[:5]: {ids[:5]}...")
    print(f"    Decoded: {decoded!r}")
    assert len(ids) > 3, f"Encoding hanya {len(ids)} token, terlalu sedikit!"

    # ===== 4. Save & reload tokenizer =====
    print("\n[4] Save tokenizer ke temp dir...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer.save_pretrained(tmpdir)

        # List file yang di-save
        files = sorted(os.listdir(tmpdir))
        print(f"    Files saved    : {files}")

        # Cek size tokenizer.json
        tok_json = os.path.join(tmpdir, "tokenizer.json")
        if os.path.exists(tok_json):
            size_kb = os.path.getsize(tok_json) / 1024
            print(f"    tokenizer.json : {size_kb:.1f} KB")
            assert size_kb > 100, (
                f"FATAL: tokenizer.json hanya {size_kb} KB, kemungkinan corrupt!"
            )

        # ===== 5. Reload =====
        print("\n[5] Reload tokenizer dari path yang baru di-save...")
        reloaded = AutoTokenizer.from_pretrained(tmpdir, trust_remote_code=True)
        reload_vocab = reloaded.vocab_size
        reload_len = len(reloaded)
        print(f"    vocab_size      = {reload_vocab}")
        print(f"    len(tokenizer)  = {reload_len}")

        assert reload_vocab == initial_vocab_size, (
            f"FATAL: vocab_size berubah! {initial_vocab_size} -> {reload_vocab}\n"
            "Ini adalah bug 'tokenizer cuma 1 token'!"
        )
        assert reload_len == after_setup_len, (
            f"FATAL: jumlah token berubah! {after_setup_len} -> {reload_len}"
        )

        # Round trip dengan reloaded tokenizer
        reload_ids = reloaded.encode(test_text, add_special_tokens=False)
        assert reload_ids == ids, (
            f"FATAL: encoding berbeda setelah reload!\n"
            f"  Asli      : {ids[:10]}...\n"
            f"  Reloaded  : {reload_ids[:10]}...\n"
        )
        print(f"    Round-trip OK: encoding identik")

    print(f"\n{'='*60}")
    print(f"✅ SEMUA TEST LULUS untuk: {model_path}")
    print(f"{'='*60}\n")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_tokenizer_save.py <model_path>")
        print("Example:")
        print("  python test_tokenizer_save.py Qwen/Qwen2.5-0.5B")
        print("  python test_tokenizer_save.py /path/to/checkpoint")
        sys.exit(1)

    model_path = sys.argv[1]
    try:
        test_tokenizer(model_path)
    except AssertionError as e:
        print(f"\n❌ TEST GAGAL:\n{e}\n")
        sys.exit(1)