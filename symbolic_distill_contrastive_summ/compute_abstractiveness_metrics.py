"""
Hitung metrik abstraksi per-instance dari generation.json.

Metrik per dialog x model:
  - novel_1gram, novel_2gram, novel_3gram, novel_4gram
        : proporsi n-gram unik di prediksi yang TIDAK ada di sumber
          (See et al. 2017, Pointer-Generator Networks)
  - coverage
        : proporsi token prediksi yang berasal dari sumber
          (Grusky et al. 2018, Newsroom)
  - density
        : rata-rata kuadrat panjang fragmen ekstraksi
          (Grusky et al. 2018) - tinggi = banyak salinan utuh dari sumber
  - n_words_pred, n_words_src
        : panjang prediksi & sumber untuk diagnostik

Untuk Qwen-style prompt yang punya wrapper instruksi
('You are an information extraction system... Dialogue: ... Summary:'),
script otomatis mengekstrak bagian antara 'Dialogue:' dan 'Summary:'.
Untuk BART-style prompt yang isinya dialog murni, prompt dipakai apa adanya.

Output:
  - <out>.csv             : satu baris per (dialog, model) + diagnostik
  - <out>.jsonl           : sama, format JSON lines untuk audit
  - <out>_summary.csv     : rata-rata per model

Cara pakai:
    python compute_abstractiveness_metrics.py --gen generated_qwen.json \\
        --model qwen --out abstractiveness_qwen.csv
    python compute_abstractiveness_metrics.py --inspect --gen generated_bart.json
"""

import os
import re
import json
import argparse

import pandas as pd


# ----------------------------------------------------------------------
# Untuk Qwen-style prompt yang punya wrapper instruksi
# ----------------------------------------------------------------------
DIALOGUE_RE = re.compile(r"Dialogue\s*:\s*(.*?)\s*Summary\s*:",
                         re.DOTALL | re.IGNORECASE)


def extract_source_dialogue(prompt: str) -> str:
    """
    Ambil bagian dialog dari prompt.
    - Qwen-style: ada wrapper instruksi; ambil yang di antara
      'Dialogue:' dan 'Summary:'.
    - BART-style: prompt sudah dialog murni; kembalikan apa adanya.
    """
    m = DIALOGUE_RE.search(prompt or "")
    if m:
        return m.group(1).strip()
    return (prompt or "").strip()


# ----------------------------------------------------------------------
# Tokenisasi & n-gram
# ----------------------------------------------------------------------
def tokenize(text: str):
    return re.findall(r"\w+", str(text).lower())


def ngrams(tokens, n: int):
    if len(tokens) < n:
        return []
    return list(zip(*[tokens[i:] for i in range(n)]))


# ----------------------------------------------------------------------
# Novel n-gram ratio (See et al. 2017)
# ----------------------------------------------------------------------
def novel_ngram_ratio(pred_tokens, src_tokens, n: int):
    p_set = set(ngrams(pred_tokens, n))
    if not p_set:
        return None
    s_set = set(ngrams(src_tokens, n))
    return len(p_set - s_set) / len(p_set)


# ----------------------------------------------------------------------
# Extractive fragments (Grusky et al. 2018) — kembalikan list panjang
# fragmen yang berhasil disalin utuh dari sumber, lalu hitung
# coverage & density darinya.
# ----------------------------------------------------------------------
def extractive_fragments(summary_tokens, source_tokens):
    """Greedy longest-match dari kiri ke kanan."""
    F = []
    i = 0
    s_len = len(summary_tokens)
    src_len = len(source_tokens)
    while i < s_len:
        best_len = 0
        token_i = summary_tokens[i]
        for j in range(src_len):
            if source_tokens[j] != token_i:
                continue
            k = 0
            while (i + k < s_len and j + k < src_len and
                   summary_tokens[i + k] == source_tokens[j + k]):
                k += 1
            if k > best_len:
                best_len = k
        if best_len > 0:
            F.append(best_len)
            i += best_len
        else:
            i += 1
    return F


def abstractiveness_metrics(pred: str, source: str):
    pred_tok = tokenize(pred)
    src_tok = tokenize(source)
    out = {f"novel_{n}gram": novel_ngram_ratio(pred_tok, src_tok, n)
           for n in (1, 2, 3, 4)}
    if pred_tok:
        frags = extractive_fragments(pred_tok, src_tok)
        out["coverage"] = sum(frags) / len(pred_tok)
        out["density"] = sum(f * f for f in frags) / len(pred_tok)
        out["n_fragments"] = len(frags)
    else:
        out["coverage"] = None
        out["density"] = None
        out["n_fragments"] = 0
    out["n_words_pred"] = len(pred_tok)
    out["n_words_src"] = len(src_tok)
    return out


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------
def inspect(path):
    data = json.load(open(path, "r", encoding="utf-8"))
    print("root:", type(data).__name__)
    if isinstance(data, list):
        print("len:", len(data), "| contoh[0]:")
        print(json.dumps(data[0], ensure_ascii=False, indent=2)[:1200])
    elif isinstance(data, dict):
        print("keys:", list(data.keys())[:20])


def load_generation(path, default_model):
    data = json.load(open(path, "r", encoding="utf-8"))
    rows = []
    if isinstance(data, list):
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            sid = item.get("sample_idx", item.get("idx", i))
            did = item.get("dialogue_id") or item.get("room_id")
            pred = (item.get("prediction") or item.get("generated")
                    or item.get("output"))
            if pred is not None:
                rows.append({"sample_idx": sid, "dialogue_id": did,
                             "model": item.get("model", default_model),
                             "prediction": pred,
                             "prompt": item.get("prompt", "")})
    return rows


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(gen_path, out_csv, default_model):
    gens = load_generation(gen_path, default_model)
    print(f"[load] {len(gens)} prediksi dari {gen_path}")

    per_row, skipped = [], 0
    for g in gens:
        if not g.get("prediction") or not g.get("prompt"):
            skipped += 1
            continue
        source = extract_source_dialogue(g["prompt"])
        am = abstractiveness_metrics(g["prediction"], source)
        per_row.append({
            "model": g.get("model", default_model),
            "sample_idx": g.get("sample_idx"),
            "dialogue_id": g.get("dialogue_id"),
            **am,
        })

    if not per_row:
        print("[!] Tidak ada baris terhitung. Cek struktur file via --inspect.")
        return

    df = pd.DataFrame(per_row)

    # --- per-instance CSV ---
    df.to_csv(out_csv, index=False)

    # --- per-instance JSONL ---
    jsonl_path = out_csv.rsplit(".", 1)[0] + ".jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in per_row:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- ringkasan per model ---
    metric_cols = ["novel_1gram", "novel_2gram", "novel_3gram", "novel_4gram",
                   "coverage", "density", "n_fragments",
                   "n_words_pred", "n_words_src"]
    summary = df.groupby("model")[metric_cols].mean().round(4)
    summary_path = out_csv.rsplit(".", 1)[0] + "_summary.csv"
    summary.to_csv(summary_path)

    print(f"\n[skip] {skipped} prediksi dilewati (kosong / tanpa prompt)")
    print("\n=== Rata-rata per model ===")
    print(summary.to_string())
    print(f"\n[done] per-instance CSV  -> {out_csv}")
    print(f"[done] per-instance JSONL -> {jsonl_path}")
    print(f"[done] ringkasan          -> {summary_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="generation.json")
    ap.add_argument("--model", default=None,
                    help="Label model default jika tidak ada di file.")
    ap.add_argument("--out", default="abstractiveness_metrics.csv")
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()
    default_model = a.model or os.path.splitext(os.path.basename(a.gen))[0]
    if a.inspect:
        inspect(a.gen)
    else:
        main(a.gen, a.out, default_model)
