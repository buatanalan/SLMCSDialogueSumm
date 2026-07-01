"""
Hitung metrik kepatuhan format per-instance dari generation.json.

Metrik per dialog x model:
  - has_user_profile_strict : 1 jika output mengandung header
                              'User Profile:' persis sesuai template prompt
  - has_profile_any         : 1 jika ada 'Profile:' bentuk apapun
                              (varian BART sering hilangkan kata 'User')
  - has_problem             : 1 jika ada header 'Problem:'
  - has_resolution          : 1 jika ada header 'Resolution:'
  - has_all_three           : 1 jika ketiga section lengkap
  - n_bullets               : jumlah bullet point pada output
  - bullets_violation       : 1 jika jumlah bullet > 5 (langgar 'Max 5 bullets')
  - is_truncated            : 1 jika output TIDAK diakhiri tanda baca
                              terminal (proxy untuk output kepotong max_token)
  - has_boilerplate         : 1 jika output diawali frasa pembuka
                              ('Here is...', 'Sure...', dll)
  - n_words, n_chars        : panjang output prediksi

Output:
  - <out>.csv             : satu baris per (dialog, model) + kolom diagnostik
  - <out>.jsonl           : sama, format JSON lines untuk audit
  - <out>_summary.csv     : rata-rata per model

Cara pakai:
    python compute_format_metrics.py --gen generated_bart.json \\
        --model bart --out format_bart.csv
    python compute_format_metrics.py --inspect --gen generated_bart.json
"""

import os
import re
import json
import argparse

import pandas as pd


# ----------------------------------------------------------------------
# Pola untuk deteksi format
# ----------------------------------------------------------------------
USER_PROFILE_STRICT_RE = re.compile(r"^\s*User\s+Profile\s*[:：]",
                                    re.IGNORECASE | re.MULTILINE)
PROFILE_ANY_RE = re.compile(r"^\s*(User\s+)?Profile\s*[:：]",
                            re.IGNORECASE | re.MULTILINE)
PROBLEM_RE = re.compile(r"^\s*Problem\s*[:：]",
                        re.IGNORECASE | re.MULTILINE)
RESOLUTION_RE = re.compile(r"^\s*Resolution\s*[:：]",
                           re.IGNORECASE | re.MULTILINE)
BULLET_RE = re.compile(r"^\s*[-•*]\s+", re.MULTILINE)
BOILERPLATE_RE = re.compile(
    r"^\s*(here\s+is|here\s*'?s|sure|certainly|of\s+course|"
    r"below\s+is|let\s+me|i\s+will|i\s*'?ll|okay|alright)",
    re.IGNORECASE)
TERMINAL_PUNCT = (".", "!", "?", '"', ")", "]", "—", "'")
MAX_BULLETS = 5


# ----------------------------------------------------------------------
# Format compliance — kembalikan flag biner & jumlah untuk per-instance audit
# ----------------------------------------------------------------------
def format_metrics(pred: str):
    pred = (pred or "").strip()
    bullets_found = BULLET_RE.findall(pred)
    n_bullets = len(bullets_found)

    has_up_strict = bool(USER_PROFILE_STRICT_RE.search(pred))
    has_profile_any = bool(PROFILE_ANY_RE.search(pred))
    has_problem = bool(PROBLEM_RE.search(pred))
    has_resolution = bool(RESOLUTION_RE.search(pred))
    has_all_three = has_profile_any and has_problem and has_resolution

    is_truncated = bool(pred) and not pred.rstrip().endswith(TERMINAL_PUNCT)
    has_boilerplate = bool(BOILERPLATE_RE.match(pred[:80]))

    return {
        "has_user_profile_strict": int(has_up_strict),
        "has_profile_any": int(has_profile_any),
        "has_problem": int(has_problem),
        "has_resolution": int(has_resolution),
        "has_all_three": int(has_all_three),
        "n_bullets": n_bullets,
        "bullets_violation": int(n_bullets > MAX_BULLETS),
        "is_truncated": int(is_truncated),
        "has_boilerplate": int(has_boilerplate),
        "n_words": len(pred.split()),
        "n_chars": len(pred),
    }


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
                rows.append({"sample_idx": i, "model": default_model,
                             "prediction": str(item)})
                continue
            sid = item.get("sample_idx", item.get("idx", i))
            did = item.get("dialogue_id") or item.get("room_id")
            pred = (item.get("prediction") or item.get("generated")
                    or item.get("output"))
            if pred is not None:
                rows.append({"sample_idx": sid, "dialogue_id": did,
                             "model": item.get("model", default_model),
                             "prediction": pred})
    elif isinstance(data, dict):
        for model, preds in data.items():
            if isinstance(preds, list):
                for i, p in enumerate(preds):
                    rows.append({"sample_idx": i, "model": model,
                                 "prediction": p})
    return rows


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(gen_path, out_csv, default_model):
    gens = load_generation(gen_path, default_model)
    print(f"[load] {len(gens)} prediksi dari {gen_path}")

    per_row, skipped = [], 0
    for g in gens:
        if not g.get("prediction"):
            skipped += 1
            continue
        fm = format_metrics(g["prediction"])
        per_row.append({
            "model": g.get("model", default_model),
            "sample_idx": g.get("sample_idx"),
            "dialogue_id": g.get("dialogue_id"),
            **fm,
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
    metric_cols = ["has_user_profile_strict", "has_profile_any",
                   "has_problem", "has_resolution", "has_all_three",
                   "n_bullets", "bullets_violation", "is_truncated",
                   "has_boilerplate", "n_words", "n_chars"]
    summary = df.groupby("model")[metric_cols].mean().round(4)
    summary_path = out_csv.rsplit(".", 1)[0] + "_summary.csv"
    summary.to_csv(summary_path)

    print(f"\n[skip] {skipped} prediksi dilewati (kosong)")
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
    ap.add_argument("--out", default="format_metrics.csv")
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()
    default_model = a.model or os.path.splitext(os.path.basename(a.gen))[0]
    if a.inspect:
        inspect(a.gen)
    else:
        main(a.gen, a.out, default_model)
