"""
Hitung metrik saliency per-instance dari generation.json + matched.json.

Metrik per dialog x model:
  - slot_recall    : proporsi slot-value ground-truth yang muncul di summary
  - slot_precision : proxy berbasis token numerik (kontrol over-claim)
  - slot_f1
  - scr            : Salient Content Ratio (kalimat salient / total kalimat)

Output:
  - <out>.csv            : satu baris per (dialog, model) + kolom diagnostik
  - <out>.jsonl          : sama, format JSON lines untuk audit (slot hit/miss)
  - <out>_summary.csv    : rata-rata per model

Cara pakai:
    python compute_saliency_metrics.py --gen generation.json \
        --matched matched.json --out saliency_metrics.csv
    python compute_saliency_metrics.py --inspect --gen generation.json
"""

import re
import json
import argparse

import pandas as pd


# ----------------------------------------------------------------------
# Normalisasi & alias value
# ----------------------------------------------------------------------
VALUE_ALIASES = {
    "centre": ["centre", "center", "central", "city centre", "town centre"],
    "cheap": ["cheap", "inexpensive", "budget", "low priced", "affordable"],
    "moderate": ["moderate", "moderately priced", "mid range", "medium price"],
    "expensive": ["expensive", "high end", "pricey", "upscale"],
    "north": ["north", "northern"], "south": ["south", "southern"],
    "east": ["east", "eastern"], "west": ["west", "western"],
}
SKIP_VALUES = {"", "yes", "no", "none", "dontcare", "do n't care",
               "don't care", "not mentioned"}


def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\s:]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def value_variants(value: str):
    v = value.strip().lower()
    variants = {v}
    for canon, alts in VALUE_ALIASES.items():
        if v == canon or v in alts:
            variants.update(alts + [canon])
    return {x for x in variants if x and x not in SKIP_VALUES}


def value_in_text(value: str, norm_summary: str) -> bool:
    for var in value_variants(value):
        nv = normalize(var)
        if nv and re.search(rf"(?<!\w){re.escape(nv)}(?!\w)", norm_summary):
            return True
    return False


def gold_values(slots: dict):
    vals = set()
    for _slot, values in (slots or {}).items():
        for v in values:
            v = str(v).strip()
            if v.lower() not in SKIP_VALUES:
                vals.add(v)
    return vals


# ----------------------------------------------------------------------
# Slot recall / precision / f1 — kembalikan juga hit & miss per-instance
# ----------------------------------------------------------------------
def slot_metrics(summary: str, gold: set):
    norm = normalize(summary)
    hits = [v for v in gold if value_in_text(v, norm)]
    misses = [v for v in gold if v not in hits]
    recall = len(hits) / len(gold) if gold else None

    info_tokens = set(re.findall(r"\b\d[\w:]*\b", norm))
    if info_tokens:
        gold_blob = " ".join(normalize(v) for v in gold)
        matched = sum(1 for tk in info_tokens if tk in gold_blob)
        precision = matched / len(info_tokens)
    else:
        precision = 1.0

    if recall is None:
        f1 = None
    elif (recall + precision) > 0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = 0.0
    return {
        "slot_recall": recall,
        "slot_precision": precision,
        "slot_f1": f1,
        "slot_hits": sorted(hits),
        "slot_misses": sorted(misses),
        "n_info_tokens": len(info_tokens),
    }


# ----------------------------------------------------------------------
# SCR — kembalikan juga hitungan kalimat untuk diagnostik
# ----------------------------------------------------------------------
FILLER_PATTERNS = [
    r"\b(hi|hello|hey|good (morning|afternoon|evening))\b",
    r"\b(thank you|thanks|you'?re welcome|no problem)\b",
    r"\b(goodbye|bye|have a (great|nice|good) day|take care)\b",
    r"\b(sure|of course|certainly|absolutely|okay|ok|alright)\b",
    r"\b(is there anything else|anything else|how can i help|how may i help)\b",
    r"\b(please|kindly)\b",
    r"\b(greeting|pleasantr|closing remark)\w*",
]
FILLER_RE = re.compile("|".join(FILLER_PATTERNS), re.IGNORECASE)


def split_sentences(summary: str):
    parts = re.split(r"(?<=[.!?])\s+|\n+", str(summary).strip())
    return [p.strip() for p in parts if p.strip()]


def scr_metrics(summary: str, gold: set):
    sents = split_sentences(summary)
    if not sents:
        return {"scr": None, "n_sentences": 0,
                "n_salient": 0, "n_filler": 0}
    salient = filler = 0
    for s in sents:
        ns = normalize(s)
        has_gold = any(value_in_text(v, ns) for v in gold) if gold else False
        has_info = bool(re.search(r"\b\d", ns))
        is_filler = bool(FILLER_RE.search(s)) and not (has_gold or has_info)
        if is_filler:
            filler += 1
        elif has_gold or has_info:
            salient += 1
    return {
        "scr": salient / len(sents),
        "n_sentences": len(sents),
        "n_salient": salient,
        "n_filler": filler,
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


def load_generation(path):
    data = json.load(open(path, "r", encoding="utf-8"))
    rows = []
    if isinstance(data, list):
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                rows.append({"sample_idx": i, "model": "pred",
                             "summary": str(item)})
                continue
            sid = item.get("sample_idx", item.get("idx", i))
            did = item.get("dialogue_id") or item.get("room_id")
            if any(k in item for k in ("prediction", "generated", "output")):
                rows.append({"sample_idx": sid, "dialogue_id": did,
                             "model": item.get("model", "pred"),
                             "summary": (item.get("prediction")
                                         or item.get("generated")
                                         or item.get("output"))})
            else:
                model_cols = [k for k in item
                              if re.fullmatch(r"m\d+|pred|summary", k)]
                for mc in model_cols:
                    rows.append({"sample_idx": sid, "dialogue_id": did,
                                 "model": mc, "summary": item[mc]})
    elif isinstance(data, dict):
        for model, summaries in data.items():
            if isinstance(summaries, list):
                for i, summ in enumerate(summaries):
                    rows.append({"sample_idx": i, "model": model,
                                 "summary": summ})
    return rows


def load_matched(path):
    data = json.load(open(path, "r", encoding="utf-8"))
    matched = data.get("matched", data) if isinstance(data, dict) else data
    by_idx, by_did = {}, {}
    for m in matched:
        if "sample_idx" in m:
            by_idx[m["sample_idx"]] = m
        if m.get("dialogue_id"):
            by_did[m["dialogue_id"]] = m
    return by_idx, by_did


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(gen_path, matched_path, out_csv):
    gens = load_generation(gen_path)
    by_idx, by_did = load_matched(matched_path)
    print(f"[load] {len(gens)} prediksi, "
          f"{len(by_idx)} matched(idx), {len(by_did)} matched(id)")

    per_row, skipped = [], 0
    for g in gens:
        m = None
        if g.get("dialogue_id") and g["dialogue_id"] in by_did:
            m = by_did[g["dialogue_id"]]
        elif g.get("sample_idx") in by_idx:
            m = by_idx[g["sample_idx"]]
        if m is None or not g.get("summary"):
            skipped += 1
            continue
        gold = gold_values(m.get("slots", {}))
        # if not gold:
        #     skipped += 1
        #     continue

        sm = slot_metrics(g["summary"], gold)
        sc = scr_metrics(g["summary"], gold)
        per_row.append({
            "model": g.get("model", "pred"),
            "sample_idx": g.get("sample_idx"),
            "dialogue_id": g.get("dialogue_id") or m.get("dialogue_id"),
            "n_gold_slots": len(gold),
            **{k: sm[k] for k in ("slot_recall", "slot_precision", "slot_f1")},
            **sc,
            "slot_hits": sm["slot_hits"],
            "slot_misses": sm["slot_misses"],
            "n_info_tokens": sm["n_info_tokens"],
        })

    if not per_row:
        print("[!] Tidak ada baris terhitung. Cek penautan gen<->matched "
              "dan struktur generation.json via --inspect.")
        return

    df = pd.DataFrame(per_row)

    # --- per-instance CSV (list di-serialize agar rapi di sel) ---
    df_csv = df.copy()
    for col in ("slot_hits", "slot_misses"):
        df_csv[col] = df_csv[col].apply(lambda x: " | ".join(map(str, x)))
    df_csv.to_csv(out_csv, index=False)

    # --- per-instance JSONL (audit penuh, list tetap list) ---
    jsonl_path = out_csv.rsplit(".", 1)[0] + ".jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in per_row:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- ringkasan per model ---
    summary = (df.groupby("model")[["slot_recall", "slot_precision",
                                    "slot_f1", "scr"]].mean().round(4))
    summary_path = out_csv.rsplit(".", 1)[0] + "_summary.csv"
    summary.to_csv(summary_path)

    print(f"\n[skip] {skipped} prediksi dilewati (tak tertaut / tanpa gold)")
    print("\n=== Rata-rata per model ===")
    print(summary.to_string())
    print(f"\n[done] per-instance CSV  -> {out_csv}")
    print(f"[done] per-instance JSONL -> {jsonl_path}")
    print(f"[done] ringkasan          -> {summary_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="generation.json")
    ap.add_argument("--matched", default="matched.json")
    ap.add_argument("--out", default="saliency_metrics.csv")
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()
    if a.inspect:
        inspect(a.gen)
    else:
        main(a.gen, a.matched, a.out)