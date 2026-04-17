# ── Load MultiWOZ ──────────────────────────────────────────
# Ganti path sesuai lokasi file Anda
import os
import json
import pandas as pd
import argparse

def main(input_file, output_file):
    multiwoz_path = f"multiwoz/data/MultiWOZ_2.2/{input_file}"
    list_file = [os.path.join(multiwoz_path, f) for f in os.listdir(multiwoz_path) if f.endswith(".json")]
    mwoz_raws = []
    for i in list_file:
        print(f"✓ Found file: {i}")
        multiwoz_raw_file = json.load(open(i, "r", encoding="utf-8"))
        mwoz_raws.extend(multiwoz_raw_file)
        
    # # Flatten ke DataFrame
    mwoz_rows = []
    for row in mwoz_raws:
        dialog_id = row['dialogue_id']
        for i, turn in enumerate(row['turns']):
            role = "user" if turn['speaker'].lower() == "user" else "agent"
            mwoz_rows.append({
                "room_id": dialog_id,
                "message_id": turn['turn_id'],
                "role": role,
                "message": turn['utterance']
            })
    # print(mwoz_rows[0].keys())
    df_mwoz = pd.DataFrame(mwoz_rows)
    print(f"✓ MultiWOZ: {df_mwoz['message_id'].nunique():,} dialogues, {len(df_mwoz):,} utterances")
    df_mwoz.to_json(f'generate_dataset/multiwoz/{output_file}')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Generate summary dataset from processed dialogues.")
    argparser.add_argument("--input_file", type=str, required=True, help="Path to the processed dialogues file (CSV or JSON).")
    argparser.add_argument("--output_file", type=str, required=True, help="Path to save the generated summaries (CSV or JSON).")
    argparser.add_argument("--n_completions", type=int, default=3, help="Number of summaries to generate per dialogue.")
    argparser.add_argument("--model_name", type=str, default="gpt-4o", help="OpenAI model to use for generation.")
    args = argparser.parse_args()

    main(args.input_file, args.output_file)
