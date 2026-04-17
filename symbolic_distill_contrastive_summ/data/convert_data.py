import argparse
import json
import pathlib

def convert_jsonl_to_json(input_file, output_file):
    
    base_dir = pathlib.Path(__file__).parent.parent.parent / input_file
    
    dataset = []
    
    with open(base_dir, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue  # Lewati baris kosong
            
            try:
                data = json.loads(line)
                
                # Memastikan 'summaries' adalah LIST (sesuai error sebelumnya)
                if 'summaries' in data and not isinstance(data['summaries'], list):
                    data['summaries'] = [data['summaries']]
                                
                dataset.append(data)
            except json.JSONDecodeError as e:
                print(f"Error di baris {i}: {e}")

    # Simpan sebagai JSON Array standar
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    
    print(f"Selesai! {len(dataset)} baris berhasil dikonversi ke {output_file}")

# Jalankan konversi

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Convert data to required format for training.")
    argparser.add_argument("--input_file", type=str, required=True, help="Path to the processed dialogues file (CSV or JSON).")
    args = argparser.parse_args()

    convert_jsonl_to_json(args.input_file, f"data/{args.input_file.split('/')[-1].split('.')[0]}.json")