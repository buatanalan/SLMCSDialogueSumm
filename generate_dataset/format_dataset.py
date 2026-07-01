import pandas as pd
import os
import pathlib
import json
import argparse
import pathlib

def main(input_path, output_path):

    base_dir = pathlib.Path(__file__).parent.parent

    fp = os.path.join(base_dir, input_path)
    print(fp)
    # Menyimpan dataframe yang berhasil dibaca ke dalam list
    
    # Mengambil ekstensi file dan mengubah ke huruf kecil (contoh: '.csv' atau '.json')        
    try:
        # Cek ekstensi dan baca dengan fungsi yang sesuai
        if fp.endswith('.csv'):
            df = pd.read_csv(fp)
            dataframes =  df
            
        elif fp.endswith('.json'):
            # Tambahkan orient='records' jika bentuk JSON Anda adalah list of dictionary
            df = pd.read_json(fp) 
            dataframes = df
            print(dataframes.head())
            
        else:
            print(f"Peringatan: File '{fp}' diabaikan karena format '{ekstensi}' tidak didukung.")
            
    except Exception as e:
        print(f"Gagal membaca file '{fp}': {e}")
    
    dataframes['utterance'] = dataframes['role'] + ": " + dataframes['message']
    
    dialogues = dataframes.groupby('room_id')['utterance'].apply(list).reset_index(name='dialogue')
    
    # Save processed data to JSON file
    with open(output_path, 'w') as f:
        json.dump(["\n".join(dialogue) for dialogue in dialogues['dialogue'].tolist()], f)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Generate summary dataset from processed dialogues.")
    argparser.add_argument("--input_file", type=str, required=True, help="Path to the processed dialogues file (CSV or JSON).")
    argparser.add_argument("--output_file", type=str, required=True, help="Path to save the generated summaries (CSV or JSON).")

    args = argparser.parse_args()

    main(args.input_file, args.output_file)
