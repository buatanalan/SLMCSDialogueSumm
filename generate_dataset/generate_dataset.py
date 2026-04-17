import os
import pathlib
from openai import OpenAI
import argparse
import json
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()


client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENAI_API_KEY"),
)

prompt_path = pathlib.Path(__file__).parent / 'prompt.json'

with open(prompt_path, "r") as f:
    data = json.loads(f.read())
    prompt_data_pos = data['prompt_positive']
    prompt_data_neg = data['prompt_negative']
    # prompt_data_neg = json.loads(f.read())['prompt_negative']

def generate_summary_data(processed_input, n_completions=3, model_name="gpt-4o", prompt="Buatlah ringkasan percakapan berikut dengan format yang jelas dan informatif:\n\n"):
    """
    Menghasilkan n ringkasan dari satu input berdasarkan instruksi spesifik.
    """
    
    # Menyusun instruksi sesuai format yang Anda minta
    # Menggunakan \n (newline) sebagai representasi dari CHAR(10)
    instruction = (
        prompt +
        processed_input
    )
    
    try:
        responses = []
        for i in range(n_completions):
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "Anda adalah asisten AI ahli peringkasan percakapan dalam Bahasa Indonesia."
                },
                {
                    "role": "user", 
                    "content": instruction
                }
            ],
            temperature=0.7, # Nilai yang seimbang untuk variasi ringkasan
            max_tokens=500
        )
            responses.append(response)
        # Mengambil semua hasil ringkasan
        summaries = [response.choices[0].message.content for response in responses]
        return {"dialogue": processed_input, "summaries": summaries }

    except Exception as e:
        print(f"Error pada API hit: {e}")
        return []

def load_processed_dialogues(file_path):
    """
    Fungsi untuk memuat dialog yang sudah diproses dari file (CSV atau JSON).
    Implementasikan sesuai format file Anda.
    """
    # Contoh implementasi untuk CSV
    import pandas as pd
    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
        return df['processed_dialogue'].tolist()  # Sesuaikan dengan nama kolom Anda
    elif file_path.suffix == '.json' or file_path.suffix == '.jsonl':
        import json
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return [item for item in data]  # Sesuaikan dengan struktur JSON Anda
            else:
                return [data]  # Jika hanya satu item
    else:
        raise ValueError("Unsupported file format. Please use CSV or JSON.")
    
def process_and_save(data, output_path):
    output_file_path = pathlib.Path(__file__).parent / f"dialogue_{output_path}.jsonl"

    results = []
    for dialogue in data:
        for i in range(args.n_completions):
            result_pos = generate_summary_data(
                processed_input=dialogue,
                n_completions=args.n_completions,
                model_name=args.model_name,
                prompt=prompt_data_pos
            )
            result_neg = generate_summary_data(
                processed_input=dialogue,
                n_completions=args.n_completions,
                model_name=args.model_name,
                prompt=prompt_data_neg
            )

        results.append({
            "dialogue": dialogue,
            "summary": result_pos["summaries"][0],
            "summaries": [ {"pos": res_pos, "neg": res_neg} for res_pos, res_neg in zip(result_pos["summaries"], result_neg["summaries"]) ]
        })

    with open(output_file_path, 'w') as f:
        for item in results:
            json.dump(item, f)
            f.write('\n')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Generate summary dataset from processed dialogues.")
    argparser.add_argument("--input_file", type=str, required=True, help="Path to the processed dialogues file (CSV or JSON).")
    argparser.add_argument("--output_file", type=str, required=True, help="Path to save the generated summaries (CSV or JSON).")
    argparser.add_argument("--n_completions", type=int, default=3, help="Number of summaries to generate per dialogue.")
    argparser.add_argument("--model_name", type=str, default="gpt-4o", help="OpenAI model to use for generation.")
    args = argparser.parse_args()

    # Load processed dialogues from the input file (implementasikan sesuai format file Anda)
    prompt_path = pathlib.Path(__file__).parent / 'prompt.json'

    input_file_path = pathlib.Path(__file__).parent / args.input_file
    
    processed_dialogues = load_processed_dialogues(input_file_path)[:10]
    print(f"generate summary for {len(processed_dialogues)} dialogues")
    # Generate summaries untuk setiap dialog

    # Step 1: split data
    temp_data, test_data = train_test_split(
        processed_dialogues,
        test_size=0.2,
        random_state=42
    )

    train_data, val_data = train_test_split(
        temp_data,
        test_size=0.25,
        random_state=42
    )

    process_and_save(train_data, "train")
    process_and_save(val_data, "validation")
    process_and_save(test_data, "test")
    

    
    
