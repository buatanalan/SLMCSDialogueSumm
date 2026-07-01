import pandas as pd
import pathlib
import argparse


def main(rang, length, train_test):
    print(f"rang={rang}, length={length}")

    # Path folder tempat script berada
    path = pathlib.Path(__file__).parent

    try:
        df_list = []

        # Contoh:
        # rang=100, length=500
        # -> 100, 200, 300, 400, 500
        for i in range(rang, length + rang, rang):
            file_path = path / f"dialogue_{train_test}_{i}_annotated.jsonl"

            if not file_path.exists():
                print(f"File tidak ditemukan: {file_path}")
                continue

            print(f"Membaca: {file_path}")
            df = pd.read_json(file_path, lines=True)
            df_list.append(df)

        if not df_list:
            print("Tidak ada file yang berhasil dibaca.")
            return

        # Gabungkan semua dataframe
        df_concat = pd.concat(df_list, ignore_index=True)

        df_concat = df_concat.sample(frac=1, random_state=42).reset_index(drop=True)

        # Simpan hasil gabungan
        output_path = path / f"all_dialogue_{train_test}_{length}_annotated.jsonl"
        df_concat.to_json(
            output_path,
            orient="records",
            lines=True,
            force_ascii=False
        )

        print(f"File berhasil disimpan di: {output_path}")

    except Exception as e:
        print(f"Gagal membaca file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merging dataset")

    parser.add_argument(
        "--rang",
        type=int,
        required=True,
        help="Interval file, contoh: 100"
    )

    parser.add_argument(
        "--length",
        type=int,
        required=True,
        help="Batas akhir file, contoh: 500"
    )

    parser.add_argument(
        "--train_test",
        type=str,
        default="train"
    )

    args = parser.parse_args()

    main(args.rang, args.length, args.train_test)