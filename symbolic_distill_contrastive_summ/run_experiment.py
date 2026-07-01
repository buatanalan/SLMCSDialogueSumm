import yaml
import subprocess
import collections.abc
from pathlib import Path
import os

def load_yaml(path: str | Path) -> dict:
    """Memuat file YAML dan mengembalikan dictionary kosong jika file tidak valid."""
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def deep_update(d: dict, u: dict) -> dict:
    """Menggabungkan dictionary secara mendalam (rekursif)."""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def load_checkpoint(method_num: str, model_name_or_path: str) -> str:
    """Mengambil path dari checkpoint dengan angka iterasi/step tertinggi."""
    output_dir = Path(f"./{method_num}_outputs") / model_name_or_path / "outputs"
    # 1. Validasi apakah direktori ada
    if not os.path.exists(output_dir):
        print(f"Direktori tidak ditemukan: {output_dir}")
        return ""

    # 2. Ambil hanya item yang diawali dengan "checkpoint-"
    list_dir = os.listdir(output_dir)
    list_checkpoint = [d for d in list_dir if d.startswith("checkpoint-")]

    # 3. Jika kosong, kembalikan string kosong
    if not list_checkpoint:
        print(f"Tidak ada checkpoint di dalam {output_dir}")
        return ""

    # 4. Urutkan berdasarkan angka di belakang tanda "-"
    # Menggunakan split("-")[1] untuk mengambil angka, lalu dikonversi ke int
    list_checkpoint.sort(key=lambda x: int(x.split("-")[1]))

    # 5. Ambil elemen terakhir (angka terbesar)
    latest_checkpoint = list_checkpoint[-1]
    
    # 6. Gabungkan menjadi path utuh
    latest_path = os.path.join(output_dir, latest_checkpoint)
    
    print(f"Checkpoint terakhir yang ditemukan: {latest_checkpoint}")
    return latest_path

# 1. Memuat konfigurasi metode dan dataset utama
method_num = "method6"

method_cfg = load_yaml(f"configs/methods/{method_num}.yaml")
dataset_path = method_cfg.get('datasets')
dataset_cfg = load_yaml(f"configs/datasets/{dataset_path}")

experiments = method_cfg.get('experiments', [])

if not experiments:
    print("Tidak ada eksperimen yang ditemukan di dalam method_cfg.")

# 2. Iterasi untuk setiap eksperimen
for idx, exp in enumerate(experiments):
    exp_code = exp.get('code')
    if not exp_code:
        print(f"Melewati eksperimen indeks {idx}: Kunci 'code' tidak ditemukan.")
        continue
        
    print(f"\n{'='*50}")
    print(f"Memulai Eksperimen [{idx + 1}/{len(experiments)}]: {exp_code}")
    print(f"{'='*50}")

    try:
        # 3. Memuat konfigurasi spesifik untuk eksperimen ini
        exp_cfg = load_yaml(f"configs/experiments/{exp_code}")
        base_cfg = load_yaml(f"configs/technique/{exp_cfg['technique']}")

        model_val = exp_cfg.get('model', {})
        if isinstance(model_val, str):
            model_cfg = load_yaml(f"configs/models/{model_val}")
        else:
            model_cfg = {"model": model_val} if model_val else {}

        # 4. Menggabungkan semua konfigurasi
        cfg = {}
        for c in [base_cfg, model_cfg, dataset_cfg, exp_cfg]:
            cfg = deep_update(cfg, c)

        # 5. Menyiapkan direktori output
        output_name = cfg.get('output_name', f"output_{exp_code.split('.')[0]}")
        output_dir = Path(f"./{method_num}_outputs") / output_name / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        if cfg["model"]["name_or_path"] not in ["facebook/bart-base", "Qwen/qwen2.5-0.5B", "facebook/bart-large"]:
            cfg["model"]["name_or_path"] = load_checkpoint(method_num, cfg["model"]["name_or_path"])
            print(cfg["model"]["name_or_path"])
        
        # 6. Menyusun argumen command line dasar
        cmd = [
            "python",
            "-m", str(cfg["command"]),
            "--model_name_or_path", str(cfg["model"]["name_or_path"]),
            "--output_dir", str(output_dir),
            "--test_file", str(cfg["dataset"]["test_file"]),
            "--validation_file", str(cfg["dataset"]["validation_file"]),
            "--max_source_length", str(cfg["generation"]["max_source_length"]),
            "--max_target_length", str(cfg["generation"]["max_target_length"]),
            "--prompt_path", str(method_cfg["prompt_path"]),
        ]

        training_cfg = cfg.get("training", {})
        is_contrastive = (
            "margin_contrast" in cfg.get("command", "")
            or "pair_contrast" in cfg.get("command", "")
        )

        # 7. Argumen dataset / kolom (berbeda untuk contrastive vs biasa)
        if is_contrastive:
            cmd.extend([
                "--theta", str(cfg["training"]["theta"]),
                "--alpha", str(cfg["training"]["alpha"]),
                "--model_type", str(cfg["model"]["model_type"]),
                "--contrastive_train_file", str(cfg["dataset"]["contrastive_train_file"]),
                "--contrastive_dev_file", str(cfg["dataset"]["contrastive_dev_file"]),
            ])
        else:
            cmd.extend([
                "--train_file", str(cfg["dataset"]["train_file"]),
                "--text_column", str(cfg["columns"]["text_column"]),
                "--summary_column", str(cfg["columns"]["summary_column"]),
            ])

        # 8. Teruskan SEMUA argumen di blok training secara dinamis.
        #    Apa pun yang ditambahkan di YAML (warmup_steps, max_grad_norm,
        #    adam_epsilon, fp16, save_total_limit, eval_strategy, dst.)
        #    otomatis ikut tanpa perlu edit kode lagi.

        # Key training yang BUKAN argumen HfArgumentParser / sudah/akan
        # ditangani manual, jangan diteruskan mentah:
        skip_keys = {
            "epochs",          # dipetakan ke num_train_epochs di bawah
            "theta", "alpha",  # khusus contrastive, sudah ditangani di bagian 7
        }

        # num_train_epochs (kompat dengan penamaan lama "epochs")
        epochs = cfg.get("num_train_epochs", training_cfg.get("epochs"))
        if epochs is not None:
            cmd += ["--num_train_epochs", str(epochs)]

        # Flag boolean yang hanya boleh aktif saat BUKAN contrastive
        # (script contrastive punya jalur sendiri, mempertahankan logika lama)
        contrastive_excluded_bools = {
            # "do_train", "do_eval", "do_predict",
            "predict_with_generate", "overwrite_output_dir",
        }

        for key, value in training_cfg.items():
            print(key, value)
            if key in skip_keys or value is None:
                continue

            flag = f"--{key}"

            if isinstance(value, bool):
                # Pertahankan perilaku lama: sebagian bool dimatikan saat contrastive
                if is_contrastive and key in contrastive_excluded_bools:
                    continue

                # HfArgumentParser: bool yang True dikirim flag-saja;
                # yang False dikirim eksplisit agar argumen ber-default-True
                # (mis. ignore_pad_token_for_loss) tetap bisa dimatikan.
                if value:
                    cmd.append(flag)
                else:
                    cmd += [flag, "False"]
            else:
                cmd += [flag, str(value)]

        # (opsional) verifikasi argumen yang dikirim
        print("CMD:", " ".join(cmd))

        # 9. Eksekusi Subprocess
        print(f"Menjalankan perintah untuk {exp_code}...")
        subprocess.run(cmd, check=True)
        print(f"Eksperimen {exp_code} selesai dengan sukses.\n")

        print(f"Perbaikan tokenizer di chekpoint\n")
        cmds = f"python -m margin_contrast.fix_checkpoint_tokenizer --checkpoint {load_checkpoint(method_num, cfg['model']['name_or_path'])} --base Qwen/Qwen2.5-0.5B"
        subprocess.run(cmds, check=True)

    except FileNotFoundError as e:
        print(f"Eksperimen {exp_code} dilewati karena file tidak ditemukan: {e}")
    except KeyError as e:
        print(f"Eksperimen {exp_code} gagal karena konfigurasi tidak lengkap. Kunci yang hilang: {e}")
    except subprocess.CalledProcessError as e:
        print(f"Eksperimen {exp_code} gagal saat eksekusi model (Exit code: {e.returncode}). Melanjutkan ke eksperimen berikutnya...")
    except Exception as e:
        print(f"Terjadi kesalahan tidak terduga pada eksperimen {exp_code}: {e}")

print("Seluruh antrean eksperimen telah selesai diproses.")