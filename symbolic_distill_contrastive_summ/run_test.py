import subprocess
import collections.abc
from pathlib import Path
import yaml

def load_yaml(path: str | Path) -> dict:
    """Memuat file YAML dan mengembalikan dictionary kosong jika file tidak valid."""
    filepath = Path(path)
    if not filepath.is_file():
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

    return output_dir
    
    if not output_dir.is_dir():
        print(f"Direktori tidak ditemukan: {output_dir}")
        return ""

    list_checkpoint = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]

    if not list_checkpoint:
        print(f"Tidak ada checkpoint di dalam {output_dir}")
        return ""

    list_checkpoint.sort(key=lambda x: int(x.name.split("-")[1]))
    latest_checkpoint = list_checkpoint[-1]
    
    print(f"Checkpoint terakhir yang ditemukan: {latest_checkpoint.name}")
    return str(latest_checkpoint)

def main():
    method_num = "method6"

    # 1. Memuat konfigurasi utama
    method_cfg = load_yaml(f"configs/methods/{method_num}.yaml")
    dataset_path = method_cfg.get('datasets')
    dataset_cfg = load_yaml(f"configs/datasets/{dataset_path}")
    
    # Memuat command list
    commands = load_yaml(f"configs/test/test1.yaml")
    if not commands:
        commands = {"basic_eval": {}, "unieval": {}} # Fallback jika YAML kosong

    experiments = method_cfg.get('experiments', [])
    if not experiments:
        print("Tidak ada eksperimen yang ditemukan di dalam method_cfg.")
        return

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
            # 3. Memuat dan menggabungkan semua konfigurasi
            exp_cfg = load_yaml(f"configs/experiments/{exp_code}")
            base_cfg = load_yaml(f"configs/technique/{exp_cfg['technique']}")

            model_val = exp_cfg.get('model', {})
            if isinstance(model_val, str):
                model_cfg = load_yaml(f"configs/models/{model_val}")
            else:
                model_cfg = {"model": model_val} if model_val else {}

            cfg = {}
            for c in [base_cfg, model_cfg, dataset_cfg, exp_cfg]:
                cfg = deep_update(cfg, c)

            # 4. Menyiapkan direktori output dasar
            output_name = cfg.get('output_name', f"output_{exp_code.split('.')[0]}")
            output_base_dir = Path(f"./{method_num}_outputs") / output_name
            output_base_dir.mkdir(parents=True, exist_ok=True)

            # 5. Validasi model checkpoint
            cfg["model"]["name_or_path"] = load_checkpoint(method_num, cfg["output_name"])

            # 6. Iterasi untuk setiap versi command di dalam test1.yaml
            for cmd_name, cmd_info in commands.items():
                print(f"\n--- Menyiapkan Command: {cmd_name} ---")
                
                # Mendefinisikan nama file output spesifik untuk versi ini agar tidak tertimpa
                rouge_file = output_base_dir / f"rouge.json"
                generation_file = output_base_dir / f"generated.json"
                
                # Mendefinisikan folder unieval (dibuat jika belum ada)
                unieval_file = output_base_dir / f"unieval.json"

                align_score_file = output_base_dir / "alignscore.json"

                new_score_file = output_base_dir / "saliencescore.json"

                new_cam_file = output_base_dir / "cam.json"

                new_cfm_file = output_base_dir / "cfm.json"

                module_name = cmd_info.get("command", cfg.get("command"))
                cmd = ["python", "-m", str(module_name)]

                # Susun argumen utama berdasarkan jenis evaluasi
                if cmd_name == "unieval":
                    cmd.extend([
                        "--data_file", str(cfg["dataset"]["validation_file"]),
                        "--output_file", str(unieval_file),
                        "--preds_file", str(generation_file),
                        "--dlg_column", str(cfg["columns"]["text_column"]),
                        "--reference_column", str(cfg["columns"]["summary_column"]),
                        "--dataset_name", "multiwoz",
                    ])
                elif cmd_name == "align_score":
                    cmd.extend([
                        "--data_file", str(cfg["dataset"]["validation_file"]),
                        "--output_file", str(align_score_file),
                        "--preds_file", str(generation_file),
                        "--dlg_column", str(cfg["columns"]["text_column"]),
                        "--dataset_name", "multiwoz",
                    ])
                elif cmd_name == "csr":
                    print(generation_file)
                    cmd.extend([
                        "--gen", str(generation_file),
                        "--matched", "../generate_dataset/matched.json",
                        "--out", str(new_score_file)
                    ])

                elif cmd_name == "cam":
                    cmd.extend([
                        "--gen", str(generation_file),
                        "--out", str(new_cam_file)
                    ])

                elif cmd_name == "cfm":
                    cmd.extend([
                        "--gen", str(generation_file),
                        "--out", str(new_cfm_file)
                    ])
                    
                else:  # Berlaku untuk "basic_eval" atau default lainnya
                    cmd.extend([
                        "--model_path", str(cfg["model"]["name_or_path"]),
                        "--input_file", str(cfg["dataset"]["validation_file"]),
                        "--metric_output", str(rouge_file),
                        "--prediction_output", str(generation_file),
                        "--input_col", str(cfg["columns"]["text_column"]),
                        "--target_col", str(cfg["columns"]["summary_column"]),
                        "--prompt_path", str(method_cfg["prompt_path"])
                    ])

                # Ambil argumen SPESIFIK tambahan dari test1.yaml jika ada
                specific_args = cmd_info.get("args", {})
                
                if "--max_new_token" not in specific_args and "generation" in cfg and "max_source_length" in cfg["generation"] and cmd_name not in ["unieval","align_score", "csr", "cam", "cfm"]:
                    cmd.extend(["--max_new_token", str(cfg["generation"]["max_source_length"])])

                for arg_key, arg_val in specific_args.items():
                    if isinstance(arg_val, bool):
                        if arg_val:  
                            cmd.append(str(arg_key))
                    else:
                        cmd.extend([str(arg_key), str(arg_val)])
                try:
                    # 7. Eksekusi Subprocess
                    print(f"Mengeksekusi {exp_code} (Versi: {cmd_name})...")
                    subprocess.run(cmd, check=True)
                    print(f"Versi {cmd_name} selesai.\n")
                except subprocess.CalledProcessError as e:
                    print(f"Eksperimen {exp_code} gagal saat eksekusi model (Exit code: {e.returncode}). Melanjutkan...")
        except FileNotFoundError as e:
            print(f"Eksperimen {exp_code} dilewati karena file tidak ditemukan: {e}")
        except KeyError as e:
            print(f"Eksperimen {exp_code} gagal karena konfigurasi tidak lengkap. Kunci yang hilang: {e}")
        except subprocess.CalledProcessError as e:
            print(f"Eksperimen {exp_code} gagal saat eksekusi model (Exit code: {e.returncode}). Melanjutkan...")
        except Exception as e:
            print(f"Terjadi kesalahan tidak terduga pada eksperimen {exp_code}: {e}")

    print("Seluruh antrean eksperimen telah selesai diproses.")

if __name__ == "__main__":
    main()