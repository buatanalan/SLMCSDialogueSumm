"""
Sampling 2500 dialog dari MultiWOZ menggunakan:
1. Clustering K-Means (k=100) pada embedding dialog
2. Per cluster, ambil 25 sampel dengan strategi:
   - 1 dialog terdekat ke centroid (representatif tipikal)
   - 24 dialog via k-Center Greedy (farthest points)
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm


# ============================================================
# 1. LOAD & PREPROCESS DATA
# ============================================================

def load_dialogue_test(path: str) -> list[dict]:
    """
    Load dataset dialogue_test_200. Sesuaikan dengan format file Anda.
    Diasumsikan format: [{id: ..., dialogue: ...}, ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f'Loaded {len(data)} dialogues from {path}')
    dialogues = []
    for item in range(len(data)):
        dialogues.append({
            "id": item,
            "dialogue": data[item]["dialogue"]
        })
    return dialogues

def load_multiwoz(path: str) -> list[dict]:
    """
    Load dataset MultiWOZ. Sesuaikan dengan format file Anda.
    Diasumsikan format: {dialogue_id: {log: [...], goal: {...}}}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dialogues = []
    for dial_id, content in enumerate(data):
        # Gabungkan semua turn menjadi satu string untuk embedding
        dialogues.append({
            "id": dial_id,
            "dialogue": content
        })
    return dialogues


# ============================================================
# 2. EMBEDDING
# ============================================================

def encode_dialogues(dialogues: list[dict], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Encode setiap dialog menjadi vector embedding."""
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [d["dialogue"] for d in dialogues]
    print(f"Encoding {len(texts)} dialogues...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # Normalisasi L2 → cosine similarity = dot product
    # Ini penting agar jarak Euclidean ≈ cosine distance
    embeddings = normalize(embeddings, norm="l2")
    return embeddings


# ============================================================
# 3. CLUSTERING
# ============================================================

def cluster_embeddings(embeddings: np.ndarray, k: int = 100, seed: int = 42):
    """K-Means clustering."""
    print(f"Running K-Means with k={k}...")
    km = KMeans(n_clusters=k, random_state=seed, n_init=10, verbose=0)
    labels = km.fit_predict(embeddings)
    return labels, km.cluster_centers_


# ============================================================
# 4. SAMPLING: CENTROID + FARTHEST POINTS
# ============================================================

def select_centroid_and_farthest(
    cluster_embeddings: np.ndarray,
    cluster_indices: np.ndarray,
    centroid: np.ndarray,
    n_samples: int = 25,
    restricted_embeddings: np.ndarray = None,
):
    """
    Pilih n_samples dari satu cluster:
    - 1 titik terdekat ke centroid
    - n_samples-1 titik via k-Center Greedy (farthest-first traversal)

    Args:
        cluster_embeddings: (n_cluster_points, dim) embeddings dalam cluster
        cluster_indices:    (n_cluster_points,) indeks global dialog
        centroid:           (dim,) titik tengah cluster
        n_samples:          jumlah sampel diambil

    Returns:
        list indeks global yang terpilih
    """
    n_points = len(cluster_embeddings)

    # Edge case: cluster lebih kecil dari kuota → ambil semua
    if n_points <= n_samples:
        new_cluster_indices = []
        for i in range(n_points):
            if restricted_embeddings is not None:
                # Cek apakah titik ini terlalu dekat dengan restricted set
                dist_to_restricted = np.linalg.norm(
                    cluster_embeddings[i] - restricted_embeddings, axis=1
                )
                if (dist_to_restricted < 0.1).any():  # threshold jarak (cosine distance)
                    continue  # skip titik ini karena terlalu mirip dengan test set
            new_cluster_indices.append(cluster_indices[i])
        leng = len(new_cluster_indices)
        train_idx = [new_cluster_indices[i] for i in range(int(leng*0.8))]
        test_idx = [new_cluster_indices[i] for i in range(int(leng*0.8), leng)]
        return train_idx, test_idx

    selected_local = []  # indeks lokal dalam cluster
    # set new embedding that is not too close to restricted set
    if restricted_embeddings is not None:
        new_cluster_embeddings = []
        new_cluster_indices = []
        for i in range(n_points):
            dist_to_restricted = np.linalg.norm(
                cluster_embeddings[i] - restricted_embeddings, axis=1
            )
            if (dist_to_restricted < 0.1).any():  # threshold jarak (cosine distance)
                continue  # skip titik ini karena terlalu mirip dengan test set
            new_cluster_embeddings.append(cluster_embeddings[i])
            new_cluster_indices.append(cluster_indices[i])
        cluster_embeddings = np.array(new_cluster_embeddings)
        cluster_indices = np.array(new_cluster_indices)
        n_points = len(cluster_embeddings)

        if n_points == 0:
            return []  # semua titik di cluster ini terlalu mirip dengan test set

    # --- Langkah 1: titik terdekat ke centroid ---
    dist_to_centroid = np.linalg.norm(new_cluster_embeddings - centroid, axis=1)
    first_idx = int(np.argmin(dist_to_centroid))
    selected_local.append(first_idx)

    # --- Langkah 2: farthest-first traversal untuk sisanya ---
    # min_dist[i] = jarak titik i ke titik terdekat di antara yang sudah dipilih
    min_dist = np.linalg.norm(
        new_cluster_embeddings - new_cluster_embeddings[first_idx], axis=1
    )

    for _ in range(n_samples - 1):
        # Pilih titik dengan min_dist terbesar (paling jauh dari semua yg terpilih)
        next_idx = int(np.argmax(min_dist))
        selected_local.append(next_idx)

        # Update min_dist: untuk setiap titik, cek apakah titik baru ini
        # lebih dekat daripada titik terpilih sebelumnya
        new_dist = np.linalg.norm(
            new_cluster_embeddings - new_cluster_embeddings[next_idx], axis=1
        )
        min_dist = np.minimum(min_dist, new_dist)

        # Set jarak titik yang sudah terpilih ke 0 agar tidak terpilih lagi
        min_dist[next_idx] = 0.0

    # Konversi indeks lokal → indeks global
    leng = len(selected_local)

    train_idx = [new_cluster_indices[i] for i in selected_local[:int(leng*0.8)]]
    test_idx = [new_cluster_indices[i] for i in selected_local[int(leng*0.8):]]
    return train_idx, test_idx

def get_additional_samples(
    n_additional: int,
    embeddings: np.ndarray,
    labels: np.ndarray,
    restricted_embeddings: np.ndarray,
    train_idx, test_idx
):
    """
    Ambil sejumlah n sampel tambahan untuk menggenapkan jumlah sampel keseluruhan menjadi 2500, secara random diambil satu persatu per cluster.
    """
    n_clusters = len(np.unique(labels))
    additional_idx = []
    print(f"Sampling {n_additional} additional points...")
    for _ in tqdm(range(n_additional), desc="Sampling additional points"):
        # Pilih cluster secara acak
        cluster_id = np.random.choice(n_clusters)
        mask = labels == cluster_id
        cluster_idx = np.where(mask)[0]
        cluster_emb = embeddings[mask]

        # Cek apakah ada titik yang belum terpilih di cluster ini
        available_idx = [idx for idx in cluster_idx if idx not in train_idx and idx not in test_idx]
        if not available_idx:
            continue  # skip jika semua titik di cluster ini sudah terpilih

        # Pilih satu titik secara acak dari yang tersedia
        selected_idx = np.random.choice(available_idx)

        # Cek apakah titik ini terlalu dekat dengan restricted set
        dist_to_restricted = np.linalg.norm(
            embeddings[selected_idx] - restricted_embeddings, axis=1
        )
        if (dist_to_restricted < 0.1).any():  # threshold jarak (cosine distance)
            continue  # skip titik ini karena terlalu mirip dengan test set

        # Tambahkan ke train atau test set secara acak
        additional_idx.append(selected_idx)
    leng = 2000 - len(train_idx)
    print(f"Selected {len(additional_idx)} additional points, splitting into Train/Test...")
    additional_train_idx = additional_idx[:leng]
    additional_test_idx = additional_idx[leng:]
    return additional_train_idx, additional_test_idx




def sample_from_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    n_per_cluster: int = 25,
    restricted_embeddings: np.ndarray = None,
) -> list[int]:
    """Jalankan sampling untuk semua cluster."""
    selected_train = []
    selected_test = []
    n_clusters = len(centroids)

    for cluster_id in tqdm(range(n_clusters), desc="Sampling per cluster"):
        mask = labels == cluster_id
        cluster_idx = np.where(mask)[0]
        cluster_emb = embeddings[mask]

        picked_train, picked_test = select_centroid_and_farthest(
            cluster_embeddings=cluster_emb,
            cluster_indices=cluster_idx,
            centroid=centroids[cluster_id],
            n_samples=n_per_cluster,
            restricted_embeddings=restricted_embeddings,
        )
        selected_train.extend(picked_train)
        selected_test.extend(picked_test)

    return selected_train, selected_test


# ============================================================
# 5. DIAGNOSTIK (OPSIONAL TAPI BERGUNA)
# ============================================================

def diagnose_clusters(labels: np.ndarray, n_per_cluster: int = 25):
    """Cek apakah ada cluster yang terlalu kecil."""
    unique, counts = np.unique(labels, return_counts=True)
    print("\n=== Cluster Size Diagnostics ===")
    print(f"Min cluster size: {counts.min()}")
    print(f"Max cluster size: {counts.max()}")
    print(f"Mean cluster size: {counts.mean():.1f}")
    print(f"Median cluster size: {np.median(counts):.1f}")

    small = (counts < n_per_cluster).sum()
    if small > 0:
        print(f"⚠️  {small} cluster(s) < {n_per_cluster} sampel → akan diambil semuanya")
    print()


# ============================================================
# 6. MAIN PIPELINE
# ============================================================

def main(
    input_path: str = "data/multiwoz.json",
    output_path: str = "data/multiwoz_sampled_2500.json",
    n_clusters: int = 100,
    n_per_cluster: int = 25,
    seed: int = 42,
):
    # Load
    test_dialogues = load_dialogue_test("./dialogue_test.json")
    print(test_dialogues[0])
    dialogues = load_multiwoz(input_path)
    print(dialogues[0])
    print(f"Loaded {len(dialogues)} dialogues")

    # Embed
    embeddings = encode_dialogues(dialogues)
    restricted_embeddings = encode_dialogues(test_dialogues)

    # Cluster
    labels, centroids = cluster_embeddings(embeddings, k=n_clusters, seed=seed)
    diagnose_clusters(labels, n_per_cluster)

    # Sample
    selected_train_idx, selected_test_idx = sample_from_clusters(
        embeddings, labels, centroids, n_per_cluster=n_per_cluster, restricted_embeddings=restricted_embeddings
    )
    print(f"\nTotal selected: {len(selected_train_idx)} Train & {len(selected_test_idx)} Test")
    additional_train_idx, additional_test_idx = get_additional_samples(
        n_additional=2500 - (len(selected_train_idx) + len(selected_test_idx)),
        embeddings=embeddings,
        labels=labels,
        restricted_embeddings=restricted_embeddings,
        train_idx=selected_train_idx,
        test_idx=selected_test_idx
    )
    print(f"Additional sampled: {len(additional_train_idx)} Train & {len(additional_test_idx)} Test")
    selected_train_idx.extend(additional_train_idx)
    selected_test_idx.extend(additional_test_idx)
    print(f"Final total: {len(selected_train_idx)} Train & {len(selected_test_idx)} Test")

    # Simpan hasil
    sampled_train = [dialogues[i]["dialogue"] for i in selected_train_idx]
    sampled_test = [dialogues[i]["dialogue"] for i in selected_test_idx]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    train_output_path = output_path.replace(".json", "_train.json")
    test_output_path = output_path.replace(".json", "_test.json")
    
    with open(train_output_path, "w", encoding="utf-8") as f:
        json.dump(sampled_train, f, indent=2, ensure_ascii=False)
    
    with open(test_output_path, "w", encoding="utf-8") as f:
        json.dump(sampled_test, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_path}")

    # Simpan juga indeks terpilih + label cluster untuk audit
    audit = {
        "selected_ids": [dialogues[i]["id"] for i in selected_train_idx],
        "cluster_assignments": {
            dialogues[i]["id"]: int(labels[i]) for i in selected_train_idx
        },
    }
    audit_path = output_path.replace(".json", "_audit.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print(f"Audit log saved to {audit_path}")


if __name__ == "__main__":
    main(
        input_path="multiwoz/dialogue_train.json",        # ← sesuaikan path
        output_path="multiwoz/multiwoz_sampled_2500.json",
        n_clusters=100,
        n_per_cluster=25,
        seed=42,
    )