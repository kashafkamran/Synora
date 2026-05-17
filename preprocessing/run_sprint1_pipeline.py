"""
US-03 Integration Script
Synora — Browser-Based Federated Learning Toolkit

Shows exactly how US-01 → US-02 → US-03 connect.
Run this directly on the server to preprocess all client shards:

    python run_sprint1_pipeline.py

What it does:
  1. Loads the dataset (US-01)
  2. Partitions into client shards (US-02 — simplified version here)
  3. Runs the full US-03 preprocessing pipeline
  4. Prints a summary of everything that was produced
"""

import json
import csv
import os
import random
from pathlib import Path

# ── US-03 pipeline ────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent / "data"))
from preprocessing import (
    run_preprocessing_pipeline,
    OUTPUT_DIR,
    MAX_SEQUENCE_LENGTH,
)


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

NUM_CLIENTS   = 3       # Number of simulated browser clients
RANDOM_SEED   = 42      # For reproducible partitioning
IID_MODE      = True    # True = IID partition, False = non-IID

# Path to your dataset file (US-01 output)
# Expected format: CSV with "text" and "label" columns
# or JSON list of {"text": "...", "label": "..."} objects
DATASET_PATH  = Path(__file__).parent / "data" / "dataset.csv"


# ─────────────────────────────────────────────
#  US-01: LOAD DATASET
# ─────────────────────────────────────────────

def load_dataset(path: Path) -> list[dict]:
    """
    Load the Kenyan language dataset from CSV or JSON.

    Expected CSV columns: text, label
    Expected JSON format: [{"text": "...", "label": "..."}, ...]

    If no real dataset file exists yet, generates mock data for testing.
    """
    if not path.exists():
        print(f"[US-01] Dataset not found at {path}.")
        print("[US-01] Generating mock Swahili dataset for testing...")
        return _generate_mock_dataset()

    suffix = path.suffix.lower()

    if suffix == ".csv":
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "text" in row and "label" in row:
                    samples.append({"text": row["text"], "label": row["label"]})
        print(f"[US-01] Loaded {len(samples)} samples from {path}")
        return samples

    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = [{"text": s["text"], "label": s["label"]} for s in data]
        print(f"[US-01] Loaded {len(samples)} samples from {path}")
        return samples

    else:
        raise ValueError(f"Unsupported dataset format: {suffix}. Use .csv or .json")


def _generate_mock_dataset(n: int = 90) -> list[dict]:
    """
    Generate a mock Swahili/Kenyan language dataset for testing.
    Replace this with your real dataset (Masakhane, AfricaNLP, etc.)
    """
    templates = {
        "greeting": [
            "Habari yako leo asubuhi",
            "Mambo vipi rafiki yangu",
            "Habari za jioni leo hii",
            "Shikamoo mzee wangu mkubwa",
            "Habari za asubuhi rafiki",
        ],
        "positive": [
            "Ninafurahi sana kukuona tena",
            "Asante sana kwa msaada wako",
            "Mambo mazuri sana leo hii",
            "Nzuri sana asante wewe mwenyewe",
            "Nimefurahi sana na habari hii",
        ],
        "complaint": [
            "Ninasikia njaa sana leo",
            "Sijui la kufanya sasa hivi",
            "Kuna tatizo kubwa sana hapa",
            "Sijapata chakula leo kabisa",
            "Mimi nina maumivu makali sana",
        ],
        "request": [
            "Tafadhali nisaidie na hili tatizo",
            "Ninahitaji chakula na maji sasa",
            "Tafadhali niambie ukweli wote",
            "Unaweza kunisaidia tafadhali leo",
            "Nahitaji msaada wako sasa hivi",
        ],
    }

    samples = []
    random.seed(RANDOM_SEED)
    labels = list(templates.keys())

    for _ in range(n):
        label = random.choice(labels)
        text = random.choice(templates[label])
        samples.append({"text": text, "label": label})

    random.shuffle(samples)
    print(f"[US-01] Generated {len(samples)} mock samples with labels: {labels}")
    return samples


# ─────────────────────────────────────────────
#  US-02: PARTITION INTO CLIENT SHARDS
# ─────────────────────────────────────────────

def partition_iid(samples: list[dict], num_clients: int) -> dict[int, list[dict]]:
    """
    IID partitioning: Randomly shuffle and split evenly.
    Each client gets approximately the same class distribution.
    """
    random.seed(RANDOM_SEED)
    shuffled = samples.copy()
    random.shuffle(shuffled)

    shards = {}
    chunk_size = len(shuffled) // num_clients
    for i in range(num_clients):
        start = i * chunk_size
        end   = start + chunk_size if i < num_clients - 1 else len(shuffled)
        shards[i] = shuffled[start:end]
        print(f"[US-02] Client {i}: {len(shards[i])} samples (IID)")

    return shards


def partition_non_iid(samples: list[dict], num_clients: int) -> dict[int, list[dict]]:
    """
    Non-IID partitioning: Each client gets mostly one label class.
    Simulates realistic federated data conditions.
    """
    random.seed(RANDOM_SEED)

    # Group by label
    by_label: dict[str, list[dict]] = {}
    for s in samples:
        by_label.setdefault(s["label"], []).append(s)

    labels = list(by_label.keys())
    shards = {i: [] for i in range(num_clients)}

    # Assign each client a dominant label
    for i in range(num_clients):
        dominant_label = labels[i % len(labels)]
        dominant_pool = by_label[dominant_label]

        # 80% from dominant label, 20% from others
        n_dominant = max(1, int(len(samples) / num_clients * 0.8))
        n_other    = max(1, int(len(samples) / num_clients * 0.2))

        random.shuffle(dominant_pool)
        shards[i].extend(dominant_pool[:n_dominant])

        other_samples = [s for s in samples if s["label"] != dominant_label]
        random.shuffle(other_samples)
        shards[i].extend(other_samples[:n_other])

        random.shuffle(shards[i])
        print(f"[US-02] Client {i}: {len(shards[i])} samples "
              f"(non-IID, dominant={dominant_label})")

    return shards


# ─────────────────────────────────────────────
#  MAIN: Run full Sprint 1 pipeline
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Sprint 1 Pipeline: US-01 → US-02 → US-03")
    print("=" * 60)

    # ── US-01: Load dataset ───────────────────────────────────
    print("\n── US-01: Load Dataset ──")
    samples = load_dataset(DATASET_PATH)
    print(f"  Total samples: {len(samples)}")

    # Show label distribution
    from collections import Counter
    dist = Counter(s["label"] for s in samples)
    print(f"  Label distribution: {dict(dist)}")

    # ── US-02: Partition into client shards ───────────────────
    print(f"\n── US-02: Partition ({('IID' if IID_MODE else 'Non-IID')}) ──")
    if IID_MODE:
        shards = partition_iid(samples, NUM_CLIENTS)
    else:
        shards = partition_non_iid(samples, NUM_CLIENTS)

    # ── US-03: Preprocess all shards ─────────────────────────
    print("\n── US-03: Automated Text Preprocessing Pipeline ──")
    result = run_preprocessing_pipeline(shards)

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ Sprint 1 Pipeline Complete")
    print("=" * 60)
    print(f"  Dataset samples  : {len(samples)}")
    print(f"  Clients          : {NUM_CLIENTS}")
    print(f"  Partition mode   : {'IID' if IID_MODE else 'Non-IID'}")
    print(f"  Vocabulary size  : {result['vocab_size']}")
    print(f"  Sequence length  : {result['sequence_length']}")
    print(f"  Label classes    : {result['label_map']}")
    print(f"\n  Output files in  : {OUTPUT_DIR}/")
    for cid, path in result["output_paths"].items():
        size = Path(path).stat().st_size // 1024
        print(f"    client_{cid}: {path} ({size} KB)")

    print("\n  Next step: Start your Flask server and open the browser client.")
    print("  The browser will fetch each client's shard from:")
    print("    GET /data/preprocessed/client_<id>_preprocessed.json")


if __name__ == "__main__":
    main()
