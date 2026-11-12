"""
SBFLT-10: Dataset Partitioning Module
Implements IID and Non-IID partitioning strategies
for simulated federated learning browser clients.

Supports Dholuo, Kalenjin, and Kidawida datasets.

Author: Kashaf Kamran
Sprint: 1
"""

import numpy as np
from collections import defaultdict


def partition_iid(dataset, num_clients, seed=42):
    """
    Partition dataset into IID client shards.
    Each client receives equal randomly sampled
    subset with uniform class distribution.

    Args:
        dataset (list): List of (text, label) tuples
        num_clients (int): Number of simulated clients
        seed (int): Random seed for reproducibility

    Returns:
        dict: {client_id: [(text, label), ...]}

    Raises:
        ValueError: If num_clients exceeds dataset size
    """
    if num_clients > len(dataset):
        raise ValueError(
            f"num_clients ({num_clients}) cannot exceed "
            f"dataset size ({len(dataset)})"
        )

    # Set seed for reproducibility
    np.random.seed(seed)

    # Shuffle dataset indices randomly
    indices = np.random.permutation(len(dataset))

    # Split indices equally among clients
    client_indices = np.array_split(indices, num_clients)

    # Build client partitions
    partitions = {}
    for client_id, idx_list in enumerate(client_indices):
        partitions[client_id] = [
            dataset[i] for i in idx_list
        ]
        print(
            f"Client {client_id}: "
            f"{len(partitions[client_id])} samples (IID)"
        )

    return partitions


def partition_non_iid(
    dataset, num_clients, alpha=0.5, seed=42
):
    """
    Partition dataset into Non-IID client shards
    using Dirichlet distribution for label skew.

    Lower alpha = more skewed (extreme non-IID)
    Higher alpha = more uniform (closer to IID)
    Recommended values:
        alpha=0.1 for extreme non-IID
        alpha=0.5 for moderate non-IID
        alpha=1.0 for mild non-IID

    Args:
        dataset (list): List of (text, label) tuples
        num_clients (int): Number of simulated clients
        alpha (float): Dirichlet concentration parameter
        seed (int): Random seed for reproducibility

    Returns:
        dict: {client_id: [(text, label), ...]}
    """
    # Set seed for reproducibility
    np.random.seed(seed)

    # Group sample indices by class label
    label_to_indices = defaultdict(list)
    for idx, (text, label) in enumerate(dataset):
        label_to_indices[label].append(idx)

    # Get unique labels
    labels = list(label_to_indices.keys())

    # Initialise empty client partitions
    client_partitions = defaultdict(list)

    # Apply Dirichlet distribution per class label
    for label in labels:
        indices = label_to_indices[label]
        np.random.shuffle(indices)

        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(
            alpha=np.repeat(alpha, num_clients)
        )

        # Convert proportions to actual sample counts
        counts = (
            proportions * len(indices)
        ).astype(int)

        # Fix rounding to ensure all samples assigned
        counts[-1] = len(indices) - counts[:-1].sum()

        # Assign indices to clients
        start = 0
        for client_id, count in enumerate(counts):
            end = start + count
            assigned = indices[start:end]
            client_partitions[client_id].extend(
                [dataset[i] for i in assigned]
            )
            start = end

    # Convert to regular dict
    partitions = {}
    for client_id in range(num_clients):
        partitions[client_id] = client_partitions[client_id]
        print(
            f"Client {client_id}: "
            f"{len(partitions[client_id])} samples "
            f"(Non-IID alpha={alpha})"
        )

    return partitions


def validate_partitions(partitions, original_dataset):
    """
    Validates partition integrity.
    Ensures all samples assigned exactly once
    and partition sizes sum to original dataset size.

    Args:
        partitions (dict): Client partitions
        original_dataset (list): Original full dataset

    Returns:
        dict: Validation report with pass/fail status
    """
    total_assigned = sum(
        len(p) for p in partitions.values()
    )
    original_size = len(original_dataset)

    # Check for zero overlap between partitions
    all_texts = []
    for partition in partitions.values():
        for text, label in partition:
            all_texts.append(text)

    unique_texts = len(set(all_texts))
    no_duplicates = unique_texts == total_assigned

    report = {
        "total_clients": len(partitions),
        "original_size": original_size,
        "total_assigned": total_assigned,
        "sizes_match": total_assigned == original_size,
        "no_duplicates": no_duplicates,
        "validation_passed": (
            total_assigned == original_size
            and no_duplicates
        ),
        "client_sizes": {
            client_id: len(partition)
            for client_id, partition
            in partitions.items()
        }
    }

    # Print validation summary
    print("\n--- Partition Validation Report ---")
    print(f"Original size:   {original_size}")
    print(f"Total assigned:  {total_assigned}")
    print(f"Sizes match:     {report['sizes_match']}")
    print(f"No duplicates:   {report['no_duplicates']}")
    print(
        f"Validation:      "
        f"{'PASSED' if report['validation_passed'] else 'FAILED'}"
    )

    return report


def get_label_distribution(partitions):
    """
    Returns class label distribution per client.
    Used to verify IID uniformity vs Non-IID skew.

    Args:
        partitions (dict): Client partitions

    Returns:
        dict: {client_id: {label: count}}
    """
    distribution = {}

    for client_id, samples in partitions.items():
        label_counts = defaultdict(int)
        for text, label in samples:
            label_counts[label] += 1
        distribution[client_id] = dict(label_counts)

        print(
            f"Client {client_id} distribution: "
            f"{dict(label_counts)}"
        )

    return distribution


def save_partitions(partitions, output_dir):
    """
    Saves client partitions to text files
    for inspection and debugging.

    Args:
        partitions (dict): Client partitions
        output_dir (str): Directory to save files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for client_id, samples in partitions.items():
        filepath = os.path.join(
            output_dir,
            f"client_{client_id}_partition.txt"
        )
        with open(filepath, "w", encoding="utf-8") as f:
            for text, label in samples:
                f.write(f"{label}\t{text}\n")

    print(
        f"\nPartitions saved to: {output_dir}"
    )