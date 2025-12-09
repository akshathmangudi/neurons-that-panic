"""
Loading prompts from SST-2 dataset
"""

from typing import List, Tuple

from datasets import load_dataset


def load_prompts(
    n: int = 200, dataset_name: str = "sst2"
) -> Tuple[List[str], List[int]]:
    """
    Load prompts and labels from SST-2 validation set.
    Returns (prompts, labels) where labels are 0 (negative) or 1 (positive).
    """
    dataset = load_dataset("glue", dataset_name, split="validation")

    prompts = [
        f"Review: {dataset[i]['sentence']}\nSentiment:"
        for i in range(min(n, len(dataset)))
    ]

    labels = [dataset[i]["label"] for i in range(len(prompts))]

    return prompts, labels
