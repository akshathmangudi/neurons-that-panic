"""
Causal patching
"""

import numpy as np
import torch

from src.adv import get_labels
from src.utils import build_position_map


def _build_hooks_dict(components_df):
    """Build hooks dictionary from components dataframe."""
    hooks_dict = {}
    for _, row in components_df.iterrows():
        layer = row["layer"]
        comp_type = row["type"]
        comp_idx = row["index"]

        if comp_type == "mlp":
            hook = f"blocks.{layer}.hook_mlp_out"
        elif comp_type == "attn":
            hook = f"blocks.{layer}.attn.hook_z"
        else:
            continue

        if hook not in hooks_dict:
            hooks_dict[hook] = []
        hooks_dict[hook].append((comp_idx, comp_type))
    return hooks_dict


def _compute_baseline_scores(
    model, clean_prompts, adv_prompts, labels, label_tokens, batch
):
    """Compute baseline clean and adversarial scores."""
    clean_scores = []
    adv_scores = []

    for i in range(0, len(clean_prompts), batch):
        batch_clean = clean_prompts[i : i + batch]
        batch_adv = adv_prompts[i : i + batch]
        batch_labels = labels[i : i + batch]

        clean_tokens = model.to_tokens(batch_clean)
        adv_tokens = model.to_tokens(batch_adv)

        with torch.no_grad():
            clean_logits = model(clean_tokens)
            adv_logits = model(adv_tokens)

            for j, label in enumerate(batch_labels):
                if label in label_tokens:
                    clean_prob = torch.softmax(clean_logits[j, -1, :], dim=-1)[
                        label_tokens[label]
                    ].item()
                    adv_prob = torch.softmax(adv_logits[j, -1, :], dim=-1)[
                        label_tokens[label]
                    ].item()
                    clean_scores.append(clean_prob)
                    adv_scores.append(adv_prob)
                else:
                    clean_scores.append(0.0)
                    adv_scores.append(0.0)

    return clean_scores, adv_scores


def _make_patch_hook(hook_name, clean_acts_single, hooks_dict, position_map):
    """Create a patch hook function for the given hook name."""

    def patch_hook(activations, hook):
        if hook.name != hook_name or hook.name not in hooks_dict:
            return activations

        components_to_patch = hooks_dict[hook.name]
        patched = activations.clone()

        if "mlp" in hook.name:
            for comp_idx, _ in components_to_patch:
                for clean_pos, adv_pos in position_map.items():
                    if (
                        clean_pos < clean_acts_single.shape[1]
                        and adv_pos < activations.shape[1]
                    ):
                        patched[0, adv_pos, comp_idx] = clean_acts_single[
                            0, clean_pos, comp_idx
                        ]
        elif "attn" in hook.name and "hook_z" in hook.name:
            for comp_idx, _ in components_to_patch:
                for clean_pos, adv_pos in position_map.items():
                    if (
                        clean_pos < clean_acts_single.shape[1]
                        and adv_pos < activations.shape[1]
                    ):
                        patched[0, adv_pos, comp_idx, :] = clean_acts_single[
                            0, clean_pos, comp_idx, :
                        ]

        return patched

    return patch_hook


def _compute_recovery(clean_scores, adv_scores, patched_scores, eps=1e-10):
    """Compute recovery metric from scores."""
    recoveries = []
    for clean, adv, patched in zip(clean_scores, adv_scores, patched_scores):
        denominator = clean - adv + eps
        if denominator > eps:
            recovery = (patched - adv) / denominator
        else:
            recovery = 0.0
        recoveries.append(recovery)
    return recoveries


def _run_patching_trial(
    model, clean_prompts, adv_prompts, labels, hooks_dict, metadata, label_tokens
):
    """Run a single patching trial with given hooks dictionary."""
    patched_scores = []

    for i in range(len(adv_prompts)):
        adv_prompt = adv_prompts[i]
        clean_prompt = clean_prompts[i]
        label = labels[i]
        insertion_positions = metadata["insertion_positions"][i]

        adv_tokens = model.to_tokens(adv_prompt)
        clean_tokens = model.to_tokens(clean_prompt)

        with torch.no_grad():
            _, clean_cache = model.run_with_cache(clean_tokens)

            position_map = build_position_map(
                clean_tokens, adv_tokens, insertion_positions
            )

            fwd_hooks = []
            for hook_name in hooks_dict.keys():
                if hook_name in clean_cache:
                    clean_acts_single = clean_cache[hook_name]
                    fwd_hooks.append(
                        (
                            hook_name,
                            _make_patch_hook(
                                hook_name, clean_acts_single, hooks_dict, position_map
                            ),
                        )
                    )

            patched_logits = model.run_with_hooks(adv_tokens, fwd_hooks=fwd_hooks)

            if label in label_tokens:
                patched_prob = torch.softmax(patched_logits[0, -1, :], dim=-1)[
                    label_tokens[label]
                ].item()
                patched_scores.append(patched_prob)
            else:
                patched_scores.append(0.0)

    return patched_scores


def patch(
    model,
    clean_prompts,
    adv_prompts,
    labels,
    df,
    metadata,
    top_k=20,
    n_random=5,
    batch=16,
):
    """Perform causal patching on top-K panic components."""
    label_tokens = get_labels(model)
    eps = 1e-10

    # Get top-K components and build hooks dict
    top_components = df.head(top_k)
    hooks_to_patch = _build_hooks_dict(top_components)

    # Get baseline scores
    clean_scores, adv_scores = _compute_baseline_scores(
        model, clean_prompts, adv_prompts, labels, label_tokens, batch
    )

    # Run patching with top-K components
    patched_scores = _run_patching_trial(
        model,
        clean_prompts,
        adv_prompts,
        labels,
        hooks_to_patch,
        metadata,
        label_tokens,
    )

    # Compute recovery metric
    recoveries = _compute_recovery(clean_scores, adv_scores, patched_scores, eps)
    mean_recovery = np.mean(recoveries)

    # Random baseline
    random_recoveries = []
    for trial in range(n_random):
        random_components = df.sample(n=top_k)
        random_hooks = _build_hooks_dict(random_components)

        random_patched_scores = _run_patching_trial(
            model,
            clean_prompts,
            adv_prompts,
            labels,
            random_hooks,
            metadata,
            label_tokens,
        )

        random_recoveries_trial = _compute_recovery(
            clean_scores, adv_scores, random_patched_scores, eps
        )
        random_recoveries.append(np.mean(random_recoveries_trial))

    mean_random_recovery = np.mean(random_recoveries)
    std_random_recovery = np.std(random_recoveries)

    results = {
        "top_k": top_k,
        "mean_recovery": mean_recovery,
        "mean_random_recovery": mean_random_recovery,
        "std_random_recovery": std_random_recovery,
        "recoveries": recoveries,
        "random_recoveries": random_recoveries,
    }

    return results
