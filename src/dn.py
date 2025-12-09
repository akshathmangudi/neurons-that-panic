"""
Delta-norm computation + ranking
"""


import numpy as np
import pandas as pd


def compute(acts_clean, acts_adv, positions, metadata, model):
    """Compute Δactivation and Δnorm for all components."""
    components = []
    eps = 1e-10

    hook_points = [h for h in acts_clean.keys() if h is not None]

    for hook in hook_points:
        if hook not in acts_adv or acts_clean[hook] is None:
            continue

        clean_acts = acts_clean[hook]
        adv_acts = acts_adv[hook]

        # Parse layer and type
        parts = hook.split(".")
        layer = int(parts[1])

        if "mlp" in hook:
            comp_type = "mlp"
            n_components = clean_acts.shape[-1]
        elif "attn" in hook and "hook_z" in hook:
            comp_type = "attn"
            n_components = clean_acts.shape[2]
            clean_acts = clean_acts.mean(dim=-1)
            adv_acts = adv_acts.mean(dim=-1)
        else:
            continue

        # Compute delta at trigger positions
        for comp_idx in range(n_components):
            clean_comp = clean_acts[:, :, comp_idx]
            adv_comp = adv_acts[:, :, comp_idx]

            # Get trigger positions
            trigger_pos_list = []
            for i, (trigger_pos, insertion_positions) in enumerate(
                zip(positions, metadata["insertion_positions"])
            ):
                if trigger_pos > 0:
                    if insertion_positions:
                        trigger_pos_list.append(insertion_positions[0])
                    else:
                        trigger_pos_list.append(trigger_pos - 1)
                else:
                    trigger_pos_list.append(None)

            # Compute mean at trigger positions
            clean_trigger_vals = []
            adv_trigger_vals = []
            clean_all_vals = []

            for i, (clean_seq, adv_seq, trigger_pos) in enumerate(
                zip(clean_comp, adv_comp, trigger_pos_list)
            ):
                if trigger_pos is not None and trigger_pos < clean_seq.shape[0]:
                    clean_trigger_vals.append(clean_seq[trigger_pos].item())
                    adv_trigger_vals.append(adv_seq[trigger_pos].item())
                clean_all_vals.extend(clean_seq.cpu().tolist())

            if not clean_trigger_vals:
                continue

            mean_clean_trigger = np.mean(clean_trigger_vals)
            mean_adv_trigger = np.mean(adv_trigger_vals)
            std_clean_all = np.std(clean_all_vals) if clean_all_vals else eps

            delta = mean_adv_trigger - mean_clean_trigger
            delta_norm = delta / (std_clean_all + eps)

            components.append(
                {
                    "layer": layer,
                    "type": comp_type,
                    "index": comp_idx,
                    "delta": delta,
                    "delta_norm": delta_norm,
                    "position": "trigger",
                }
            )

    df = pd.DataFrame(components)
    return df


def rank(df):
    """Rank components by absolute delta_norm."""
    df = df.copy()
    df["abs_delta_norm"] = df["delta_norm"].abs()
    df = df.sort_values("abs_delta_norm", ascending=False)
    return df
