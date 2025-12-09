"""
Creating adversarial triggers
"""

import json
import os

import numpy as np
import torch


def build_candidates(model, n=500):
    """Build a diverse list of candidate tokens that decode to actual words."""
    special_token_ids = {0}  # Token 0 is <|endoftext|>
    if hasattr(model.tokenizer, "all_special_ids"):
        special_token_ids.update(model.tokenizer.all_special_ids)

    ranges_to_check = [
        (1000, 5000, 20),
        (5000, 15000, 10),
        (15000, 30000, 5),
        (30000, 45000, 3),
    ]

    good_tokens = []
    for vocab_start, vocab_end, step in ranges_to_check:
        for token_id in range(vocab_start, min(vocab_end, model.cfg.d_vocab), step):
            if token_id in special_token_ids:
                continue

            try:
                token_str = model.to_string(torch.tensor([token_id]))
                if (
                    token_str
                    and len(token_str.strip()) > 0
                    and not token_str.startswith("<|")
                    and not token_str.startswith("[")
                    and len(token_str) < 15
                    and any(c.isalnum() for c in token_str)
                ):
                    good_tokens.append(token_id)
                    if len(good_tokens) >= n:
                        return good_tokens
            except Exception:
                continue

    return good_tokens[:n]


def get_labels(model):
    """Get token IDs for 'negative' and 'positive' labels."""
    label_tokens = {}
    for label_val, label_name in [(0, "negative"), (1, "positive")]:
        try:
            token_id = model.to_single_token(f" {label_name}")
            label_tokens[label_val] = token_id
        except Exception:
            try:
                token_id = model.to_single_token(label_name)
                label_tokens[label_val] = token_id
            except Exception:
                tokens = model.tokenizer.encode(
                    f" {label_name}", add_special_tokens=False
                )
                if tokens:
                    label_tokens[label_val] = tokens[0]
    return label_tokens


def eval_candidates(model, base_tokens, pos, candidates, label_token, batch=32):
    """Evaluate multiple candidate tokens in a batched forward pass."""
    valid_candidates = [t for t in candidates if t != 0]
    if not valid_candidates:
        return []

    objectives = []
    device = base_tokens.device
    dtype = base_tokens.dtype

    for i in range(0, len(valid_candidates), batch):
        batch_candidates = valid_candidates[i : i + batch]
        batch_sequences = []
        for token_id in batch_candidates:
            modified = torch.cat(
                [
                    base_tokens[:pos],
                    torch.tensor([token_id], device=device, dtype=dtype),
                    base_tokens[pos:],
                ]
            )
            batch_sequences.append(modified)

        batch_tensor = torch.stack(batch_sequences)

        with torch.no_grad():
            logits = model(batch_tensor)
            next_token_logits = logits[:, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            adv_probs = next_token_probs[:, label_token]
            batch_objectives = -torch.log(adv_probs + 1e-10)
            objectives.extend(batch_objectives.cpu().tolist())

    return objectives


def _insert_trigger(prompt, trigger_tokens):
    """Insert trigger tokens after 'Sentiment:' in the prompt."""
    parts = prompt.split()
    try:
        sentiment_idx = next(
            i for i, tok in enumerate(parts) if tok.startswith("Sentiment")
        )
    except StopIteration:
        sentiment_idx = 0
    insert_position = sentiment_idx + 1
    adv_parts = parts[:insert_position] + trigger_tokens + parts[insert_position:]
    return " ".join(adv_parts)


def _insert_token(tokens, token_id, position):
    """Insert a token at the given position in the token sequence."""
    return torch.cat(
        [
            tokens[:position],
            torch.tensor([token_id], device=tokens.device, dtype=tokens.dtype),
            tokens[position:],
        ]
    )


def _evaluate_token_objective(model, tokens, label_token):
    """Evaluate and compute objective for a token sequence."""
    with torch.no_grad():
        logits = model(tokens.unsqueeze(0))
        next_token_logits = logits[0, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        objective = -np.log(next_token_probs[label_token].item() + 1e-10)
    return objective


def _get_candidate_positions(insert_pos, seq_len):
    """Get candidate positions around an inserted token for next insertion."""
    candidate_positions = []
    if insert_pos > 0:
        candidate_positions.append(insert_pos - 1)
    candidate_positions.append(insert_pos)
    if insert_pos < seq_len - 1:
        candidate_positions.append(insert_pos + 1)
    candidate_positions.append(insert_pos + 2)
    return [p for p in candidate_positions if 0 < p < seq_len]


def _find_best_token(
    model,
    base_tokens,
    positions,
    candidates,
    label_token,
    batch,
    threshold=float("-inf"),
):
    """Find the best token and position from candidates."""
    best_token = None
    best_position = None
    best_objective = threshold

    for pos in positions:
        objectives = eval_candidates(
            model, base_tokens, pos, candidates, label_token, batch
        )
        valid_candidates = [t for t in candidates if t != 0]

        for idx, objective in enumerate(objectives):
            if objective > best_objective + 1e-6:
                best_objective = objective
                best_token = valid_candidates[idx]
                best_position = pos

    return best_token, best_position, best_objective


def _create_empty_metadata(i, prompt, label):
    """Create empty metadata entry for prompts that couldn't be modified."""
    return {
        "id": i,
        "clean_prompt": prompt,
        "clean_label": int(label),
        "adv_tokens": [],
        "adv_decoded": "",
        "positions": [],
        "objective_k": {"k1": None, "k2": None, "k3": None},
        "relative_improvements": {"r2": None, "r3": None},
        "ablation_marginals": [],
        "chosen_k": 0,
    }


def gen_adv(model, prompt, label, candidates, positions, label_tokens, batch=32):
    """Generate adversarial prompt by inserting 1-3 tokens using iterative greedy approach."""
    tokens = model.to_tokens(prompt)[0]
    seq_len = tokens.shape[0]

    if label not in label_tokens:
        return (
            prompt,
            0,
            0,
            [],
            [],
            {"k1": None, "k2": None, "k3": None},
            {"r2": None, "r3": None},
            [],
        )

    correct_label_token = label_tokens[label]
    valid_positions = [p for p in positions if 0 < p < seq_len]
    if not valid_positions:
        return (
            prompt,
            0,
            0,
            [],
            [],
            {"k1": None, "k2": None, "k3": None},
            {"r2": None, "r3": None},
            [],
        )

    # Find best 1-token insertion
    best_1_token, best_1_position, best_1_objective = _find_best_token(
        model, tokens, valid_positions, candidates, correct_label_token, batch
    )

    if best_1_token is None or best_1_position is None:
        return (
            prompt,
            0,
            0,
            [],
            [],
            {"k1": None, "k2": None, "k3": None},
            {"r2": None, "r3": None},
            [],
        )

    tokens_1 = _insert_token(tokens, best_1_token, best_1_position)
    objective_1_actual = _evaluate_token_objective(model, tokens_1, correct_label_token)

    result_1 = {
        "tokens": tokens_1,
        "objective": objective_1_actual,
        "num_tokens": 1,
        "inserted_tokens": [best_1_token],
        "positions": [best_1_position],
    }

    best_result = result_1

    # Try adding 2nd token
    candidate_positions_2 = _get_candidate_positions(best_1_position, tokens_1.shape[0])
    best_2_token, best_2_position, best_2_objective = _find_best_token(
        model,
        tokens_1,
        candidate_positions_2,
        candidates,
        correct_label_token,
        batch,
        objective_1_actual,
    )

    if (
        best_2_token is not None
        and best_2_position is not None
        and best_2_objective > objective_1_actual + 1e-6
    ):
        tokens_2 = _insert_token(tokens_1, best_2_token, best_2_position)
        objective_2_actual = _evaluate_token_objective(
            model, tokens_2, correct_label_token
        )

        result_2 = {
            "tokens": tokens_2,
            "objective": objective_2_actual,
            "num_tokens": 2,
            "inserted_tokens": [best_1_token, best_2_token],
            "positions": [best_1_position, best_2_position],
        }
        best_result = result_2

        # Try adding 3rd token
        candidate_positions_3 = _get_candidate_positions(
            best_2_position, tokens_2.shape[0]
        )
        best_3_token, best_3_position, best_3_objective = _find_best_token(
            model,
            tokens_2,
            candidate_positions_3,
            candidates,
            correct_label_token,
            batch,
            objective_2_actual,
        )

        if (
            best_3_token is not None
            and best_3_position is not None
            and best_3_objective > objective_2_actual + 1e-6
        ):
            tokens_3 = _insert_token(tokens_2, best_3_token, best_3_position)
            objective_3_actual = _evaluate_token_objective(
                model, tokens_3, correct_label_token
            )

            result_3 = {
                "tokens": tokens_3,
                "objective": objective_3_actual,
                "num_tokens": 3,
                "inserted_tokens": [best_1_token, best_2_token, best_3_token],
                "positions": [best_1_position, best_2_position, best_3_position],
            }
            best_result = result_3

    inserted_token_strings = [
        model.to_string(torch.tensor([tid])) for tid in best_result["inserted_tokens"]
    ]
    adv_prompt = _insert_trigger(prompt, inserted_token_strings)
    trigger_pos = best_result["positions"][0] + 1 if best_result["positions"] else 0

    objective_k = {
        "k1": objective_1_actual if "objective_1_actual" in locals() else None,
        "k2": objective_2_actual if "objective_2_actual" in locals() else None,
        "k3": objective_3_actual if "objective_3_actual" in locals() else None,
    }

    relative_improvements = {"r2": None, "r3": None}
    if (
        objective_k["k1"] is not None
        and objective_k["k2"] is not None
        and objective_k["k1"] > 0
    ):
        relative_improvements["r2"] = (
            objective_k["k2"] - objective_k["k1"]
        ) / objective_k["k1"]
    if (
        objective_k["k2"] is not None
        and objective_k["k3"] is not None
        and objective_k["k2"] > 0
    ):
        relative_improvements["r3"] = (
            objective_k["k3"] - objective_k["k2"]
        ) / objective_k["k2"]

    ablation_marginals = []
    if best_result["num_tokens"] == 3 and "tokens_3" in locals():
        inserted_tokens = best_result["inserted_tokens"]
        insertion_positions = best_result["positions"]
        baseline_obj = objective_k["k3"]
        sorted_indices = sorted(range(3), key=lambda i: insertion_positions[i])

        for i in range(3):
            tokens_ablated = tokens.clone()
            for idx in sorted_indices:
                if idx != i:
                    tok = inserted_tokens[idx]
                    pos = insertion_positions[idx]
                    offset = sum(
                        1
                        for j in sorted_indices
                        if j < idx and j != i and insertion_positions[j] <= pos
                    )
                    insert_pos = pos + offset
                    tokens_ablated = _insert_token(tokens_ablated, tok, insert_pos)

            objective_ablated = _evaluate_token_objective(
                model, tokens_ablated, correct_label_token
            )

            if baseline_obj is not None and baseline_obj > 1e-10:
                marginal = (objective_ablated - baseline_obj) / baseline_obj
            else:
                marginal = 0.0
            ablation_marginals.append(marginal)
    elif best_result["num_tokens"] == 2:
        if (
            objective_k["k1"] is not None
            and objective_k["k2"] is not None
            and objective_k["k1"] > 1e-10
        ):
            marginal_2 = (objective_k["k2"] - objective_k["k1"]) / objective_k["k1"]
            ablation_marginals = [0.0, marginal_2]
        else:
            ablation_marginals = [0.0, 0.0]
    elif best_result["num_tokens"] == 1:
        ablation_marginals = [0.0]

    return (
        adv_prompt,
        trigger_pos,
        best_result["num_tokens"],
        best_result["inserted_tokens"],
        best_result["positions"],
        objective_k,
        relative_improvements,
        ablation_marginals,
    )


def gen_all(model, prompts, labels, n_cand=500, batch=32, out_dir="artifacts"):
    """Generate adversarial prompts for all clean prompts using iterative greedy 1-3 token insertion."""
    os.makedirs(out_dir, exist_ok=True)

    label_tokens = get_labels(model)
    candidate_tokens = build_candidates(model, n_cand)

    if not candidate_tokens:
        raise ValueError("Failed to build candidate token list")

    adv_prompts = []
    trigger_positions = []
    metadata_list = []

    metadata_file = os.path.join(out_dir, "adv_metadata.jsonl")
    if os.path.exists(metadata_file):
        os.remove(metadata_file)

    for i, (prompt, label) in enumerate(zip(prompts, labels)):
        review_text = prompt.split("\nSentiment:")[0]
        review_tokens = model.to_tokens(review_text)[0]
        max_insertion_pos = review_tokens.shape[0] - 1

        if max_insertion_pos < 2:
            adv_prompts.append(prompt)
            trigger_positions.append(0)
            metadata_list.append(_create_empty_metadata(i, prompt, label))
            continue

        positions = []
        if max_insertion_pos > 1:
            positions.append(1)
        if max_insertion_pos > 3:
            positions.append(min(max_insertion_pos // 2, max_insertion_pos - 1))
        if max_insertion_pos > 2:
            positions.append(max_insertion_pos)
        positions = sorted(set([p for p in positions if 0 < p <= max_insertion_pos]))[
            :3
        ]

        if not positions:
            adv_prompts.append(prompt)
            trigger_positions.append(0)
            metadata_list.append(_create_empty_metadata(i, prompt, label))
            continue

        result = gen_adv(
            model, prompt, label, candidate_tokens, positions, label_tokens, batch
        )
        (
            adv_prompt,
            trigger_pos,
            num_tokens,
            inserted_tokens,
            insertion_positions,
            objective_k,
            relative_improvements,
            ablation_marginals,
        ) = result

        adv_decoded = " ".join(
            [model.to_string(torch.tensor([tid])) for tid in inserted_tokens]
        )

        metadata_entry = {
            "id": i,
            "clean_prompt": prompt,
            "clean_label": int(label),
            "adv_tokens": [int(tid) for tid in inserted_tokens],
            "adv_decoded": adv_decoded,
            "positions": [int(pos) for pos in insertion_positions],
            "objective_k": {
                "k1": (
                    float(objective_k["k1"]) if objective_k["k1"] is not None else None
                ),
                "k2": (
                    float(objective_k["k2"]) if objective_k["k2"] is not None else None
                ),
                "k3": (
                    float(objective_k["k3"]) if objective_k["k3"] is not None else None
                ),
            },
            "relative_improvements": {
                "r2": (
                    float(relative_improvements["r2"])
                    if relative_improvements["r2"] is not None
                    else None
                ),
                "r3": (
                    float(relative_improvements["r3"])
                    if relative_improvements["r3"] is not None
                    else None
                ),
            },
            "ablation_marginals": [float(m) for m in ablation_marginals],
            "chosen_k": int(num_tokens),
        }

        with open(metadata_file, "a") as f:
            f.write(json.dumps(metadata_entry) + "\n")

        adv_prompts.append(adv_prompt)
        trigger_positions.append(trigger_pos)
        metadata_list.append(metadata_entry)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(prompts)} prompts")

    metadata = {
        "num_tokens": [m["chosen_k"] for m in metadata_list],
        "inserted_tokens": [m["adv_tokens"] for m in metadata_list],
        "insertion_positions": [m["positions"] for m in metadata_list],
        "full_metadata": metadata_list,
    }

    return adv_prompts, trigger_positions, metadata
