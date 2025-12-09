"""
Main CLI entry point for "Neurons That Panic"
"""

import argparse
import json
import os
import platform
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from src import acts, adv, dn, patch, plot
from src.data import load_prompts
from src.model import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Neurons That Panic")

    parser.add_argument(
        "--model",
        type=str,
        default="EleutherAI/pythia-160m-deduped",
        help="HuggingFace model path",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sst2",
        help="Dataset identifier or path",
    )
    parser.add_argument(
        "--save", default="artifacts/", help="Directory to save results."
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for processing prompts."
    )
    parser.add_argument(
        "--k", type=int, default=20, help="Number of panic components to rank."
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=200,
        help="How many prompts to run the experiment on.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for inference."
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=500,
        help="Number of candidate tokens for adversarial generation.",
    )
    parser.add_argument(
        "--n-random",
        type=int,
        default=5,
        help="Number of random trials for patching baseline.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    return parser.parse_args()


def main(args):
    # Set seeds
    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Create output directory
    os.makedirs(args.save, exist_ok=True)
    os.makedirs(os.path.join(args.save, "plots"), exist_ok=True)
    os.makedirs(os.path.join(args.save, "tables"), exist_ok=True)

    # 1. Load model
    if args.verbose:
        print("Loading model...")
    try:
        model = load_model(args.model, device=args.device)
    except Exception as e:
        print(f"Failed to execute pipeline: {e}")
        return

    # 2. Load data
    if args.verbose:
        print("Loading prompts...")
    clean_prompts, labels = load_prompts(args.num_prompts, args.dataset)

    # Save clean prompts
    with open(os.path.join(args.save, "clean_prompts.txt"), "w") as f:
        for prompt in clean_prompts:
            f.write(prompt + "\n")

    # 3. Make adversarial prompts
    if args.verbose:
        print("Generating adversarial prompts...")
    adv_prompts, trigger_positions, metadata = adv.gen_all(
        model,
        clean_prompts,
        labels,
        n_cand=args.n_candidates,
        batch=32,
        out_dir=args.save,
    )

    # Save adversarial prompts
    with open(os.path.join(args.save, "adv_prompts.txt"), "w") as f:
        for prompt in adv_prompts:
            f.write(prompt + "\n")

    # 4. Get activations
    if args.verbose:
        print("Extracting activations...")
    acts_clean = acts.extract(model, clean_prompts, batch=args.batch_size)
    acts_adv = acts.extract(model, adv_prompts, batch=args.batch_size)

    # Save activations
    torch.save(acts_clean, os.path.join(args.save, "activations_clean.pt"))
    torch.save(acts_adv, os.path.join(args.save, "activations_adv.pt"))

    # 5. Compute delta norm
    if args.verbose:
        print("Computing delta-norm...")
    df = dn.compute(acts_clean, acts_adv, trigger_positions, metadata, model)
    df = dn.rank(df)

    # Save panic components
    df.to_csv(os.path.join(args.save, "panic_components.csv"), index=False)

    # 6. Run patching
    if args.verbose:
        print("Running causal patching...")
    patch_results = patch.patch(
        model,
        clean_prompts,
        adv_prompts,
        labels,
        df,
        metadata,
        top_k=args.k,
        n_random=args.n_random,
        batch=args.batch_size,
    )

    # Save patch results
    patch_df = pd.DataFrame(
        {
            "top_k": [patch_results["top_k"]],
            "mean_recovery": [patch_results["mean_recovery"]],
            "mean_random_recovery": [patch_results["mean_random_recovery"]],
            "std_random_recovery": [patch_results["std_random_recovery"]],
        }
    )
    patch_df.to_csv(os.path.join(args.save, "patch_results.csv"), index=False)

    # 7. Generate plots
    if args.verbose:
        print("Generating plots...")
    plot.setup_style()
    plot.layerwise(df, os.path.join(args.save, "plots/layerwise_delta_norm.png"))
    plot.positional(
        df,
        top_k=args.k,
        out_path=os.path.join(args.save, "plots/position_localization.png"),
    )
    plot.distributions(
        df, os.path.join(args.save, "plots/delta_norm_distributions.png")
    )
    plot.patching(
        patch_results, os.path.join(args.save, "plots/causal_patching_results.png")
    )

    # 8. Save run manifest
    run_manifest = {
        "model_id": args.model,
        "model_config": {
            "n_layers": model.cfg.n_layers,
            "d_model": model.cfg.d_model,
            "d_mlp": model.cfg.d_mlp,
            "n_heads": model.cfg.n_heads,
            "d_vocab": model.cfg.d_vocab,
            "n_ctx": model.cfg.n_ctx,
        },
        "versions": {"pytorch": torch.__version__, "python": platform.python_version()},
        "hardware": {
            "device": args.device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "experiment_config": {
            "n_prompts": len(clean_prompts),
            "n_candidates": args.n_candidates,
            "batch_size": args.batch_size,
            "seed": SEED,
            "top_k_patching": args.k,
            "n_random_trials": args.n_random,
        },
        "timestamp": datetime.now().isoformat(),
    }

    with open(os.path.join(args.save, "run_manifest.json"), "w") as f:
        json.dump(run_manifest, f, indent=2)

    if args.verbose:
        print("Experiment complete!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
