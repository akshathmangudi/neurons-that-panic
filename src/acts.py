"""
Activation extraction + hooks
"""


import torch


def get_hooks(model):
    """Get default hook points for MLP and attention outputs."""
    hook_points = []
    for layer_idx in range(model.cfg.n_layers):
        hook_points.append(f"blocks.{layer_idx}.hook_mlp_out")
        hook_points.append(f"blocks.{layer_idx}.attn.hook_z")
    return hook_points


def extract(model, prompts, hooks=None, batch=16):
    """Extract activations for a list of prompts."""
    if hooks is None:
        hooks = get_hooks(model)

    activations = {hook: [] for hook in hooks}

    # Find max sequence length
    max_seq_len = 0
    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        max_seq_len = max(max_seq_len, tokens.shape[1])

    for i in range(0, len(prompts), batch):
        batch_prompts = prompts[i : i + batch]
        batch_tokens = model.to_tokens(batch_prompts)

        with torch.no_grad():
            _, cache = model.run_with_cache(batch_tokens)

        # Extract and pad activations
        for hook in hooks:
            if hook in cache:
                hook_acts = cache[hook].cpu()
                seq_len = hook_acts.shape[1]

                # Pad if needed
                if seq_len < max_seq_len:
                    pad_shape = list(hook_acts.shape)
                    pad_shape[1] = max_seq_len - seq_len
                    padding = torch.zeros(
                        pad_shape, dtype=hook_acts.dtype, device=hook_acts.device
                    )
                    hook_acts = torch.cat([hook_acts, padding], dim=1)

                activations[hook].append(hook_acts)

    # Concatenate batches
    for hook in hooks:
        if activations[hook]:
            activations[hook] = torch.cat(activations[hook], dim=0)
        else:
            activations[hook] = None

    return activations
