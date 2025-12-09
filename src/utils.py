"""
Shared utility functions across modules
"""



def build_position_map(clean_tokens, adv_tokens, insertion_positions):
    """
    Build mapping from clean token positions to adversarial token positions.

    Args:
        clean_tokens: Clean token tensor [1, clean_seq_len]
        adv_tokens: Adversarial token tensor [1, adv_seq_len]
        insertion_positions: List of positions where tokens were inserted in clean sequence

    Returns:
        Dictionary mapping clean_pos -> adv_pos
    """
    clean_to_adv_map = {}
    clean_len = clean_tokens.shape[1]
    adv_len = adv_tokens.shape[1]

    if insertion_positions:
        sorted_insertions = sorted(insertion_positions)
        for clean_pos in range(clean_len):
            insertions_before = sum(
                1 for ins_pos in sorted_insertions if ins_pos <= clean_pos
            )
            adv_pos = clean_pos + insertions_before
            if adv_pos < adv_len:
                clean_to_adv_map[clean_pos] = adv_pos
    else:
        for pos in range(min(clean_len, adv_len)):
            clean_to_adv_map[pos] = pos

    return clean_to_adv_map
