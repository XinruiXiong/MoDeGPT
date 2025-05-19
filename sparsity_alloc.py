import math
import logging

logger = logging.getLogger("MoDeGPT")

def allocate_global_sparsity(bi_scores, model_config, target_keep_ratio: float, temperature: float = 1.0,
                              min_keep: float = 0.05, max_keep: float = 0.95):
    """
    Allocate layer-wise retention ratio φ_i based on BI scores.

    Implements softmax allocation with optional min/max constraints and normalization.

    Layer 0 is skipped from compression and assigned keep_ratio = 1.0.

    Args:
        bi_scores (List[float]): Per-layer BI scores
        model_config: HF model config
        target_keep_ratio (float): Global average keep ratio (e.g., 0.5)
        temperature (float): Softmax temperature
        min_keep (float): Minimum per-layer keep ratio
        max_keep (float): Maximum per-layer keep ratio

    Returns:
        List[float]: Per-layer keep ratios φ_i
    """
    n_layers = len(bi_scores)
    expected_layers = getattr(model_config, "num_hidden_layers", None) \
                   or getattr(model_config, "n_layer", None) \
                   or getattr(model_config, "num_layers", None)
    if expected_layers is None or n_layers != expected_layers:
        raise ValueError(f"Mismatch between BI scores ({n_layers}) and model layers ({expected_layers})")

    logger.info(f"Allocating global sparsity using target keep ratio {target_keep_ratio:.4f} and temperature {temperature}")

    # === Step 1: Fix Layer 0 ===
    keep_ratios = [1.0]  # Layer 0 is always kept

    # === Step 2: Compute softmax for layers 1 to n-1 ===
    bi_sub = bi_scores[1:]
    max_score = max(bi_sub)
    shifted = [s - max_score for s in bi_sub]  # numerical stability
    raw_importance = [math.exp(-s / temperature) for s in shifted]
    total = sum(raw_importance)
    normalized = [x / total for x in raw_importance]

    # === Step 3: Allocate keep_ratios for remaining layers ===
    remaining_budget = target_keep_ratio * n_layers - 1.0  # subtract Layer 0's fixed 1.0
    initial = [w * remaining_budget for w in normalized]
    clipped = [min(max(r, min_keep), max_keep) for r in initial]

    # === Step 4: Renormalize clipped layers if needed ===
    clipped_sum = sum(clipped)
    if not math.isclose(clipped_sum, remaining_budget, rel_tol=1e-3):
        scale = remaining_budget / clipped_sum
        clipped = [min(max(r * scale, min_keep), max_keep) for r in clipped]

    # === Step 5: Combine final keep_ratios ===
    keep_ratios.extend(clipped)

    # === Step 6: Logging ===
    for i, (score, ratio) in enumerate(zip(bi_scores, keep_ratios)):
        logger.info(f"Layer {i:2d}: keep_ratio = {ratio:.4f} (BI = {score:.4f})")

    return keep_ratios
