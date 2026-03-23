from typing import Dict

import einops
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_boxes: int) -> torch.Tensor:
    """
    Compute the Dice loss for binary masks.

    Args:
        inputs (Tensor): raw logits, shape [N, ...].
        targets (Tensor): binary masks with same shape as inputs.
        num_boxes (int): normalization factor (usually batch size).

    Returns:
        Tensor: scalar Dice loss.
    """
    inputs = inputs.sigmoid().flatten(1).float()
    targets = targets.flatten(1).float()

    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(1) + targets.sum(1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_boxes: int,
        alpha: float = 0.25,
        gamma: float = 2.0,
) -> torch.Tensor:
    """
    Compute the sigmoid focal loss (used in RetinaNet).

    Args:
        inputs (Tensor): raw logits, shape [N, ...].
        targets (Tensor): binary masks with same shape as inputs.
        num_boxes (int): normalization factor (usually batch size).
        alpha (float): class balancing factor (0–1). Default: 0.25.
        gamma (float): focusing parameter. Default: 2.0.

    Returns:
        Tensor: scalar focal loss.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def loss_masks(
        outputs: torch.Tensor,
        masks: torch.Tensor,
        num_frames: int,
) -> Dict[str, torch.Tensor]:
    """
    Compute focal and dice loss on predicted masks.

    Args:
        outputs (Tensor): B*T, h, W.
        masks (Tensor): B, T, H, W (binary 0/1).
        num_frames (int): number of last frames to include (T → all, T-1 → skip first).

    Returns:
        dict: {"loss_mask": Tensor, "loss_dice": Tensor}
    """
    bs, T = masks.shape[:2]
    start = max(0, T - num_frames)

    src = einops.rearrange(outputs, '(b t) h w -> b t h w', b=bs)[:, start:T]
    tgt = masks[:, start:]

    # flatten to [B, F*H*W]
    tgt = tgt.to(src)
    src = src.flatten(1).to(torch.float32)
    tgt = tgt.flatten(1).to(src.dtype)

    # drop NaN rows (if any)
    keep = ~torch.isnan(src).any(dim=1)
    if not keep.all():
        src, tgt = src[keep], tgt[keep]
        bs = int(keep.sum())

    return {
        "loss_mask": sigmoid_focal_loss(src, tgt, bs),
        "loss_dice": dice_loss(src, tgt, bs),
    }


def hungarian_matching(
        preds: list,
        gt_masks: torch.Tensor,
        iou_threshold: float = 0.5,
):
    """
    Args:
        preds: List of [H, W] tensors (logits)
        gt_masks: [N_gt, H, W] tensor
    Returns:
        matched_indices: List of tuples (pred_idx, gt_idx)
    """
    if len(preds) == 0 or len(gt_masks) == 0:
        return []

    num_preds = len(preds)
    num_gts = gt_masks.shape[0]

    # 1. Initialize Cost Matrix [num_preds, num_gts]
    # We use (1 - IoU) as the cost because the algorithm minimizes total cost.
    cost_matrix = torch.zeros((num_preds, num_gts))

    with torch.no_grad():
        for i, p_mask in enumerate(preds):
            p_sig = p_mask.sigmoid().squeeze()

            # Calculate IoU for all GTs at once (vectorized)
            # Intersection: [N_gt], Union: [N_gt]
            intersection = (p_sig * gt_masks).sum(dim=(-1, -2))
            union = p_sig.sum() + gt_masks.sum(dim=(-1, -2)) - intersection
            ious = intersection / (union + 1e-6)

            cost_matrix[i] = 1.0 - ious

    # 2. Solve the Linear Sum Assignment Problem
    # This finds the indices that minimize the total cost
    row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())

    final_matches = []
    matched_preds_set = set()
    matched_gts_set = set()

    # 3. Apply the 0.5 IoU Threshold
    for p_idx, g_idx in zip(row_ind, col_ind):
        cost = cost_matrix[p_idx, g_idx].item()
        iou = 1.0 - cost
        if iou >= iou_threshold:
            final_matches.append((p_idx, g_idx, iou))
            matched_preds_set.add(p_idx)
            matched_gts_set.add(g_idx)

    # 4. Anything not meeting the threshold is "unmatched"
    unmatched_preds = [i for i in range(num_preds) if i not in matched_preds_set]
    unmatched_gts = [i for i in range(num_gts) if i not in matched_gts_set]

    return final_matches, unmatched_preds, unmatched_gts





def combined_instance_loss(
        p_mask,
        p_score,
        gt_mask,
        gt_score,
        metrics: dict,
        dice_factor: float = 1.,
        bce_factor: float = 1.,
        objectness_factor: float = 1.,
):
    # Use a combo of BCE and Dice for stable gradients
    loss_dice = dice_loss(p_mask.sigmoid(), gt_mask)
    loss_ce = F.binary_cross_entropy_with_logits(p_mask, gt_mask)
    loss_objectness = F.binary_cross_entropy(p_score.squeeze(), gt_score)

    metrics["loss_dice"] += loss_dice.detach().item()
    metrics["loss_ce"] += loss_ce.detach().item()
    metrics["loss_objectness"] += loss_objectness.detach().item()

    return dice_factor * loss_dice + bce_factor * loss_ce + objectness_factor * loss_objectness


def loss_instances(
        preds: list,
        gt_masks: torch.Tensor,
        scores: torch.Tensor,
        matching_iou: float = 0.,
        exclude_pred_ids: list[int] = None,
):
    # 1. Get our assignments
    # Assuming hungarian_matching returns:
    # matches: list[(p_idx, g_idx)], unmatched_preds: list[p_idx], unmatched_gts: list[g_idx]
    matches, FP_indices, FN_indices = hungarian_matching(preds, gt_masks, iou_threshold=matching_iou)

    # Initialize total_loss to ensure it can accumulate gradients
    # Use a small epsilon to ensure gradients can flow even if all components are zero
    total_loss = torch.tensor(1e-8, device=gt_masks.device, requires_grad=True)
    metrics = {"loss_dice": 0., "loss_ce": 0., "loss_objectness": 0.}
    TP = len(matches)
    FP = len(FP_indices)  # Should always be 0
    FN = len(FN_indices)

    # Regularization: penalize overlapping predictions
    # Ensure regularization operates on sigmoid'd predictions for proper gradients
    if len(preds) > 0:
        pred_masks_sigmoid = torch.stack([p.sigmoid() for p in preds])
        regularization_loss = regularize_overlap(pred_masks_sigmoid, weight=0.1)
        total_loss = total_loss + regularization_loss
    else:
        regularization_loss = torch.tensor(0.0, device=gt_masks.device)

    # --- 2. MATCHED LOSS (True Positives) ---
    # Goal: Refine the shape of correctly identified instances
    if len(matches) > 0:
        p_indices, g_indices, scores_matched = [], [], []
        for p_idx, g_idx, _ in matches:
            # Filter matches; used to filter out prompted predictions
            if exclude_pred_ids and p_idx in exclude_pred_ids:
                continue
            p_indices.append(p_idx)
            g_indices.append(g_idx)
            # Track objectness scores for matched instances
            if p_idx < len(scores):
                scores_matched.append(scores[p_idx])
        
        if len(p_indices) > 0:
            p_masks = torch.stack([preds[i] for i in p_indices])
            gt_masks_matched = torch.stack([gt_masks[i] for i in g_indices]).float()

            fit = goodness_of_fit(p_masks, gt_masks_matched)
            # Use dice_loss: higher when predictions are BAD, lower when GOOD
            total_loss += fit["dice_loss"].sum()  # Sum across matched instances
            
            # Add objectness loss for matched instances
            if len(scores_matched) > 0:
                scores_matched = torch.stack(scores_matched).squeeze(-1)  # [N_matched]
                objectness_loss = F.binary_cross_entropy(scores_matched, torch.ones_like(scores_matched))
                total_loss += objectness_loss

    normalize = TP + FN - (len(exclude_pred_ids) if exclude_pred_ids is not None else 0)
    
    # Guard against division by zero
    if normalize <= 0:
        normalize = 1
    
    metrics = {k: v / normalize for k, v in metrics.items()}

    # Normalize by the minimum of the missed or additional predictions.
    # If we predict too many instances, we normalize by the number of actual instances
    # If we predict too few instances, we normalize by the number of predictions
    final_loss = total_loss / normalize
    
    # CRITICAL: Ensure loss requires gradients during training
    # If it doesn't, there's a bug in gradient flow that needs fixing
    if not final_loss.requires_grad and len(preds) > 0:
        # This should never happen during training - indicates gradient flow is broken
        print(f"WARNING: Loss does not require gradients! preds shape: {[p.shape for p in preds]}, "
              f"matches: {len(matches)}, regularization: {regularization_loss.item():.6f}")
        # Force gradients by adding a small gradient-requiring term
        final_loss = final_loss + torch.tensor(0.0, device=final_loss.device, requires_grad=True)
    
    return final_loss, metrics, matches


def goodness_of_fit(
        preds: torch.Tensor,
        gts: torch.Tensor,
        smooth: float = 1e-6,
):
    """Compute Dice and IoU losses for instance predictions.
    
    Args:
        preds: [N, ...] predicted masks (raw logits from model)
        gts: [N, ...] ground truth binary masks
        smooth: smoothing factor for numerical stability
    
    Returns:
        dict with loss tensors (lower = better):
          - "dice_loss": [N] Dice loss per instance (0=perfect, ~1=worst)
          - "iou_loss": [N] IoU loss per instance (0=perfect, ~1=worst)
    """
    assert len(preds) == len(gts), f"Cannot compute goodness of fit of different length for preds and GT: {len(preds)} != {len(gts)}"
    n_instances = len(preds)
    # Convert logits to probabilities for proper loss computation
    preds_sigmoid = preds.sigmoid().view(n_instances, -1)
    gts = gts.view(n_instances, -1)

    intersection = torch.sum(torch.mul(preds_sigmoid, gts), dim=1)
    preds_union = torch.sum(preds_sigmoid, dim=1)
    gts_union = torch.sum(gts, dim=1)
    
    # Compute metrics (higher = better)
    dice_metric = (2. * intersection + smooth) / (preds_union + gts_union + smooth)
    iou_metric = (intersection + smooth) / (preds_union + gts_union - intersection + smooth)
    
    # Convert to losses (lower = better)
    dice_loss = 1.0 - dice_metric
    iou_loss = 1.0 - iou_metric

    return {
        "dice_loss": dice_loss,
        "iou_loss": iou_loss,
    }


def regularize_count(
        num_pred: int,
        num_exemplars: int,
        smooth: float = 1e-6,
):
    """ Regularize the number of predicted masks. Forces the model to make novel predictions. """
    return num_exemplars / (num_pred + smooth)


def regularize_overlap(
        predicted_masks: torch.Tensor,
        weight: float = 1.0,
        smooth: float = 1e-6,
):
    """Regularize overlap between predicted masks to encourage instance separation.
    
    Args:
        predicted_masks: [N, H, W] predicted masks (should be sigmoid'd, values in [0,1])
        weight: scaling factor for the loss (default 1.0, increase to penalize overlaps more)
        smooth: numerical stability factor
    
    Returns:
        Scalar loss tensor (penalizes overlapping masks, 0=no overlap, >0=overlap)
    """
    if predicted_masks.shape[0] < 2:
        return torch.tensor(0.0, device=predicted_masks.device)

    # Compute pairwise overlaps between all instance pairs
    n_instances = predicted_masks.shape[0]
    overlap_loss = torch.tensor(0.0, device=predicted_masks.device)
    
    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            mask_i = predicted_masks[i].flatten()
            mask_j = predicted_masks[j].flatten()
            
            # Compute intersection: overlapping area
            intersection = torch.sum(mask_i * mask_j)
            # Compute union: area covered by at least one mask
            union = torch.sum(torch.clamp(mask_i + mask_j, max=1.0))
            
            # Overlap ratio: high when masks overlap significantly
            # Should be low (near 0) for well-separated instances
            overlap = intersection / (union + smooth)
            overlap_loss = overlap_loss + overlap
    
    # Average over all pairs and scale by weight
    num_pairs = max(1, n_instances * (n_instances - 1) / 2)
    overlap_loss = (overlap_loss / num_pairs) * weight
    
    return overlap_loss
