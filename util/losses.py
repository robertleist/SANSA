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


def dice_loss(inputs, targets, smooth=1.0):
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    return 1 - dice


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
    loss_objectness = F.binary_cross_entropy(p_score, gt_score)

    metrics["loss_dice"] += loss_dice.detach().item()
    metrics["loss_ce"] += loss_ce.detach().item()
    metrics["loss_objectness"] += loss_objectness.detach().item()

    return dice_factor * loss_dice + bce_factor * loss_ce + objectness_factor * loss_objectness


def loss_instances(
        preds: list,
        gt_masks: torch.Tensor,
        scores: torch.Tensor,
        matching_iou: float = 0.5,
        dice_factor: float = 1.,
        bce_factor: float = 1.,
        objectness_factor: float = 1.,
):
    # 1. Get our assignments
    # Assuming hungarian_matching returns:
    # matches: list[(p_idx, g_idx)], unmatched_preds: list[p_idx], unmatched_gts: list[g_idx]
    matches, FP_indices, FN_indices = hungarian_matching(preds, gt_masks, iou_threshold=matching_iou)

    total_loss = torch.tensor(0.0, device=gt_masks.device)
    metrics = {"loss_dice": 0., "loss_ce": 0., "loss_objectness": 0.}
    TP = len(matches)
    FP = len(FP_indices)
    FN = len(FN_indices)
    # recall = TP / (TP + FN)
    # objectness_factor += (1  - recall)  # Does this make sense? It is supposed to encourage the model to predict more objects, so when it starts to predict less objects this part gets amplified

    # --- 2. MATCHED LOSS (True Positives) ---
    # Goal: Refine the shape of correctly identified instances
    if len(matches) > 0:
        for p_idx, g_idx, _ in matches:
            p_mask = preds[p_idx]
            p_score = scores[p_idx]
            gt_mask = gt_masks[g_idx].float().view_as(p_mask)
            gt_score = torch.tensor(1.0, device=gt_mask.device)

            total_loss += combined_instance_loss(
                p_mask, p_score, gt_mask, gt_score,
                metrics, dice_factor, bce_factor, objectness_factor)

    # Force unmatched masks to be 0
    # There should always be at least one extra mask, which has a score and all pixels towards 0.0
    for i, p_idx in enumerate(FP_indices):
        p_mask = preds[p_idx]
        p_score = scores[p_idx]
        gt_mask = torch.zeros_like(p_mask)
        gt_score = torch.tensor(0.0, device=gt_mask.device)

        total_loss += combined_instance_loss(
            p_mask, p_score, gt_mask, gt_score,
            metrics, dice_factor, bce_factor, objectness_factor)

    num_preds = TP + FP
    metrics = {k: v / num_preds for k, v in metrics.items()}
    metrics["recall"] = TP / (TP + FN)
    metrics["precision"] = TP / (TP + FP)
    metrics["accuracy"] = TP / (TP + FN + FP)
    metrics["f1-score"] = 2 * TP / (2 * TP + FP + FN)
    return total_loss / (TP + FN), metrics
