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

    total_loss = torch.tensor(0.0, device=gt_masks.device)
    metrics = {"loss_dice": 0., "loss_ce": 0., "loss_objectness": 0.}
    TP = len(matches)
    FP = len(FP_indices)  # Should always be 0
    FN = len(FN_indices)

    # Regularization
    total_loss += regularize_overlap(preds)

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
            total_loss += fit["dice"].sum()  # Sum across matched instances
            
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
    return total_loss / normalize, metrics, matches


def goodness_of_fit(
        preds: torch.Tensor,
        gts: torch.Tensor,
        smooth: float = 1e-6,
):
    assert len(preds) == len(gts), f"Cannot compute goodness of fit of different length for preds and GT: {len(preds)} != {len(gts)}"
    n_instances = len(preds)
    preds = preds.view(n_instances, -1)
    gts = gts.view(n_instances, -1)

    intersection = torch.sum(torch.mul(preds, gts), dim=1)
    preds_union = torch.sum(preds, dim=1)
    gts_union = torch.sum(gts, dim=1)
    dice = (2. * intersection + smooth) / (preds_union + gts_union + smooth)
    iou = (intersection + smooth )/ (preds_union + gts_union + smooth)

    return {
        "dice": dice,
        "iou": iou,
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
        smooth: float = 1e-6,
):
    """ Regularize the overlap between predicted masks. Forces the model to make novel predictions. """
    if predicted_masks.shape[0] < 2:
        return torch.tensor(0.0, device=predicted_masks.device)

    # 1. Compute the global union: \cup \hat{f}
    # For soft masks, the union is often approximated by the element-wise max
    global_union = torch.max(predicted_masks, dim=0)[0]
    global_union_area = torch.sum(global_union) + smooth

    total_intersection_sum = 0.0
    num_instances = predicted_masks.shape[0]

    # 2. Iterate through each instance E_j
    for j in range(num_instances):
        e_j = predicted_masks[j]

        # 3. Compute the union of all other instances: \cup_{E_l \setminus E_j}
        # We mask out the current index and take the max of the rest
        others = torch.cat([predicted_masks[:j], predicted_masks[j + 1:]], dim=0)
        union_others = torch.max(others, dim=0)[0]

        # 4. Compute intersection: E_j \cap (union_others)
        # For soft masks, intersection is element-wise multiplication
        intersection = e_j * union_others

        # 5. Sum the area of the intersection
        total_intersection_sum += torch.sum(intersection)

    # 6. Final calculation: theta * (sum of intersections / global union area)
    loss = (total_intersection_sum / global_union_area)

    return loss
