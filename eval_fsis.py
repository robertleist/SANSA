from collections import defaultdict

import torch
import torch.nn.functional as F

from util.losses import loss_instances


def eval_batch(
        pred_instances_batch: list[torch.Tensor],
        pred_scores_batch: list[torch.Tensor],
        gt_instances_batch: list[torch.Tensor],
        exclude_first_k_shots: int,
        device: torch.device,
):
    batch_metrics = defaultdict(list)
    batch_matches = defaultdict(list)
    total_loss = torch.tensor(0.0).to(device)
    for b in range(len(pred_instances_batch)):
        preds = pred_instances_batch[b]  # List of Tensors [H, W]
        scores = pred_scores_batch[b] # List of Tensors [1]
        gt_instances = gt_instances_batch[b].to(device)  # [N_gt, H, W]
        preds = F.interpolate(
            preds,
            size=gt_instances.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        preds = preds.squeeze(1)  # [N, H, W]
        scores = torch.stack(scores).squeeze(-1)  # [N]
        if len(preds) == 0 or len(gt_instances) == 0:
            continue

        # Very small matching threshold during learning, we want to match all predictions to GTs (even if low IoU) to
        # provide a learning signal. The loss function should handle the case of poor matches appropriately.
        loss, metrics, matches = loss_instances(
            preds,
            gt_instances,
            scores,
            matching_iou=1e-6,
            exclude_pred_ids=[i for i in range(exclude_first_k_shots)],
        )
        total_loss += loss
        for k, v in metrics.items():
            batch_metrics[k].append(v)
        for _, match_id, _ in matches:
            batch_matches[b].append(match_id)
    return total_loss, batch_metrics, batch_matches
