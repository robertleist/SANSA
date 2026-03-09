import math
import random
import sys
from collections import defaultdict
from typing import Iterable, Optional

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer

import util.misc as utils
from eval_fsis import eval_batch
from models.sansa.inst_sansa import InstanceSANSA
from util.promptable_utils import build_prompt_dict_fsis


def backprop_and_log(
        optimizer,
        loss,
        model,
        lr_scheduler,
        max_norm: float = 0.0,
):
    optimizer.zero_grad(set_to_none=True)
    if loss.requires_grad:
        loss.backward()

        grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
    else:
        grad_total_norm = 0.0
        print("Skipping backward: No gradients to compute for this batch.")

    if lr_scheduler is not None:
        lr_scheduler.step()

    return grad_total_norm


def log_metrics(metrics, step, metric_logger, lr, grad_total_norm, loss):
    # Logging
    avg_metrics = {k: np.average(v) for k, v in metrics.items()}
    metric_logger.update(loss=loss)
    metric_logger.update(**avg_metrics)
    metric_logger.update(lr=lr)
    metric_logger.update(grad_norm=grad_total_norm)
    mlflow.log_metric("loss", loss, step=step)
    mlflow.log_metrics(avg_metrics, step=step)
    mlflow.log_metrics({
        "lr": lr,
        "grad_norm": grad_total_norm,
    }, step=step)


def detach_memory(memory_batch: dict[int, dict[int, dict[str, torch.Tensor]]]):
    for i, mem_bank in memory_batch.items():
        for j, mem_entry in mem_bank.items():
            for k, mem_tensors in mem_entry.items():
                if isinstance(mem_tensors, torch.Tensor):
                    memory_batch[i][j][k] = mem_tensors.detach()
                elif isinstance(mem_tensors, list):
                    memory_batch[i][j][k] = [list_tensor.detach() for list_tensor in mem_tensors]
    return memory_batch



def log_an_image(
        pred_instance_masks,
        gt_instance_masks,
        image,
        step
):
    # Log artifacts
    # Resize pred masks
    pred_instance_masks = torch.stack(pred_instance_masks, 0).squeeze(1)
    pred_instance_masks = F.interpolate(
        pred_instance_masks,
        size=gt_instance_masks.shape[-2:],
        mode='bilinear',
        align_corners=False,
    ).squeeze()
    pred_mask = pred_instance_masks[0]
    for instance_mask in pred_instance_masks[1:]:
        pred_mask = torch.logical_or(pred_mask, instance_mask)
    gt_mask = gt_instance_masks[0]
    for instance_mask in gt_instance_masks[1:]:
        gt_mask = torch.logical_or(gt_mask, instance_mask)
    intersection = torch.logical_and(pred_mask, gt_mask)
    pred_only = torch.logical_and(pred_mask, torch.logical_not(intersection))
    gt_only = torch.logical_and(gt_mask, torch.logical_not(intersection))
    # Stack the image to RGB
    # Red: Missed GT instances
    # Green: Found GT instances
    # Blue: Hallucinations
    mask_rgb = torch.stack([gt_only, intersection, pred_only], dim=-1)
    mlflow.log_image(
        image=mask_rgb.cpu().numpy(),
        key="sample_mask",
        step=step + 10_000,  # Fix for MLFLOW log image bug
    )
    mlflow.log_image(
        key="sample_image",
        image=image.permute(1, 2, 0).cpu().numpy(),
        step=step + 10_000,
    )


def train_one_epoch(
        model: InstanceSANSA,
        data_loader: Iterable,
        optimizer: Optimizer,
        device: torch.device,
        epoch: int,
        max_norm: float = 0.0,
        optimize_iteration: bool = True,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        args: Optional[object] = None,
) -> dict:
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1
    best_score = np.inf

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # 1. Unpack LVIS Batch
        # images: [B, C, H, W], instances_batch: [B, N_instances, H, W]
        images = batch['image'].to(device)
        instances_batch = batch['instances']  # List or padded tensor of masks
        og_sizes = batch['org_size']

        # Max iterations must be at least the number of instances of the most
        mlflow.log_metric("num_instances", max(len(masks) for masks in instances_batch), step=global_step)
        max_iterations = min(50, int(1.3 * max(len(masks) for masks in instances_batch)))
        global_step = epoch * len(data_loader) + i
        prompt_dict = build_prompt_dict_fsis(
            instances_batch,
            args.prompt,
            # Controls the type of prompt. In this case, it is an exemplar (mask). Could also be point or box.
            num_support_prompts=args.shots,  # Controls how many instances are supported by prompts.
            train_mode=True,
            device=model.device
        )

        # 2. Forward Pass
        # We run the iterative discovery loop inside the model
        batch_loss = torch.tensor(0.0, device=device)
        batch_outputs = defaultdict(lambda: defaultdict(list))
        batch_metrics = defaultdict(list)
        memory_batch = defaultdict(dict)
        matched_instances_per_batch = defaultdict(list)
        for current_iteration in range(max_iterations):
            iter_outputs, memory_batch = model.forward(
                    image_batch=images,
                    prompt_batch=prompt_dict,
                    memory_batch=memory_batch,
                    current_iteration=current_iteration,
                )
            # Append the iter output
            for b in range(len(iter_outputs)):
                for k, v in iter_outputs[b].items():
                    batch_outputs[b][k].append(v)
            unmatched_instances_per_batch = {}
            for j, instances in enumerate(instances_batch):
                unmatched_indices = [k for k in range(len(instances)) if k not in matched_instances_per_batch[j]]
                unmatched_instances_per_batch[j] = instances_batch[j][unmatched_indices]
            # Eval this iteration
            iter_loss, iter_metrics, iter_matches = eval_batch(
                pred_instances_batch=[output["masks"] for output in iter_outputs],
                pred_scores_batch=[output["scores"] for output in iter_outputs],
                gt_instances_batch=[instances.to(device) for instances in unmatched_instances_per_batch.values()],
                exclude_first_k_shots=0,
                device=device
            )
            for k, v in iter_matches.items():
                if isinstance(v, list) and len(v) > 0:
                    matched_instances_per_batch[k] += v
            # Decide whether we optimize at the iteration level or at the sequence level
            if optimize_iteration:
                # Causes error: "Trying to backward through the graph a second time"
                grad_total_norm = backprop_and_log(
                    optimizer,
                    iter_loss,
                    model,
                    lr_scheduler,
                    max_norm
                )
                detach_memory(memory_batch)
            else:
                # We sum the loss up
                batch_loss += iter_loss
                for k, v in iter_metrics:
                    batch_metrics[k] += v if isinstance(v, list) else [v]


        # 3. Sequential Greedy Matching & Loss Calculation
        if batch_loss.item() < best_score:
            best_score = batch_loss.item()
            random_index_from_batch = random.choice(range(len(images)))
            log_an_image(
                batch_outputs[random_index_from_batch]["masks"],
                instances_batch[random_index_from_batch].to(device),
                images[random_index_from_batch],
                i + 10_000,
            )


        if not math.isfinite(batch_loss.item()):
            print(f"Loss is {batch_loss.item()}, stopping training")
            sys.exit(1)
        if not optimize_iteration:
            # We optimize the sequence
            grad_total_norm = backprop_and_log(
                optimizer,
                batch_loss,
                model,
                lr_scheduler,
                max_norm
            )
        log_metrics(
            batch_metrics,
            global_step,
            metric_logger,
            optimizer.param_groups[0]["lr"],
            grad_total_norm,
            batch_loss.item()
        )


    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
