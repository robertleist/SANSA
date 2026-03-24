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
from util.promptable_utils import build_prompt_dict_fsis


def backprop_and_log(
        optimizer,
        loss,
        model,
        lr_scheduler,
        max_norm: float = 0.0,
):
    if not loss.requires_grad:
        print(f"ERROR: Loss does not require gradients! Loss value: {loss.item():.6f}")
        return 0.0

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()

    if lr_scheduler is not None:
        lr_scheduler.step()

    return grad_total_norm


def log_metrics(metrics, step, metric_logger, lr, grad_total_norm, loss):
    avg_metrics = {k: np.average(v) for k, v in metrics.items()}
    metric_logger.update(loss=loss)
    metric_logger.update(**avg_metrics)
    metric_logger.update(lr=lr)
    metric_logger.update(grad_norm=grad_total_norm)
    mlflow.log_metric("loss", loss, step=step)
    mlflow.log_metrics(avg_metrics, step=step)
    mlflow.log_metrics({"lr": lr, "grad_norm": grad_total_norm}, step=step)


def compute_cellpose_loss(
        pred_flows_batch,
        gt_cellpose_batch,
        device: torch.device,
):
    loss = torch.tensor(0.0, device=device)
    for pred_seq, gt in zip(pred_flows_batch, gt_cellpose_batch):
        # pred_seq is [T, 3, H, W] where T iterations. take last iteration
        if isinstance(pred_seq, list):
            pred_flow = pred_seq[-1]
        else:
            pred_flow = pred_seq
        if pred_flow.shape != gt.shape:
            pred_flow = F.interpolate(pred_flow.unsqueeze(0), size=gt.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)
        loss = loss + F.mse_loss(pred_flow, gt)

    return loss / max(1, len(gt_cellpose_batch))


def _forward_and_optimize_iterations(
        model,
        images: torch.Tensor,
        instances_batch: list,
        prompt_dict: dict,
        max_iterations: int,
        optimizer: Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        max_norm: float,
        device: torch.device,
        optimize_iteration: bool,
        args: object,
        cellpose_targets_batch=None,
):
    batch_loss = torch.tensor(0.0, device=device)
    batch_outputs = defaultdict(lambda: defaultdict(list))
    batch_metrics = defaultdict(list)
    memory_batch = defaultdict(dict)
    grad_total_norm = 0.0

    for current_iteration in range(max_iterations):
        iter_outputs, memory_batch = model.forward(
            image_batch=images,
            prompt_batch=prompt_dict,
            memory_batch=memory_batch,
            current_iteration=current_iteration,
        )

        for b in range(len(iter_outputs)):
            for k, v in iter_outputs[b].items():
                batch_outputs[b][k].append(v)

        if cellpose_targets_batch is not None:
            pred_flows = [batch_outputs[b]["flows"][-1] for b in range(len(iter_outputs))]
            iter_loss = compute_cellpose_loss(pred_flows, cellpose_targets_batch, device)
            iter_metrics = {"loss_cellpose": iter_loss.item()}
        else:
            raise ValueError("cellpose_targets are required for Cellpose engine")

        if optimize_iteration:
            grad_total_norm = backprop_and_log(optimizer, iter_loss, model, lr_scheduler, max_norm)
            iter_loss = iter_loss.detach()

        torch.cuda.empty_cache()
        batch_loss = batch_loss + iter_loss

    return batch_loss, batch_outputs, batch_metrics, memory_batch, grad_total_norm


def train_one_epoch(
        model,
        data_loader: Iterable,
        optimizer: Optimizer,
        device: torch.device,
        epoch: int,
        max_norm: float = 0.0,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        args: Optional[object] = None,
) -> dict:
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1
    best_score = np.inf

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        global_step = epoch * len(data_loader) + i
        images = batch['image'].to(device)
        instances_batch = batch['instances']
        cellpose_targets = batch.get('cellpose_target', None)
        if cellpose_targets is not None:
            cellpose_targets = cellpose_targets.to(device)

        mlflow.log_metric("num_instances", max(len(masks) for masks in instances_batch), step=global_step)
        max_instances = max(len(masks) for masks in instances_batch)
        if max_instances > 20:
            max_iterations = min(20, int(0.8 * max_instances))
        else:
            max_iterations = min(30, int(1.2 * max_instances))

        prompt_dict = build_prompt_dict_fsis(
            instances_batch,
            args.prompt,
            num_support_prompts=args.shots,
            train_mode=True,
            device=model.device
        )

        batch_loss, batch_outputs, batch_metrics, memory_batch, grad_total_norm = _forward_and_optimize_iterations(
            model=model,
            images=images,
            instances_batch=instances_batch,
            prompt_dict=prompt_dict,
            max_iterations=max_iterations,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            max_norm=max_norm,
            device=device,
            optimize_iteration=args.optimize_iteration,
            args=args,
            cellpose_targets_batch=cellpose_targets,
        )

        if batch_loss.item() < best_score:
            best_score = batch_loss.item()

        if not math.isfinite(batch_loss.item()):
            print(f"Loss is {batch_loss.item()}, stopping training")
            sys.exit(1)

        if not args.optimize_iteration:
            grad_total_norm = backprop_and_log(optimizer, batch_loss, model, lr_scheduler, max_norm)

        log_metrics(batch_metrics, global_step, metric_logger, optimizer.param_groups[0]["lr"], grad_total_norm, batch_loss.item())

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
