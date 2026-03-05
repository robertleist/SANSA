import math
import random
import sys
from collections import defaultdict
from math import ceil
from typing import Iterable, Optional, List, Dict, Any

import numpy as np
import torch
from numpy.ma.extras import average
from torch.nn import Module
from torch.optim import Optimizer
import torch.nn.functional as F
import mlflow
import util.misc as utils
from util.losses import loss_masks, loss_instances  # Assume this handles Dice/BCE for matched pairs
from util.promptable_utils import build_prompt_dict, build_prompt_dict_fsis


def train_one_epoch(
        model: Module,
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

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        global_step = epoch * len(data_loader) + i
        # 1. Unpack LVIS Batch
        # images: [B, C, H, W], instances_batch: [B, N_instances, H, W]
        images = batch['image'].to(device)
        instances_batch = batch['instances']  # List or padded tensor of masks
        og_sizes = batch['org_size']

        # Max iterations must be at least the number of instances of the most
        max_iterations = max(len(masks) for masks in instances_batch) + 5
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
        # outputs is now a list (len B) of dicts containing "masks" and "scores"
        batch_outputs = model(images, prompt_dict, max_iterations=max_iterations)

        batch_loss = torch.tensor(0.0, device=device)
        batch_metrics = defaultdict(list)
        # 3. Sequential Greedy Matching & Loss Calculation
        for b in range(len(batch_outputs)):
            preds = batch_outputs[b]["masks"]  # List of Tensors [H, W]
            scores = batch_outputs[b]["scores"]  # List of Tensors [1]
            gt_instances = instances_batch[b].to(device)  # [N_gt, H, W]
            og_size = og_sizes[b]

            # Resize predictions
            preds = F.interpolate(
                preds.unsqueeze(1),
                og_size,
                mode='bilinear',
            ).squeeze(1)

            if len(preds) == 0 or len(gt_instances) == 0:
                continue

            loss, metrics = loss_instances(
                preds,
                gt_instances,
                scores
            )
            batch_loss += loss
            for k, v in metrics.items():
                batch_metrics[k].append(v)

        if not math.isfinite(batch_loss.item()):
            print(f"Loss is {batch_loss.item()}, stopping training")
            sys.exit(1)

        optimizer.zero_grad(set_to_none=True)
        if batch_loss.requires_grad:
            batch_loss.backward()
            optimizer.step()
        else:
            print("Skipping backward: No gradients to compute for this batch.")

        grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # Logging
        avg_metrics = {k: np.average(v) for k, v in batch_metrics.items()}
        metric_logger.update(loss=batch_loss.item())
        metric_logger.update(**avg_metrics)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        if i % print_freq == 0:
            # Log primary loss and learning rate
            mlflow.log_metric("iter_loss", batch_loss.item(), step=global_step)
            mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=global_step)
            mlflow.log_metric("grad_norm", grad_total_norm, step=global_step)
            # Log dynamic metrics (Dice, BCE, etc.)
            for k, v in avg_metrics.items():
                mlflow.log_metric(f"iter_{k}", v, step=global_step)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
