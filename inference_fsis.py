import argparse
from os.path import join

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

import opts
from datasets_fsis import build_fsis_dataset
from models.sansa.sansa import build_sansa
from util.commons import make_deterministic, setup_logging, resume_from_checkpoint
from util.path_utils import MLFLOW_URL
from util.promptable_utils import build_prompt_dict_fsis

mlflow.set_tracking_uri(MLFLOW_URL)


def main(args: argparse.Namespace) -> float:
    setup_logging(args.output_dir, console="info", rank=0)
    mlflow.set_experiment("FSIS")
    mlflow.start_run(run_name=f"SANSA - {args.dataset_file} (eval)")
    mlflow.log_params(args.__dict__)

    make_deterministic(args.seed)
    print(args)

    model = build_sansa(args.sam2_version, args.adaptformer_stages, args.channel_factor, args.device)
    device = torch.device(args.device)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.resume:
        resume_from_checkpoint(args.resume, model)

    print(f"number of params: {n_parameters}")
    print('Start inference')

    mAP = eval_instance(model, args)
    return mAP


def eval_instance(model: torch.nn.Module, args) -> dict:
    print(f'Evaluating mAP on {args.device}...')

    # 1. Setup Dataset & Metric
    ds = build_fsis_dataset(args.dataset_file, image_set='val', args=args)
    dataloader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # Initialize TorchMetrics mAP (iou_type='segm' equivalent)
    # We set 'masks=True' to compute Average Precision for segmentations
    metric = MeanAveragePrecision(
        iou_type="segm",
        #backend="faster_coco_eval"
    ).to(args.device)

    model.eval()

    for batch in tqdm(dataloader, desc='Running Inference'):
        image_batch = batch['image'].to(args.device)
        instances_batch = [inst.to(args.device) for inst in batch['instances']]  # GT Masks
        cat_ids = batch['category_id'].to(args.device)  # GT Category

        with torch.no_grad():
            prompt_dict = build_prompt_dict_fsis(
                masks=instances_batch,
                prompt_type=args.prompt,
                num_support_prompts=args.shots,
                train_mode=False,
                device=args.device,
            )

            max_instances = max(len(instances) for instances in instances_batch)
            outputs = model.inference(
                image_batch,
                prompt_dict,
                max_iterations=int(1.5 * max_instances),
            )

        # 2. Prepare Predictions for Metric
        # Logic: Convert logits to binary masks [N, H, W]
        pred_masks = (outputs[0]["masks"].sigmoid() > 0.5).bool()
        pred_scores = outputs[0]["scores"]
        # In LVIS eval, every pred for a sample usually shares the image's category_id
        # based on your previous script's logic
        pred_labels = torch.full((pred_masks.shape[0],), int(cat_ids[0]), device=args.device)

        preds = [
            dict(
                masks=pred_masks.squeeze(),
                scores=pred_scores.squeeze(),
                labels=pred_labels,
            )
        ]

        # 3. Prepare Ground Truth for Metric
        # Flatten the list of instances from the batch
        gt_masks = torch.stack(instances_batch, 0).squeeze()
        gt_labels = torch.full((gt_masks.shape[0],), int(cat_ids[0]), device=args.device)

        target = [
            dict(
                masks=gt_masks.bool(),
                labels=gt_labels,
            )
        ]

        # 4. Update Metric on GPU
        metric.update(preds, target)

    # 5. Compute Final Results
    results = metric.compute()

    print(f"\nEvaluation Results:")
    print(f"mAP: {results['map']:.4f}")
    print(f"mAP_50: {results['map_50']:.4f}")
    print(f"mAP_75: {results['map_75']:.4f}")

    results_dict = {}  # Transfer to loggable dict
    for k, v in results.items():
        if isinstance(v, torch.Tensor):
            try:
                results_dict[f"eval_{k}"] = v.item()
            except Exception as e:
                print(f"Cant log {k} with value {v}")
                continue
        else:
            if isinstance(v, list) or isinstance(v, tuple) or isinstance(v, np.ndarray):
                if len(v) > 1:
                    print(f"Cant log {k} with value {v}")
                    continue
            results_dict[f"eval_{k}"] = v
    return results_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SANSA evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    args.output_dir = join(args.output_dir, args.name_exp)
    main(args)
