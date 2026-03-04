import argparse
from os.path import join

import mlflow
import numpy as np
import torch
import lvis
from pycocotools import mask as mask_util
from torch.utils.data import DataLoader
from tqdm import tqdm

import opts
from datasets_fsis import build_fsis_dataset
from models.sansa.sansa import build_sansa
from util.commons import make_deterministic, setup_logging, resume_from_checkpoint
from util.promptable_utils import build_prompt_dict_fsis

mlflow.set_tracking_uri("http://127.0.0.1:5000")


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

    mIoU = eval_instance(model, args)
    return mIoU


def eval_instance(model: torch.nn.Module, args) -> float:
    print(f'Evaluating LVIS Instance mAP')

    # 1. Setup Dataset
    ds = build_fsis_dataset(args.dataset_file, image_set='val', args=args)
    dataloader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model.eval()
    results = []  # To store LVIS-style detections

    for batch in tqdm(dataloader, desc='Running Inference'):
        image_batch = batch['image'].to(args.device)
        instances_batch = batch['instances']
        cat_id = batch['category_id']  # From our revised Dataset class
        img_id = batch['img_id']  # Ensure your dataset returns the LVIS image_id

        with torch.no_grad():
            # Run the recursive SANSA loop
            # max_iterations should be high enough to catch most objects (e.g., 100 for LVIS)
            prompt_dict = build_prompt_dict_fsis(
                masks=instances_batch,
                prompt_type=args.prompt,
                num_support_prompts=args.shots,
                train_mode=False,
                device=model.device,
            )
            max_instances = max(len(instances) for instances in instances_batch)
            outputs = model(
                image_batch,
                prompt_dict,
                max_iterations=int(1.5 * max_instances),
                # Let the model run at most 1.5 times the amount of actual instances
            )

        preds = outputs[0]["masks"]  # [N, H, W] logits
        scores = outputs[0]["scores"]  # [N] confidence values

        # 2. Convert Predictions to LVIS Format
        for i in range(len(preds)):
            mask = (preds[i].sigmoid() > 0.5).cpu().numpy().astype(np.uint8)

            # Convert binary mask to COCO/LVIS RLE format
            rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")

            results.append({
                "image_id": int(img_id),
                "category_id": int(cat_id),
                "segmentation": rle,
                "score": float(scores[i])
            })

    # 3. Run LVIS Evaluation
    if len(results) == 0:
        print("No predictions made.")
        return 0.0

    # Load the ground truth annotations
    lvis_gt = lvis.LVIS(args.anno_path)
    lvis_dt = lvis.LVISResults(
        lvis_gt,
        results,
    )

    lvis_eval = lvis.LVISEval(lvis_gt, lvis_dt, iou_type='segm')
    lvis_eval.evaluate()
    lvis_eval.accumulate()
    lvis_eval.summarize()

    # Return the main mAP metric (AP @ IoU=0.5:0.95)
    return lvis_eval.stats['AP']


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SANSA evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    args.output_dir = join(args.output_dir, args.name_exp)
    main(args)
