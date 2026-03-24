import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets_fsis import build_fsis_dataset
from util.promptable_utils import build_prompt_dict_fsis


def eval_cellpose(model: torch.nn.Module, args) -> dict:
    ds = build_fsis_dataset(args.dataset_file, image_set='val', args=args)
    dataloader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
    model.eval()

    mse = 0.0
    n = 0
    with torch.no_grad():
        for batch in dataloader:
            image = batch['image'].to(args.device)
            cellpose_target = batch.get('cellpose_target', None)
            if cellpose_target is None:
                continue
            cellpose_target = cellpose_target.to(args.device)

            instances_batch = batch['instances']
            prompt_dict = build_prompt_dict_fsis(
                instances_batch,
                args.prompt,
                num_support_prompts=args.shots,
                train_mode=False,
                device=args.device,
            )

            max_instances = max(len(instances) for instances in instances_batch)
            outputs = model.inference(image, prompt_dict, max_iterations=int(1.5 * max_instances))

            pred_flow = outputs[0]['flows'][-1]  # last iteration
            if pred_flow.shape != cellpose_target.shape:
                pred_flow = torch.nn.functional.interpolate(pred_flow.unsqueeze(0), size=cellpose_target.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)

            mse += torch.mean((pred_flow - cellpose_target.squeeze(0)) ** 2).item()
            n += 1

    return {'eval_mse': mse / max(1, n)}
