import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import py3_wget
import torch
import torch.nn.functional as F
from torch import nn
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torchvision.transforms.functional import resize

from models.sam2.modeling.sam2_base import SAM2Base
from models.sam2.modeling.sam2_utils import preprocess
from models.sansa.model_utils import BackboneOutput, DecoderOutput
from util.path_utils import SAM2_PATHS_CONFIG, SAM2_WEIGHTS_URL
from util.promptable_utils import rescale_prompt


class CellposeSANSA(nn.Module):
    def __init__(self, sam: SAM2Base, device: torch.device):
        super().__init__()
        self.sam = sam
        self.device = device
        self.memory_bank: dict[int, dict[str, torch.Tensor]] = {}

        self.flow_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, kernel_size=1)
        )

    def _preprocess_visual_features(
            self, samples: torch.Tensor, image_size: int
    ) -> Tuple[torch.Tensor, int, int, List[Tuple[int, int]]]:
        B, T, C, H, W = samples.shape
        samples = samples.view(B * T, C, H, W)
        orig_size = [tuple(x.shape[-2:]) for x in samples]
        samples = torch.stack([preprocess(x, image_size) for x in samples], dim=0)
        return samples, B, T, orig_size

    def _forward_backbone(
            self, samples: torch.Tensor, orig_size: List[Tuple[int, int]]
    ) -> BackboneOutput:
        vis = self.sam.image_encoder.trunk(samples)
        feats, pos = self.sam.image_encoder.neck(vis)

        feats, pos = feats[:-1], pos[:-1]
        feats[0] = self.sam.sam_mask_decoder.conv_s0(feats[0])
        feats[1] = self.sam.sam_mask_decoder.conv_s1(feats[1])

        bb = {
            "vision_features": feats[-1],
            "vision_pos_enc": pos,
            "backbone_fpn": feats,
        }
        vision_feats, vision_pos, sizes = self.sam._prepare_backbone_features(bb)
        return BackboneOutput(orig_size, vision_feats, vision_pos, sizes)

    def _compute_decoder_out_no_mem(
            self,
            backbone_out: BackboneOutput,
            idx: int,
            prompt_input: Dict[str, torch.Tensor] | None,
    ) -> DecoderOutput:
        current_vision_feats = backbone_out.get_current_feats(idx)
        high_res_features = backbone_out.get_high_res_features(current_vision_feats)

        pix_feat_no_mem = current_vision_feats[-1:][-1] + self.sam.no_mem_embed
        pix_feat_no_mem = pix_feat_no_mem.permute(1, 2, 0).view(1, 256, 64, 64)

        decoder_out: DecoderOutput = self.sam._forward_sam_heads(
            backbone_features=pix_feat_no_mem,
            point_inputs=prompt_input,
            high_res_features=high_res_features,
        )

        decoder_out.flow = self.flow_head(pix_feat_no_mem)
        return decoder_out

    def _compute_decoder_out_w_mem(
            self,
            backbone_out: BackboneOutput,
            idx: int,
            memory_idx: int,
            memory_bank: Dict[int, Dict[str, torch.Tensor]],
    ) -> DecoderOutput:
        current_vision_feats = backbone_out.get_current_feats(idx)
        current_vision_pos_embeds = backbone_out.get_current_pos_embeds(idx)

        high_res_features = backbone_out.get_high_res_features(current_vision_feats)

        pix_feat_with_mem = self.sam._prepare_memory_conditioned_features(
            frame_idx=memory_idx,
            current_vision_feats=current_vision_feats[-1:],
            current_vision_pos_embeds=current_vision_pos_embeds[-1:],
            feat_sizes=backbone_out.feat_sizes[-1:],
            num_frames=memory_idx + 1,
            memory_bank=memory_bank
        )

        decoder_out: DecoderOutput = self.sam._forward_sam_heads(
            backbone_features=pix_feat_with_mem,
            high_res_features=high_res_features,
            multimask_output=True if memory_idx > 0 else False
        )

        decoder_out.flow = self.flow_head(pix_feat_with_mem)
        return decoder_out

    def _compute_memory_bank_dict(
            self, decoder_out: DecoderOutput, backbone_out: BackboneOutput, idx: int
    ) -> Dict[str, torch.Tensor]:
        current_vision_feats = backbone_out.get_current_feats(idx)
        feat_sizes = backbone_out.feat_sizes
        mem_feats, mem_pos = self.sam._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=decoder_out.high_res_masks,
            is_mask_from_pts=False,
        )
        return {
            "maskmem_features": mem_feats,
            "maskmem_pos_enc": mem_pos,
            "pred_masks": decoder_out.low_res_masks,
            "obj_ptr": decoder_out.obj_ptr,
        }

    def forward(
            self,
            image_batch: torch.Tensor,
            prompt_batch: List,
            memory_batch: dict[int, dict[int, dict[str, torch.Tensor]]],
            current_iteration: int,
    ):
        batch_reshaped = image_batch.unsqueeze(1)
        image_batch, B, T, orig_size = self._preprocess_visual_features(batch_reshaped, self.sam.image_size)
        assert T == 1
        backbone_output = self._forward_backbone(image_batch, orig_size)

        batch_output = []
        for b in range(B):
            outputs = {"masks": [], "scores": [], "flows": []}
            memory_bank = memory_batch[b]
            image_prompts = prompt_batch[b]

            if current_iteration < len(image_prompts):
                frame_prompt_dict = image_prompts[current_iteration]
                frame_prompt = frame_prompt_dict['prompt']
                prompt_type = frame_prompt_dict['prompt_type']
                frame_prompt = rescale_prompt(frame_prompt, prompt_type, orig_size[b], self.sam.image_size)

                if prompt_type == 'mask':
                    decoder_out: DecoderOutput = self.sam._use_mask_as_output(backbone_output, frame_prompt, b)
                    # no flow from prompt masks (set zero)
                    decoder_out.flow = torch.zeros((1, 3, 64, 64), device=image_batch.device)
                else:
                    decoder_out = self._compute_decoder_out_no_mem(backbone_output, b, prompt_input=frame_prompt)
            else:
                decoder_out = self._compute_decoder_out_w_mem(backbone_output, idx=b, memory_idx=current_iteration, memory_bank=memory_bank)

            score = torch.sigmoid(decoder_out.object_score_logits)

            mem_entry = self._compute_memory_bank_dict(decoder_out, backbone_output, idx=b)
            memory_batch[b][current_iteration] = mem_entry

            mask_out = decoder_out.masks[0]
            mask_out_resized = resize(mask_out, list(orig_size[b]))
            flow_out = resize(decoder_out.flow.squeeze(0), list(orig_size[b]))

            outputs["masks"].append(mask_out_resized)
            outputs["scores"].append(score)
            outputs["flows"].append(flow_out)

            batch_output.append({
                "masks": torch.stack(outputs["masks"], 1),
                "scores": torch.stack(outputs["scores"], 1),
                "flows": torch.stack(outputs["flows"], 1),
            })

        return batch_output, memory_batch

    def inference(
            self,
            batch: torch.Tensor,
            prompt_batch: List,
            max_iterations: int = None,
            stopping_threshold: float = 0.5,
    ):
        batch_outputs = defaultdict(lambda: defaultdict(list))
        memory_batch: dict[int, dict[int, dict[str, torch.Tensor]]] = defaultdict(lambda: defaultdict(dict))
        for current_iteration in range(max_iterations):
            iter_outputs, memory_batch = self.forward(batch, prompt_batch, memory_batch, current_iteration)
            for b in range(len(iter_outputs)):
                for k, v in iter_outputs[b].items():
                    batch_outputs[b][k].append(v)
        return batch_outputs


def build_cellpose_sansa(
        sam2_version: str = 'large',
        adaptformer_stages: List[int] = [2, 3],
        channel_factor: float = 0.3,
        device: str = 'cuda'
) -> CellposeSANSA:
    assert sam2_version in SAM2_PATHS_CONFIG.keys(), f'wrong argument sam2_version: {sam2_version}'

    sam2_weights, sam2_config = SAM2_PATHS_CONFIG[sam2_version]
    if not os.path.isfile(sam2_weights):
        print(f"Downloading SAM2-{sam2_version}")
        py3_wget.download_file(SAM2_WEIGHTS_URL[sam2_version], sam2_weights)

    with initialize(version_base=None, config_path='.', job_name='cellpose'):
        cfg = compose(config_name=sam2_config, overrides=[
            f"++model.image_encoder.trunk.adaptformer_stages={adaptformer_stages}",
            f"++model.image_encoder.trunk.adapt_dim={channel_factor}",
        ])

        OmegaConf.resolve(cfg)
        cfg.model.pred_obj_scores = False
        cfg.model.pred_obj_scores_mlp = False
        cfg.model.fixed_no_obj_ptr = False
        sam = instantiate(cfg.model, _recursive_=True)

    state_dict = torch.load(sam2_weights, map_location='cpu', weights_only=False)['model']
    sam.load_state_dict(state_dict, strict=False)
    model = CellposeSANSA(sam=sam, device=torch.device(device))

    for name, p in model.named_parameters():
        p.requires_grad = ('memory' in name.lower() or 'adapter' in name.lower())

    return model
