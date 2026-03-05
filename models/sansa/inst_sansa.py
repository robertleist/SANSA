import os
from typing import Any, Dict, List, Tuple

import py3_wget
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
import torch.nn.functional as F

from models.sam2.modeling.sam2_utils import preprocess
from models.sam2.modeling.sam2_base import SAM2Base
from models.sansa.model_utils import BackboneOutput, DecoderOutput
from util.path_utils import SAM2_PATHS_CONFIG, SAM2_WEIGHTS_URL
from util.promptable_utils import rescale_prompt


class InstanceSANSA(nn.Module):
    def __init__(self, sam: SAM2Base, device: torch.device):
        super().__init__()
        self.sam = sam
        self.device = device

    def _preprocess_visual_features(
            self, samples: torch.Tensor, image_size: int
    ) -> Tuple[torch.Tensor, int, int, List[Tuple[int, int]]]:
        """
        Flatten [B,T,C,H,W] -> [B*T,C,H,W], store original sizes, and apply SAM2 preprocess.

        Args:
            samples:   Tensor [B, T, C, H, W].
            image_size: target side for SAM2 preprocessing.

        Returns:
            (samples_bt, B, T, orig_sizes)
        """

        B, T, C, H, W = samples.shape
        samples = samples.view(B * T, C, H, W)
        orig_size = [tuple(x.shape[-2:]) for x in samples]
        samples = torch.stack([preprocess(x, image_size) for x in samples], dim=0)
        return samples, B, T, orig_size

    def _compute_decoder_out_no_mem(
            self,
            backbone_out: BackboneOutput,
            idx: int,
            prompt_input: Dict[str, torch.Tensor] | None,
    ) -> DecoderOutput:
        """
        Decode a frame without memory: used for reference frames;

        Args:
            backbone_out: backbone features.
            idx: absolute idx.
            prompt:       "mask" | "point" | "scribble" | "box".
            prompt_input: inputs for point/scribble/box (ignored for "mask").

        Returns:
            DecoderOutput.
        """
        current_vision_feats = backbone_out.get_current_feats(idx)

        high_res_features = backbone_out.get_high_res_features(current_vision_feats)

        pix_feat_no_mem = current_vision_feats[-1:][-1] + self.sam.no_mem_embed
        pix_feat_no_mem = pix_feat_no_mem.permute(1, 2, 0).view(1, 256, 64, 64)
        decoder_out: DecoderOutput = self.sam._forward_sam_heads(
            backbone_features=pix_feat_no_mem,
            point_inputs=prompt_input,
            high_res_features=high_res_features,
        )
        return decoder_out

    def _compute_decoder_out_w_mem(
            self,
            backbone_out: BackboneOutput,
            idx: int,
            memory_idx: int,
            memory_bank: Dict[int, Dict[str, torch.Tensor]],
    ) -> DecoderOutput:
        """
        Decode a frame with memory: used for target frames;

        Args:
            backbone_out: backbone features.
            idx: absolute idx.
            memory_idx:   temporal index t (0-based).
            memory_bank:  dict of memory entries from previous frames.

        Returns:
            DecoderOutput
        """
        current_vision_feats = backbone_out.get_current_feats(idx)
        current_vision_pos_embeds = backbone_out.get_current_pos_embeds(idx)

        # take only the highest res feature map
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
        return decoder_out

    def _compute_memory_bank_dict(
            self, decoder_out: DecoderOutput, backbone_out: BackboneOutput, idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Encode current prediction into memory for later frames.

        Args:
            decoder_out: decoder output with high_res/low_res masks.
            backbone_out:  backbone features.
            idx: absolute idx.

        Returns:
            Memory entry dict.
        """
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

    def _forward_backbone(
            self, samples: torch.Tensor, orig_size: List[Tuple[int, int]]
    ) -> BackboneOutput:
        """
        Run SAM2 image encoder and prepare backbone features for decoding.

        Args:
            samples:  Tensor [B*T, C, H, W] after preprocessing.
            orig_size:   list of original frame sizes.

        Returns:
            BackboneOutput.
        """
        vis = self.sam.image_encoder.trunk(samples)
        feats, pos = self.sam.image_encoder.neck(vis)

        # discard lowest resolution
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

    # Conceptual change for Instance Segmentation
    def forward(
            self,
            batch: torch.Tensor,
            prompt_batch: List[List[Dict[str, Any]]],
            max_iterations: int = None,
            stopping_threshold: float = 0.5,
    ):
        """
        Find instances in image.
        Args:
            batch: Tensor [B, C, H, W] after preprocessing.
            prompt_batch: List (B) of List of prompts per image. Usually [B, 1]
            max_iterations: Control the number of iterations.
            stopping_threshold: If the predicted objectness score is below this threshold the iteration cycle ends.
        """
        # Bring image into [B, T, C, H, W] shape. B = Batches, T = Tensors (Support and Query images originally)
        # Preprocess batches
        batch_reshaped = batch.unsqueeze(1)
        batch, B, T, orig_size = self._preprocess_visual_features(batch_reshaped, self.sam.image_size)
        backbone_output = self._forward_backbone(batch, orig_size)

        # Process sequentially -> Cannot be parallelized
        batch_output = []
        for b in range(B):
            # For each image, reset the memory bank
            self.memory_bank = {}
            outputs = {
                "masks": [],
                "scores": [],
            }
            i = 0
            while max_iterations is None or not i > max_iterations:
                # During inference, set max_iterations to None to make the model run until it predicts no more objects.
                image_prompts = prompt_batch[b]
                if i < len(image_prompts):
                    # Iterate over the prompts
                    frame_prompt_dict = image_prompts[i]
                    frame_prompt = frame_prompt_dict['prompt']
                    prompt_type = frame_prompt_dict['prompt_type']
                    frame_prompt = rescale_prompt(frame_prompt, prompt_type, orig_size[0], self.sam.image_size)
                    if prompt_type == 'mask':
                        decoder_out: DecoderOutput = self.sam._use_mask_as_output(
                            backbone_output,
                            frame_prompt,
                            b
                        )
                    else:
                        decoder_out: DecoderOutput = self._compute_decoder_out_no_mem(
                            backbone_output,
                            b,
                            prompt_input=frame_prompt
                        )
                else:
                    # Subsequent instances: Use memory to avoid previous objects
                    decoder_out = self._compute_decoder_out_w_mem(
                        backbone_output,
                        idx=0,
                        memory_idx=i,
                        memory_bank=self.memory_bank
                    )
                score = torch.sigmoid(decoder_out.object_score_logits)
                # Stop if the 'objectness' score is too low
                if score < stopping_threshold:
                    break

                # Update memory so the NEXT iteration knows what is already segmented
                mem_entry = self._compute_memory_bank_dict(decoder_out, backbone_output, idx=0)
                self.memory_bank[i] = mem_entry

                outputs["masks"].append(decoder_out.masks[0])
                outputs["scores"].append(score)
                i += 1
            batch_output.append(
                {
                    "masks": torch.stack(outputs["masks"], 1).squeeze(),
                    "scores": torch.stack(outputs["scores"], 1).squeeze(),
                }
            )
        return batch_output


def build_inst_sansa(
        sam2_version: str = 'large',
        adaptformer_stages: List[int] = [2, 3],
        channel_factor: float = 0.3,
        device: str = 'cuda'
) -> InstanceSANSA:
    assert sam2_version in SAM2_PATHS_CONFIG.keys(), f'wrong argument sam2_version: {sam2_version}'

    sam2_weights, sam2_config = SAM2_PATHS_CONFIG[sam2_version]
    if not os.path.isfile(sam2_weights):
        print(f"Downloading SAM2-{sam2_version}")
        py3_wget.download_file(SAM2_WEIGHTS_URL[sam2_version], sam2_weights)

    with initialize(version_base=None, config_path=".", job_name="test_app"):
        cfg = compose(config_name=sam2_config, overrides=[
            f"++model.image_encoder.trunk.adaptformer_stages={adaptformer_stages}",
            f"++model.image_encoder.trunk.adapt_dim={channel_factor}",
        ])

        OmegaConf.resolve(cfg)
        cfg.model.pred_obj_scores = False
        cfg.model.pred_obj_scores_mlp = False
        cfg.model.fixed_no_obj_ptr = False
        sam = instantiate(cfg.model, _recursive_=True)

    state_dict = torch.load(sam2_weights, map_location="cpu", weights_only=False)["model"]
    sam.load_state_dict(state_dict, strict=False)
    model = InstanceSANSA(sam=sam, device=torch.device(device))

    # freeze everything except adapters
    for name, p in model.named_parameters():
        p.requires_grad = ("adapter" in name)

    return model
