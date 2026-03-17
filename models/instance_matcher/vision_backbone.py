from dataclasses import dataclass

import torch
from torch import nn
from enum import Enum

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from transformers import AutoImageProcessor, DINOv3ViTModel, DINOv3ViTConfig, DINOv3ViTImageProcessorFast


@dataclass
class VisionEmbedding:
    n_batches: int
    h_patches: int
    w_patches: int
    n_features: int
    embedding: torch.Tensor


class VisionBackbone(nn.Module):
    def __init__(self):
        super(VisionBackbone, self).__init__()

    def forward(self, x) -> VisionEmbedding:
        raise NotImplementedError


class DinoModelType(Enum):
    VITS16 = "dinov3_vits16"
    VITS16PLUS = "dinov3_vits16plus"
    VITB16 = "dinov3_vitb16"
    VITL16 = "dinov3_vitl16"
    VITH16PLUS = "dinov3_vith16plus"
    VIT7B16 = "dinov3_vit7b16"


MODEL_TO_WEIGHTS = {
    DinoModelType.VITS16: None,
    DinoModelType.VITS16PLUS: "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    DinoModelType.VITB16: None,
    DinoModelType.VITL16: None,
    DinoModelType.VITH16PLUS: None,
    DinoModelType.VIT7B16: None,
}
MODEL_TO_HF_URL = {
    DinoModelType.VITS16: "facebook/dinov3-vits16-pretrain-lvd1689m",
    DinoModelType.VITS16PLUS: "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    DinoModelType.VITB16: "facebook/dinov3-vitb16-pretrain-lvd1689m",
    DinoModelType.VITL16: "facebook/dinov3-vitl16-pretrain-lvd1689m",
    DinoModelType.VITH16PLUS: "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    DinoModelType.VIT7B16: "facebook/dinov3-vit7b16-pretrain-lvd1689m",
}
MODEL_TO_NUM_LAYERS = {
    DinoModelType.VITS16: 12,
    DinoModelType.VITS16PLUS: 12,
    DinoModelType.VITB16: 12,
    DinoModelType.VITL16: 24,
    DinoModelType.VITH16PLUS: 32,
    DinoModelType.VIT7B16: 40,
}


class DINOv3Backbone(VisionBackbone):
    def __init__(self,
                 model_type: DinoModelType,
                 image_size: int = None,
                 patch_size: int = None,
                 device: str = "cpu"):
        super().__init__()
        self.model_type = model_type
        hf_url = MODEL_TO_HF_URL[model_type]
        self.device = device

        self.processor: DINOv3ViTImageProcessorFast = AutoImageProcessor.from_pretrained(
            hf_url,
            device=self.device,
        )
        self.config = DINOv3ViTConfig.from_pretrained(
            hf_url,
            device=self.device,
        )
        if image_size is not None: self.config.image_size = image_size
        if patch_size is not None: self.config.patch_size = patch_size
        self.model = DINOv3ViTModel(
            config=self.config,
        )
        self.model = self.model.to(self.device).eval()
        print(self.config)

    @property
    def embedding_dim(self) -> int:
        return self.config.hidden_size

    @torch.no_grad()
    def preprocess(self, image: Image.Image):
        w = image.width
        h = image.height
        h_patches = int(self.config.image_size / self.config.patch_size)
        w_patches = int((w * self.config.image_size) / (h * self.config.patch_size))
        tensor = self.processor(
            images=image,
            size=self.config.image_size,
            return_tensors="pt")
        return tensor, h_patches, w_patches

    @torch.no_grad()
    def embed_preprocessed(self, input) -> torch.Tensor:
        feats = self.model(**input)
        x = feats[-1].squeeze().detach().cpu()
        dim = x.shape[0]
        return x.view(dim, -1).permute(1, 0)

    @torch.no_grad()
    def embed_image(
            self,
            image: Image.Image,
            standardize: bool = False,
            keep_dim: bool = True
    ) -> torch.Tensor:
        # Save the original size
        og_h, og_w = image.height, image.width

        # Preprocess the image
        inputs, h_patches, w_patches = self.preprocess(image)

        # Compute the embedding
        outputs = self.model.forward(**inputs, output_hidden_states=False)

        # We only need the last hidden state
        last_hidden_state = outputs.last_hidden_state.squeeze()
        cls_token, reg_token, embeddings = last_hidden_state[0], last_hidden_state[1:5], last_hidden_state[5:]

        if standardize:
            # Standardize each feature dimension to zero mean and unit variance
            mean = embeddings.mean(dim=0, keepdim=True)
            std = embeddings.std(dim=0, keepdim=True)
            embeddings = (embeddings - mean) / (std + 1e-8)  # Add small epsilon to avoid division by zero

        embeddings = embeddings.reshape(h_patches, w_patches, -1)
        # Resize to original size if enabled
        if keep_dim:
            embeddings = TF.resize(
                embeddings.permute(2, 0, 1),
                [og_h, og_w]
            ).permute(1, 2, 0)
        return embeddings
