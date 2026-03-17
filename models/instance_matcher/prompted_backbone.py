import torch
from torch import nn


class PromptedBackbone(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            images,
            prompts,
    ):
        pass
