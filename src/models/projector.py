import torch
import torch.nn as nn
import re
import math

from transformers.models.clip.modeling_clip import CLIPVisionModel


class PoolerProjector(nn.Module):
    def __init__(self, d_v, d_l, vision_cfg):
        super().__init__()
        self.hw = vision_cfg.image_size // vision_cfg.patch_size

        self.conv_pool = nn.Conv2d(d_v, d_l, kernel_size=2, stride=2)

        self.proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_l, d_l),
        )

    def forward(self, x, *args, **kwargs):
        height = width = self.hw
        assert height * width == x.shape[1]
        x = x.view(x.shape[0], height, width, -1).permute(0, 3, 1, 2)
        x = self.conv_pool(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

    @property
    def config(self):
        return {"mm_projector_type": "pooler"}


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(d_v, d_l, projector_type="linear", vision_cfg=None):
    """
    Build a vision projector to project vision features to language space.
    
    Args:
        d_v: Dimension of vision features (vision encoder hidden size)
        d_l: Dimension of language features (language model hidden size)
        projector_type: Type of projector to use
        vision_cfg: Configuration for vision model (needed for pooler projector)
        
    Returns:
        A module that projects vision features to language space
    """
    if projector_type == "linear":
        return nn.Linear(d_v, d_l)

    if projector_type == "pooler":
        if vision_cfg is None:
            raise ValueError("vision_cfg is required for pooler projector")
        return PoolerProjector(d_v, d_l, vision_cfg)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(d_v, d_l)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(d_l, d_l))
        return nn.Sequential(*modules)

    mlp_gelu_resnet_match = re.match(r"^mlp(\d+)x_res(\d+)x_gelu$", projector_type)
    if mlp_gelu_resnet_match:
        mlp_depth = int(mlp_gelu_resnet_match.group(1))
        res_depth = int(mlp_gelu_resnet_match.group(2))
        modules = [nn.Linear(d_v, d_l)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(d_l, d_l))
        for _ in range(res_depth):
            modules.append(SimpleResBlock(d_l))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")