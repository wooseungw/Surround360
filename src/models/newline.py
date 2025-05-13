import math
import torch
import torch.nn as nn
from typing import Callable, Optional, Union


def add_token_per_grid(image_feature: torch.Tensor, image_newline: nn.Parameter) -> torch.Tensor:
    """
    Adds a newline token for each grid position in the image feature.
    
    Args:
        image_feature: Tensor of shape (num_frames, num_patches, hidden_size)
        image_newline: Newline token embedding
        
    Returns:
        Tensor with newline tokens added at grid positions
    """
    num_frames = image_feature.shape[0]
    resize_h = int(math.sqrt(image_feature.shape[1]))
    feature_dim = image_feature.shape[-1]
    
    # Reshape to (num_frames, 1, resize_h, resize_h, hidden_size)
    image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
    
    # Permute to (hidden_size, num_frames, resize_h, 1, resize_h)
    image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
    
    # Flatten to (hidden_size, num_frames*resize_h, resize_h)
    image_feature = image_feature.flatten(1, 2).flatten(2, 3)
    
    # Add newline token to each grid line
    image_feature = torch.cat(
        (image_feature, image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), 
        dim=-1
    )
    
    # Flatten and transpose to (num_frames*resize_h*(resize_h+1), hidden_size)
    image_feature = image_feature.flatten(1, 2).transpose(0, 1)
    
    return image_feature


def add_token_per_frame(image_feature: torch.Tensor, image_newline: nn.Parameter) -> torch.Tensor:
    """
    Adds a newline token at the end of each frame in the image feature.
    
    Args:
        image_feature: Tensor of shape (num_frames, num_patches, hidden_size)
        image_newline: Newline token embedding
        
    Returns:
        Tensor with newline tokens added at the end of each frame
    """
    # Permute to (hidden_size, num_frames, num_patches)
    image_feature = image_feature.permute(2, 0, 1).contiguous()
    
    # Add newline token to each frame
    image_feature = torch.cat(
        (image_feature, image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)),
        dim=-1
    )
    
    # Permute back to (num_frames, num_patches+1, hidden_size)
    image_feature = image_feature.permute(1, 2, 0).contiguous()
    
    return image_feature


def add_one_token(image_feature: torch.Tensor, image_newline: nn.Parameter) -> torch.Tensor:
    """
    Adds a single newline token at the end of the entire flattened image feature.
    
    Args:
        image_feature: Tensor of shape (num_frames, num_patches, hidden_size)
        image_newline: Newline token embedding
        
    Returns:
        Tensor with one newline token added at the end
    """
    # Flatten the image feature to (num_frames*num_patches, hidden_size)
    image_feature = image_feature.flatten(0, 1)
    
    # Add a single newline token at the end
    image_feature = torch.cat(
        (image_feature, image_newline[None].to(image_feature.device)),
        dim=0
    )
    
    return image_feature


def no_token(image_feature: torch.Tensor, image_newline: Optional[nn.Parameter] = None) -> torch.Tensor:
    """
    Does not add any newline token, just flattens the image feature.
    
    Args:
        image_feature: Tensor of shape (num_frames, num_patches, hidden_size)
        image_newline: Unused parameter
        
    Returns:
        Flattened tensor without any newline tokens
    """
    return image_feature.flatten(0, 1)


def build_newline_inserter(config) -> Callable:
    """
    Builds and returns the appropriate newline token insertion function based on configuration.
    
    Args:
        config: Configuration object with mm_newline_position attribute
        
    Returns:
        Function that inserts newline tokens according to the specified strategy
    """
    mm_newline_position = getattr(config, "mm_newline_position", "one_token")
    
    if mm_newline_position == "grid":
        return add_token_per_grid
    elif mm_newline_position == "frame":
        return add_token_per_frame
    elif mm_newline_position == "one_token":
        return add_one_token
    elif mm_newline_position == "no_token":
        return no_token
    else:
        raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")


# 사용 예시
"""
# Example usage:
config.mm_newline_position = "grid"  # or "frame", "one_token", "no_token"
newline_inserter = build_newline_inserter(config)
processed_features = newline_inserter(image_features, model.image_newline)
"""


class NewlineTokenInserter:
    """
    클래스 기반 구현 - 원하는 경우 이 클래스를 사용할 수도 있습니다.
    """
    def __init__(self, config):
        self.mm_newline_position = getattr(config, "mm_newline_position", "one_token")
    
    def __call__(self, image_feature: torch.Tensor, image_newline: nn.Parameter) -> torch.Tensor:
        if self.mm_newline_position == "grid":
            return add_token_per_grid(image_feature, image_newline)
        elif self.mm_newline_position == "frame":
            return add_token_per_frame(image_feature, image_newline)
        elif self.mm_newline_position == "one_token":
            return add_one_token(image_feature, image_newline)
        elif self.mm_newline_position == "no_token":
            return no_token(image_feature)
        else:
            raise ValueError(f"Unexpected mm_newline_position: {self.mm_newline_position}")