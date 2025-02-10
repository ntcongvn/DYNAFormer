# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MaskDINO https://github.com/IDEA-Research/MaskDINO by Tan-Cong Nguyen
# ------------------------------------------------------------------------

import torch
import copy
from torch import device, nn, Tensor
import os
import math
import torch.nn.functional as F
from . import box_ops


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

def get_bounding_boxes(masks):
    """
    Get bounding boxes for binary masks.
    
    Args:
        masks (torch.Tensor): A tensor of shape (N, O, H, W) representing the masks.
    
    Returns:
        torch.Tensor: A tensor of shape (N, O, 4), where each bounding box is (cx, cy, w, h)
                      with values normalized between 0 and 1.
    """
    N, O, H, W = masks.shape
    
    # Find non-zero pixels in the masks (y and x coordinates)
    y_coords = torch.arange(H, device=masks.device).view(1, 1, H, 1).expand(N, O, H, W)
    x_coords = torch.arange(W, device=masks.device).view(1, 1, 1, W).expand(N, O, H, W)
    
    # Find the min/max coordinates for the bounding box
    x_min = torch.where(masks, x_coords.float(), torch.tensor(W, device=masks.device).float()).view(N, O, -1).min(dim=2)[0]
    x_max = torch.where(masks, x_coords.float(), torch.tensor(0, device=masks.device).float()).view(N, O, -1).max(dim=2)[0]
    y_min = torch.where(masks, y_coords.float(), torch.tensor(H, device=masks.device).float()).view(N, O, -1).min(dim=2)[0]
    y_max = torch.where(masks, y_coords.float(), torch.tensor(0, device=masks.device).float()).view(N, O, -1).max(dim=2)[0]
    
    # Calculate center coordinates, width, and height, normalized to [0, 1]
    cx = (x_min + x_max) / 2 / W
    cy = (y_min + y_max) / 2 / H
    w = (x_max - x_min + 1) / W  # Add 1 to include the last pixel in width
    h = (y_max - y_min + 1) / H  # Add 1 to include the last pixel in height
    
    # Stack the bounding box coordinates into a single tensor (N, O, 4)
    bounding_boxes = torch.stack([cx, cy, w, h], dim=2)
    
    return bounding_boxes

def get_bounding_boxes_ohw(masks):
    """
    Get bounding boxes for binary masks.
    
    Args:
        masks (torch.Tensor): A tensor of shape (O, H, W) representing the masks.
    
    Returns:
        torch.Tensor: A tensor of shape (O, 4), where each bounding box is (cx, cy, w, h)
                      with values normalized between 0 and 1.
    """
    O, H, W = masks.shape
    
    # Find non-zero pixels in the masks (y and x coordinates)
    y_coords = torch.arange(H, device=masks.device).view(1, H, 1).expand(O, H, W)
    x_coords = torch.arange(W, device=masks.device).view(1, 1, W).expand(O, H, W)
    
    # Find the min/max coordinates for the bounding box
    x_min = torch.where(masks, x_coords.float(), torch.tensor(W, device=masks.device).float()).view(O, -1).min(dim=1)[0]
    x_max = torch.where(masks, x_coords.float(), torch.tensor(0, device=masks.device).float()).view(O, -1).max(dim=1)[0]
    y_min = torch.where(masks, y_coords.float(), torch.tensor(H, device=masks.device).float()).view(O, -1).min(dim=1)[0]
    y_max = torch.where(masks, y_coords.float(), torch.tensor(0, device=masks.device).float()).view(O, -1).max(dim=1)[0]
    
    # Calculate center coordinates, width, and height, normalized to [0, 1]
    cx = (x_min + x_max) / 2 / W
    cy = (y_min + y_max) / 2 / H
    w = (x_max - x_min + 1) / W  # Add 1 to include the last pixel in width
    h = (y_max - y_min + 1) / H  # Add 1 to include the last pixel in height
    
    # Stack the bounding box coordinates into a single tensor (O, 4)
    bounding_boxes = torch.stack([cx, cy, w, h], dim=1)
    
    return bounding_boxes

#Sineembed for x, y
def sineembed_for_position_xy(pos_tensor,dim=256):
    half_dim= dim/2
    scale = 2 * math.pi
    dim_t = torch.arange(half_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_dim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)
    return pos

def apply_random_mask_noise_transforms(masks, bboxes,noise_scale,new_size):
    dn_config={

      
    }
    device=masks.device
    noise_scale = torch.tensor(noise_scale).to(device)
    N, H, W = masks.shape
    # Convert bounding box coordinates to pixel units
    cx = (bboxes[:, 0] * W).unsqueeze(1)  # Center x in pixels
    cy = (bboxes[:, 1] * H).unsqueeze(1)  # Center y in pixels
    w = (bboxes[:, 2] * W).unsqueeze(1)    # Width in pixels
    h = (bboxes[:, 3] * H).unsqueeze(1)    # Height in pixels

    # Random translation offsets, ensuring masks stay within the bounds
    offset_x = (torch.rand(N).to(device) * (w.squeeze() / 2)*noise_scale)
    offset_y = (torch.rand(N).to(device) * (h.squeeze() / 2)*noise_scale)

    # Random rotation angles
    angles = (torch.rand(N).to(device) * 90 - 45)*noise_scale* (np.pi / 180)  # In radians
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    # Random scaling factors
    scale_factors = (1+torch.rand(N).to(device) * noise_scale - 0.5 * noise_scale).to(masks.device)  # Scale from 0.5 to 1.5
    
    # Create the affine transformation matrix
    affine_transforms = torch.zeros(N, 2, 3).to(masks.device)
    affine_transforms[:, 0, 0] = cos_angles * scale_factors  # Scaling along x
    affine_transforms[:, 0, 1] = -sin_angles * scale_factors  # Rotation along x
    affine_transforms[:, 1, 0] = sin_angles * scale_factors  # Rotation along y
    affine_transforms[:, 1, 1] = cos_angles * scale_factors  # Scaling along y

    # Offset should be carefully adjusted so the transformations remain visible
    affine_transforms[:, 0, 2] = (offset_x + (1 - scale_factors) * cx.squeeze()) / W  # Translate along x
    affine_transforms[:, 1, 2] = (offset_y + (1 - scale_factors) * cy.squeeze()) / H  # Translate along y

    # Reshape masks for grid sample (N, C, H, W)
    masks_4d = masks.unsqueeze(1)  # Add channel dimension

    # Create affine grid
    grid = F.affine_grid(affine_transforms, masks_4d.size(), align_corners=False)

    # Apply transformations
    mask_transformed = F.grid_sample(masks_4d.float(), grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    mask_transformed = F.interpolate(mask_transformed, size=(new_size[0], new_size[1]), mode='nearest')

    # Return the masks in original shape (N, H, W)
    return mask_transformed.squeeze().clamp(min=0.0, max=1.0)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def inverse_sigmoid_mask(x, eps=1e-11):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2).to(torch.float16)

def gen_encoder_output_proposals(memory:Tensor, memory_padding_mask:Tensor, spatial_shapes:Tensor):
    """
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
    N_, S_, C_ = memory.shape
    #base_scale = 4.0
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

        grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                        torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
        wh = torch.ones_like(grid) * 0.1 * (2.0 ** lvl)
        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += (H_ * W_)
    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))
    output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

    output_memory = memory
    output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
    return output_memory, output_proposals

def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N, layer_share=False):

    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def compute_mask_proposal(bboxes, W, H):
    """
    Compute masks for a batch of bounding boxes with the same output mask size for all.

    Args:
        bboxes: Tensor of shape (N, P, 4), where N is the batch size, P is the number of bounding boxes, 
                and 4 corresponds to the bounding box coordinates (x1, y1, x2, y2).
        W: The output mask width.
        H: The output mask height.

    Returns:
        Tensor of shape (N, P, H, W) containing the masks for each bounding box.
    """
    device=bboxes.device
    
    N, P, _ = bboxes.shape  # N: batch size, P: number of points

    # Create a meshgrid for the output mask size (W, H)
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(0, H - 1, H).to(device), 
        torch.linspace(0, W - 1, W).to(device), 
        indexing='ij'
    )
    y_grid = y_grid.unsqueeze(0).unsqueeze(0).to(device)  # Add batch and point dimensions
    x_grid = x_grid.unsqueeze(0).unsqueeze(0).to(device)

    # Calculate the center of the bounding boxes
    center_x = (bboxes[..., 0] + bboxes[..., 2]) / 2
    center_y = (bboxes[..., 1] + bboxes[..., 3]) / 2

    # Calculate half-width and half-height of bounding boxes
    half_width = (bboxes[..., 2] - bboxes[..., 0]) / 2
    half_height = (bboxes[..., 3] - bboxes[..., 1]) / 2

    # Normalize distances from the center of the bounding box

    dist_x = (x_grid - center_x.unsqueeze(-1).unsqueeze(-1)) / half_width.unsqueeze(-1).unsqueeze(-1)
    dist_y = (y_grid - center_y.unsqueeze(-1).unsqueeze(-1)) / half_height.unsqueeze(-1).unsqueeze(-1)

    # Compute the radial distance
    dist = torch.sqrt(dist_x ** 2 + dist_y ** 2)

    # Create the mask, clipping the distance to 1 and mapping to [1, 0.5]
    mask = torch.clamp(1 - dist / 2, 0.5, 1.0)

    # Set values outside the bounding box region to 0
    in_bbox_x = (x_grid >= bboxes[..., 0].unsqueeze(-1).unsqueeze(-1)) & \
                (x_grid <= bboxes[..., 2].unsqueeze(-1).unsqueeze(-1))
    in_bbox_y = (y_grid >= bboxes[..., 1].unsqueeze(-1).unsqueeze(-1)) & \
                (y_grid <= bboxes[..., 3].unsqueeze(-1).unsqueeze(-1))

    mask = mask * (in_bbox_x & in_bbox_y)

    # Return the final mask with shape (N, P, H, W)
    return mask