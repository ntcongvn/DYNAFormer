# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/fundamentalvision/Deformable-DETR
# ------------------------------------------------------------------------
# Modified from MaskDINO https://github.com/IDEA-Research/MaskDINO by Tan-Cong Nguyen
# ------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction
from ..functions.ms_deform_attn_func import ms_deform_attn_core_pytorch
from detectron2.structures import BitMasks
from dynaformer.utils import box_ops
from dynaformer.utils.utils import get_bounding_boxes


def check_points_in_mask(mask, points):
    # mask shape: (N, Q, H, W)
    # points shape: (N, Q, P, 2)
    
    N, Q, H, W = mask.shape
    _, _, P, _ = points.shape
    
    # Convert normalized points (x, y) to pixel coordinates
    pixel_coords = points.clone()
    pixel_coords[..., 0] = (points[..., 0] * W).long()  # x-coordinates
    pixel_coords[..., 1] = (points[..., 1] * H).long()  # y-coordinates
    
    # Ensure that pixel_coords are of dtype long
    pixel_coords = pixel_coords.long()
    
    # Clamp the values to ensure they are within valid pixel ranges
    pixel_coords[..., 0] = pixel_coords[..., 0].clamp(0, W - 1)
    pixel_coords[..., 1] = pixel_coords[..., 1].clamp(0, H - 1)
    
    # Create grid indices for batch and mask
    batch_indices = torch.arange(N, dtype=torch.long, device=mask.device).view(N, 1, 1)
    mask_indices = torch.arange(Q, dtype=torch.long, device=mask.device).view(1, Q, 1)
    
    # Gather the corresponding values from the mask at the given pixel coordinates
    mask_at_points = mask[
        batch_indices,           # Batch dimension
        mask_indices,            # Mask dimension
        pixel_coords[..., 1],    # y-coordinates (height)
        pixel_coords[..., 0]     # x-coordinates (width)
    ]
    
    # mask_at_points will have shape (N, Q, P), and values will be 1 or 0
    return mask_at_points


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttnMask(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=8,type_sampling_location="mask"):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttnMask to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 128

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.type_sampling_location = type_sampling_location

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * self.n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * self.n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

                    
    def forward(self, 
        query,                        #N*(D+Q)*C
        reference_bboxs,              #N*(D+Q)*4      unsigmoid
        reference_masks,              #N*(D+Q)*H*W    unsigmoid
        mask_threshold,
        input_flatten,                #N*Sum{WH}*C
        input_spatial_shapes,         #3*2
        input_level_start_index,      #Level
        input_padding_mask=None):     #N*Sum{WH}
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        

        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        #reference_masks_sig=reference_masks.sigmoid()
        if self.type_sampling_location == "both":
          #init sampling location 
          sampling_locations = sampling_offsets
          #Sampling location for Bbox
          reference_bboxs=reference_bboxs.sigmoid().unsqueeze(2).repeat(1, 1, input_spatial_shapes.shape[0], 1)
          sampling_locations[...,::2,:] = reference_bboxs[:, :, None, :, None, :2] \
                            + sampling_locations[...,::2,:] / self.n_points * reference_bboxs[:, :, None, :, None, 2:] * 0.5

          #Sampling location for Mask
          ref_masks=reference_masks>0
          mask_box_sig=get_bounding_boxes(ref_masks).unsqueeze(2).repeat(1, 1, input_spatial_shapes.shape[0], 1)
          sampling_locations[...,1::2,:] = mask_box_sig[:, :, None, :, None, :2] \
                            + sampling_locations[...,1::2,:] / self.n_points * mask_box_sig[:, :, None, :, None, 2:] * 0.5
          point_inside_mask=check_points_in_mask(ref_masks,sampling_locations[...,1::2,:].view(N, Len_q, self.n_heads*self.n_levels*(self.n_points//2), 2))
          attention_weights_panaty= torch.ones_like(attention_weights, dtype=torch.float32, device=attention_weights.device)
          attention_weights_panaty[...,1::2]=point_inside_mask.view(N, Len_q, self.n_heads,self.n_levels,(self.n_points//2))*1.0
          attention_weights=attention_weights*attention_weights_panaty
        
        elif self.type_sampling_location == "mask":
          ref_masks=reference_masks>0
          mask_box_sig=get_bounding_boxes(ref_masks).unsqueeze(2).repeat(1, 1, input_spatial_shapes.shape[0], 1)
          sampling_locations = mask_box_sig[:, :, None, :, None, :2] \
                            + sampling_offsets / self.n_points * mask_box_sig[:, :, None, :, None, 2:] * 0.5
          
          point_inside_mask=check_points_in_mask(ref_masks,sampling_locations.view(N, Len_q, self.n_heads*self.n_levels*self.n_points, 2))
          attention_weights_panaty=point_inside_mask.view(N, Len_q, self.n_heads,self.n_levels,self.n_points)*1.0
          attention_weights=attention_weights*attention_weights_panaty
          
        elif self.type_sampling_location == "bbox":
          reference_bboxs=reference_bboxs.sigmoid().unsqueeze(2).repeat(1, 1, input_spatial_shapes.shape[0], 1)
          sampling_locations = reference_bboxs[:, :, None, :, None, :2] \
                            + sampling_offsets / self.n_points * reference_bboxs[:, :, None, :, None, 2:] * 0.5
        else:
          raise NotImplementedError("cfg.MODEL.DynAMFormer.TYPE_SAMPLING_LOCATIONS is not valid.")

        attention_weights = F.softmax(attention_weights.view(N, Len_q, self.n_heads, self.n_levels * self.n_points), -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        try:
            output = MSDeformAttnFunction.apply(
                value, input_spatial_shapes, input_level_start_index, sampling_locations,  attention_weights, self.im2col_step)
        except:
            # CPU
            output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations,  attention_weights)
        # # For FLOPs calculation only
        # output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output
