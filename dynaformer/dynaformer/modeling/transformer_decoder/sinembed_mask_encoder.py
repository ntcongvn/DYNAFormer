import torch
from torch import nn, Tensor
import math
import torch.nn.functional as F
from ...utils.utils import gen_sineembed_for_position

def matrix_sinusoidal_embedding(H,W,dim=16,device="cuda"):
    dim=dim/2;
    scale = 2 * math.pi
    mask = torch.ones((1,1,H, W), device="cuda", dtype=torch.bool)
    y_embed = mask.cumsum(2, dtype=torch.float32)
    x_embed = mask.cumsum(3, dtype=torch.float32)
    eps = 1e-6
    y_embed = y_embed / (y_embed[:,:,-1:, :] + eps) * scale
    x_embed = x_embed / (x_embed[:,:, :, -1:] + eps) * scale

    dim_t = torch.arange(dim, dtype=torch.float32, device=device)
    dim_t = 10000 ** (2 * (dim_t // 2) / dim)

    pos_x = x_embed[:,:, :, :, None] / dim_t
    pos_y = y_embed[:,:, :, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:,:, :, :, 0::2].sin(), pos_x[:,:, :, :, 1::2].cos()), dim=4
    ).flatten(4)
    pos_y = torch.stack(
        (pos_y[:,:, :, :, 0::2].sin(), pos_y[:,:, :, :, 1::2].cos()), dim=4
    ).flatten(4)
    pos = torch.cat((pos_y, pos_x), dim=4).permute(0,1, 4, 2, 3)
    return pos     #1*1*16*H*W

def get_sinusoidal_embedding(shape,device):
    H,W=shape
    start_h = torch.tensor(0.0, device=device)
    end_h = torch.tensor(1.0, device=device)

    h_coords_normalized = torch.linspace(start_h, end_h, steps=H,device=device).unsqueeze(1).repeat(1, W)  # H x W
    w_coords_normalized = torch.linspace(start_h, end_h, steps=W,device=device).unsqueeze(0).repeat(H, 1)  # H x W
    points=torch.cat((h_coords_normalized.unsqueeze(-1), w_coords_normalized.unsqueeze(-1)), dim=-1)
    embedding = gen_sineembed_for_position(points)
    embedding = embedding.reshape(H*W,-1)
    return embedding



def apply_mask_and_pool(mask, embedding):
    
    Q, N, H, W = mask.shape
    
    # Flatten the mask and embedding -> shape (Q, N, H*W)
    mask_flatten = mask.reshape(Q * N, H * W)
    
    # Weight the embedding by the mask -> shape (Q*N,C)
    if mask_flatten.shape[1] != embedding.shape[0]:
      print(mask_flatten.shape)
      print(embedding.shape)
    weighted_embedding = torch.einsum('qn, nc -> qc', mask_flatten, embedding)
    
    # Normalize by the sum of the mask (to avoid biasing by larger masks)
    mask_sum = mask_flatten.sum(dim=1, keepdim=True)  # shape (Q*N, 1)
    pooled_embedding = weighted_embedding / (mask_sum + 1e-6)  # Avoid division by zero
    
    # Reshape to (Q, N, C)
    pooled_embedding = pooled_embedding.view(Q, N, -1)
    
    return pooled_embedding


def gen_sineembed_for_mask(mask_tensor,sinusoidal_emb,spatial_shapes=None):
    #d_model=torch.tensor(d_model).to(mask_tensor.device)
    if len(mask_tensor.shape) == 4:
        if spatial_shapes is not None:
          mask_tensor = F.interpolate(mask_tensor, size=(int(spatial_shapes[0].item()), int(spatial_shapes[1].item())), mode='bilinear', align_corners=False)
        Q, N, H, W=mask_tensor.shape
        # Get sinusoidal embedding for the given mask size
        #sinusoidal_emb = get_sinusoidal_embedding(H, W,str(mask_tensor.device))  # shape (H*W, 2C)

        # Apply mask and pool the results
        mask_emb = apply_mask_and_pool(mask_tensor, sinusoidal_emb)  # shape (Q, N, 2C)
    else:
        raise ValueError("Unknown masl_tensor shape:{}".format(mask_tensor.shape))
    return mask_emb
