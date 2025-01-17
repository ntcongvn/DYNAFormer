# ------------------------------------------------------------------------
# Created by Tan-Cong Nguyen
# ------------------------------------------------------------------------
import torch
import copy
from torch import device, nn, Tensor
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from PIL import Image

def test_draw_masks(mask_tensor, save_path):
    """
    Draws masks on a single image where transparency depends on mask value (0 for fully transparent, 
    1 for fully opaque), and each mask has a different color. The background will be black.

    Parameters:
    - mask_tensor (Tensor or numpy array): A tensor of shape (N, H, W) where N is the number of masks,
                                           H is the height, and W is the width. Supports PyTorch tensors or NumPy arrays.
    - save_path (str): The file path where the resulting image will be saved.
    """
    # Convert PyTorch tensor to NumPy array if necessary
    if torch.is_tensor(mask_tensor):
        mask_tensor = mask_tensor.detach().cpu().numpy()

    N, H, W = mask_tensor.shape

    # Define random colors for each mask (RGB)
    colors = np.random.rand(N, 3)

    # Create a blank RGBA canvas with a black background (RGB = 0, Alpha = 0)
    result_image = np.zeros((H, W, 4), dtype=np.float32)

    for i in range(N):
        mask = mask_tensor[i]
        color = colors[i]

        # Reshape mask to have an extra dimension for broadcasting over RGB channels
        mask_3d = mask[..., np.newaxis]

        # Accumulate RGB values with the mask (black background remains where mask is 0)
        result_image[..., :3] += color * mask_3d  # Broadcasting mask over color

        # Set the alpha channel: 0 for mask=0 (fully transparent), 1 for mask=1 (fully opaque)
        result_image[..., 3] = np.maximum(result_image[..., 3], mask)

    # Invert alpha channel: 0 -> fully transparent, 1 -> fully opaque
    result_image[..., 3] = 1 - result_image[..., 3]

    # Convert result to an image format: scale to [0, 255] and uint8
    result_image = (result_image * 255).astype(np.uint8)
    pil_image = Image.fromarray(result_image, 'RGBA')

    # Save the image
    pil_image.save(save_path)


def plot_bboxes_colored_save(bboxes, W, H, save_path):
    """
    Visualize bounding boxes on an image of width W and height H with different colors,
    and save the figure to a specified path.
    
    Args:
        bboxes (torch.Tensor or numpy.ndarray): Tensor of shape (N, 4), where each row is (cx, cy, w, h)
                                in normalized format (between 0 and 1). Can be a CUDA tensor or CPU tensor.
        W (int): Width of the image.
        H (int): Height of the image.
        save_path (str): The path where the figure will be saved (e.g., 'output/figure.png').
    """
    # Ensure bboxes are moved to CPU and converted to numpy if they are a CUDA tensor
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    
    # Create a figure and axis
    fig, ax = plt.subplots(1)
    
    # Draw an empty image (white background)
    ax.imshow(np.ones((H, W, 3)))
    
    # Generate random colors for each bounding box
    colors = []
    for _ in range(bboxes.shape[0]):
        colors.append((random.random(), random.random(), random.random()))
    
    # Loop through all the bounding boxes
    for i, bbox in enumerate(bboxes):
        # Extract cx, cy, w, h from normalized bbox
        cx, cy, w, h = bbox
        
        # Convert normalized values to pixel coordinates
        px = cx * W
        py = cy * H
        pw = w * W
        ph = h * H
        
        # Create a rectangle for the bbox (Rectangle takes top-left x, y, width, height)
        rect = patches.Rectangle(
            (px - pw / 2, py - ph / 2), pw, ph, 
            linewidth=2, edgecolor=colors[i], facecolor='none'
        )
        
        # Add the rectangle to the plot
        ax.add_patch(rect)
    
    # Set the aspect ratio to ensure the image isn't distorted
    ax.set_aspect('equal')
    
    # Save the figure to the specified path
    plt.savefig(save_path, bbox_inches='tight')  # 'bbox_inches' ensures nothing gets cropped
    
    # Close the figure to free memory
    plt.close()