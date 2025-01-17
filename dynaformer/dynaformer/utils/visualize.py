import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import matplotlib.patches as patches
import time

def visualize_attention(Reference_masks, Reference_bboxs, Sampling_locations, save_path='.'):
    """
    Vẽ Reference_masks màu trắng trên nền đen, Reference_bboxs màu xanh, Sampling_locations màu đỏ.
    Đối với mỗi mẫu trong batch, tạo một ảnh lớn có chứa Q ảnh con.
    Sau đó lưu ảnh với tên được đặt theo timestamp của hệ thống.
    
    Args:
        Reference_masks (tensor): Mask có shape (N, Q, H, W).
        Reference_bboxs (tensor): Bounding box có shape (N, Q, 4), giá trị chuẩn hóa [cx, cy, w, h].
        Sampling_locations (tensor): Các điểm sampling có shape (N, Q, P, 2), giá trị chuẩn hóa [x, y].
        save_path (str): Đường dẫn lưu file (mặc định là thư mục hiện tại).
    """
    
    N, Q, H, W = Reference_masks.shape
    
    # Loop qua từng mẫu trong batch
    for n in range(N):
        # Tạo một figure chứa Q ảnh con
        fig, axes = plt.subplots(1, Q, figsize=(Q * 5, 5))  # Q ảnh con trong một hàng
        
        for q in range(Q):
            ax = axes[q] if Q > 1 else axes  # Trường hợp Q = 1 sẽ không cần lấy phần tử từ mảng axes
            
            # Hiển thị mask nền đen và các đối tượng màu trắng
            mask = Reference_masks[n, q].cpu().numpy()  # Chuyển từ tensor trên GPU sang numpy array trên CPU
            ax.imshow(mask, cmap='gray', vmin=0, vmax=1)

            # Vẽ bounding boxes màu xanh (blue)
            cx, cy, w, h = Reference_bboxs[n, q].cpu().numpy()  # Chuyển tensor về numpy
            x1 = (cx - w / 2) * W
            y1 = (cy - h / 2) * H
            width = w * W
            height = h * H
            rect = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)

            # Vẽ các điểm sampling màu đỏ (red)
            sampling_points = Sampling_locations[n, q].detach().cpu().numpy()  # Chuyển tensor về numpy
            sampling_points[:, 0] = sampling_points[:, 0] * W  # scale x tọa độ
            sampling_points[:, 1] = sampling_points[:, 1] * H  # scale y tọa độ
            ax.scatter(sampling_points[:, 0], sampling_points[:, 1], color='red', s=10)

            # Tắt các thông tin trục
            ax.set_axis_off()

        # Lưu hình ảnh với tên theo timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_name = f"sample_{n}_visualization_{timestamp}.png"
        file_path = os.path.join(save_path, file_name)
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    print(f"Images saved to {save_path}")

#visualize(reference_masks[:,:20,:,:], mask_box_sig[:,:20,0,:], sampling_locations[:,:20,0,0,:,:], save_path='/content/sample_data')





def visualize_masks_and_bboxes(masks, bboxes, save_dir='.'):
    """
    Visualizes masks and corresponding bounding boxes and saves the image
    with a timestamped filename in the specified directory.

    Parameters:
    - masks: A numpy array of shape (N, H, W) containing the masks (binary).
    - bboxes: A numpy array of shape (N, 4) containing the bounding boxes
              in the format [cx, cy, w, h].
    - save_dir: A string specifying the directory to save the image.
    """
    
    # Kiểm tra và chuyển tensor về CPU nếu cần
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()  # Chuyển sang numpy array
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()  # Chuyển sang numpy array
    
    N, H, W = masks.shape  # Lấy H và W từ masks
    fig, ax = plt.subplots(1, N, figsize=(15, 5))
    
    for i in range(N):
        # Create a blank image
        img = np.zeros((H, W))  # Sử dụng H và W để tạo hình ảnh trắng
        
        # Overlay the mask
        img[masks[i] == 1] = 1  # Set the masked areas to 1 (or any other color)
        
        # Display the mask
        ax[i].imshow(img, cmap='gray', alpha=0.5)
        
        # Extract bounding box parameters
        cx, cy, w, h = bboxes[i]
        # Convert normalized coordinates to pixel values
        bbox_x = cx * W - (w * W) / 2
        bbox_y = cy * H - (h * H) / 2
        
        # Create a rectangle patch
        rect = patches.Rectangle((bbox_x, bbox_y), w * W, h * H,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax[i].add_patch(rect)
        
        ax[i].set_title(f'Mask {i+1}')
        ax[i].axis('off')
    
    plt.tight_layout()

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the figure with a timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(save_dir, f'masks_and_bboxes_{timestamp}.png'), bbox_inches='tight')
    plt.show()

#visualize_masks_and_bboxes(known_masks_expand[:5], known_boxes_expaned[:5], save_dir='/content/44444')