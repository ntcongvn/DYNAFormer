from typing import Optional, List, Union
import torch
from torch import nn, Tensor
from torch.cuda.amp import autocast
import torch.nn.functional as F
import math

class LightMaskEncoder(nn.Module):
    def __init__(self, H=114, W=114, out_dim=256):  # Kích thước đầu ra vẫn là 512
        super(LightMaskEncoder, self).__init__()
        
        # Positional Encoding - Created once and reused
        #self.register_buffer('positional_encoding', matrix_sinusoidal_embedding(H, W)) #H*W*256
        
        # Lớp Depthwise Separable Convolution để giảm số tham số
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)   
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  
        self.bn4 = nn.BatchNorm2d(256)
        # Global Average Pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected layer to get 256 dimension output
        self.fc = nn.Linear(256, out_dim)

        # Initialize parameters
        self.reset_parameters()


    def reset_parameters(self):
        
        #Reset or initialize the parameters of the network.
        
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv4.weight, mode='fan_out', nonlinearity='relu')
        if self.conv1.bias is not None:
            nn.init.constant_(self.conv1.bias, 0)
        if self.conv2.bias is not None:
            nn.init.constant_(self.conv2.bias, 0)
        if self.conv3.bias is not None:
            nn.init.constant_(self.conv3.bias, 0)
        if self.conv4.bias is not None:
            nn.init.constant_(self.conv4.bias, 0)

        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    #@autocast()  # Enable mixed precision computation
    def forward(self, x, spatial_shape=None):
        
        #x: input tensor of shape (Q, N, H, W)
        #Output: tensor of shape (Q, N, 256)
        
        if spatial_shape is not None:
          x = F.interpolate(x, size=(int(spatial_shape[0].item()), int(spatial_shape[1].item())), mode='bilinear', align_corners=False)
        
        x= x.unsqueeze(2)
        Q, N,C, H, W = x.shape
           #1*1*32*H*W
        # Flatten the batch and query dimensions
        x = x.reshape(Q * N, C, H, W)
        
        # Pass through lightweight CNN layers (optimized for GPU)
        x = F.relu(self.bn1(self.conv1(x)))  # (Q*N, 32, H/2, W/2)
        x = F.relu(self.bn2(self.conv2(x)))  # (Q*N, 64, H/4, W/4)
        x = F.relu(self.bn3(self.conv3(x)))  # (Q*N, 128, H/8, W/8)
        x = F.relu(self.bn4(self.conv4(x)))  # (Q*N, 128, H/8, W/8)
        
        # Apply global average pooling
        x = self.global_avg_pool(x)  # (Q*N, 256, 1, 1)
        
        # Flatten and apply fully connected layer
        x = x.view(Q * N, -1)  # Flatten
        x = self.fc(x)  # Shape (Q*N, 256)
        
        # Reshape back to (Q, N, 256)
        x = x.view(Q, N, -1)
        return x