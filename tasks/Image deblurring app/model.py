import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class MobileNetDeblurNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Load lightweight MobileNet backbone for texture feature extraction
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.encoder = backbone.features  # Feature map output: [B, 576, H/32, W/32]
        
        # TODO: Implement Feature Pyramid Network (FPN) connections to preserve fine edges
        self.decoder = nn.Sequential(
            nn.Conv2d(576, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # TODO: Add transposed convolutions or bilinear upsampling blocks back to original image size
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extract features using MobileNet
        features = self.encoder(x)
        
        # Decode features
        out = self.decoder(features)
        
        # TODO: Implement multi-scale global residual skip connection: return out + x
        return out