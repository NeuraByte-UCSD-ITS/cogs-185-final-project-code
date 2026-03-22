import torch
import torch.nn as nn
import torch.nn.functional as F

class MathVisionModel(nn.Module):
    def __init__(self, img_size=64):
        super(MathVisionModel, self).__init__()
        
        # Simple CNN Backbone
        # Input: (B, 1, 64, 64)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> (32, 32, 32)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> (64, 16, 16)
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> (128, 8, 8)
            
            nn.Flatten()
        )
        
        # Latent size after flatten: 128 * 8 * 8 = 8192
        self.fc_shared = nn.Linear(128 * (img_size // 8) * (img_size // 8), 256)
        
        # Head 1: Geometry Regression (Sin, Cos) - 2 values
        # We use Tanh because sin/cos are in [-1, 1]
        self.head_geo = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2), 
            nn.Tanh() 
        )
        
        # Head 2: Quadrant Classification - 4 logits
        self.head_quad = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        features = self.features(x)
        features = F.relu(self.fc_shared(features))
        
        geo_out = self.head_geo(features) # (B, 2) -> [sin, cos]
        quad_out = self.head_quad(features) # (B, 4) -> logits
        
        return geo_out, quad_out
