"""
World Encoder Model for Phase 2.
Takes gridworld images and outputs a latent representation + direction prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class WorldEncoder(nn.Module):
    """
    CNN encoder for gridworld frames.
    Outputs:
        - latent (256-dim): Learned representation of the scene
        - direction (2-dim): Predicted (dx, dy) normalized direction to target
    """
    def __init__(self, img_size=64, latent_dim=256):
        super(WorldEncoder, self).__init__()
        
        # Input: (B, 3, 64, 64) - RGB image
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (32, 32, 32)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (64, 16, 16)
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (128, 8, 8)
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (256, 4, 4)
            
            nn.Flatten()
        )
        
        # Feature size: 256 * 4 * 4 = 4096
        feature_size = 256 * (img_size // 16) ** 2
        
        # Latent projection
        self.fc_latent = nn.Linear(feature_size, latent_dim)
        
        # Direction prediction head (dx, dy)
        self.head_direction = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh()  # dx, dy in [-1, 1]
        )
        
    def forward(self, x):
        features = self.features(x)
        latent = F.relu(self.fc_latent(features))
        direction = self.head_direction(latent)
        return latent, direction
    
    def encode(self, x):
        """Get only the latent representation."""
        features = self.features(x)
        latent = F.relu(self.fc_latent(features))
        return latent


if __name__ == "__main__":
    # Test
    model = WorldEncoder(img_size=64, latent_dim=256)
    dummy_input = torch.randn(4, 3, 64, 64)
    latent, direction = model(dummy_input)
    print(f"Latent shape: {latent.shape}")
    print(f"Direction shape: {direction.shape}")
