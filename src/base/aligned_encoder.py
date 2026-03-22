"""
Aligned Encoder V2 - matches train_phase3_v2.py architecture.
For use in Phase 4 RL training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import MathVisionModel
from world_encoder import WorldEncoder


class AlignedEncoder(nn.Module):
    """
    Unified model that projects both Math and World encoders into a shared latent space.
    
    The alignment is achieved by:
    1. Loading pretrained Math Encoder and World Encoder
    2. Adding projection layers to map both to the same dimensionality
    3. Training with a contrastive/alignment loss
    """
    
    def __init__(self, shared_dim=128, freeze_encoders=True):
        super(AlignedEncoder, self).__init__()
        
        self.shared_dim = shared_dim
        
        # Load pretrained encoders (we'll load weights later)
        self.math_encoder = MathVisionModel(img_size=64)
        self.world_encoder = WorldEncoder(img_size=64, latent_dim=256)
        
        # Freeze pretrained weights (optional)
        if freeze_encoders:
            for param in self.math_encoder.parameters():
                param.requires_grad = False
            for param in self.world_encoder.parameters():
                param.requires_grad = False
        
        # Projection heads to shared space
        # Math encoder outputs 256-dim from fc_shared
        self.math_projector = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, shared_dim)
        )
        
        # World encoder outputs 256-dim latent
        self.world_projector = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, shared_dim)
        )
        
    def load_pretrained(self, math_ckpt, world_ckpt, device='cpu'):
        """Load pretrained encoder weights."""
        self.math_encoder.load_state_dict(
            torch.load(math_ckpt, map_location=device, weights_only=True)
        )
        self.world_encoder.load_state_dict(
            torch.load(world_ckpt, map_location=device, weights_only=True)
        )
        print(f"Loaded pretrained encoders from {math_ckpt} and {world_ckpt}")
        
    def encode_math(self, math_img):
        """
        Encode a unit circle image into the shared latent space.
        Args:
            math_img: (B, 1, 64, 64) grayscale unit circle image
        Returns:
            shared_embedding: (B, shared_dim)
            direction: (B, 2) predicted sin/cos
        """
        # Get features from math encoder
        features = self.math_encoder.features(math_img)
        features = F.relu(self.math_encoder.fc_shared(features))
        
        # Get direction prediction
        direction = self.math_encoder.head_geo(features)  # (B, 2) [sin, cos]
        
        # Project to shared space
        shared_embedding = self.math_projector(features)
        shared_embedding = F.normalize(shared_embedding, p=2, dim=1)
        
        return shared_embedding, direction
    
    def encode_world(self, world_img):
        """
        Encode a gridworld image into the shared latent space.
        Args:
            world_img: (B, 3, 64, 64) RGB gridworld image
        Returns:
            shared_embedding: (B, shared_dim)
            direction: (B, 2) predicted dx/dy to target
        """
        latent, direction = self.world_encoder(world_img)
        
        # Project to shared space
        shared_embedding = self.world_projector(latent)
        shared_embedding = F.normalize(shared_embedding, p=2, dim=1)
        
        return shared_embedding, direction
    
    def forward(self, math_img=None, world_img=None):
        """
        Forward pass for either or both inputs.
        """
        outputs = {}
        
        if math_img is not None:
            math_emb, math_dir = self.encode_math(math_img)
            outputs['math_embedding'] = math_emb
            outputs['math_direction'] = math_dir
            
        if world_img is not None:
            world_emb, world_dir = self.encode_world(world_img)
            outputs['world_embedding'] = world_emb
            outputs['world_direction'] = world_dir
            
        return outputs


class AlignmentLoss(nn.Module):
    """
    Loss function that aligns embeddings when they represent the same direction.
    
    For a pair (math_img, world_img) with the same target direction:
    - Their embeddings should be close (cosine similarity → 1)
    - Their predicted directions should match the ground truth
    """
    
    def __init__(self, direction_weight=1.0, alignment_weight=1.0):
        super(AlignmentLoss, self).__init__()
        self.direction_weight = direction_weight
        self.alignment_weight = alignment_weight
        self.mse = nn.MSELoss()
        self.cosine = nn.CosineSimilarity(dim=1)
        
    def forward(self, math_emb, world_emb, math_dir_pred, world_dir_pred, direction_gt):
        """
        Args:
            math_emb: (B, shared_dim) math embedding
            world_emb: (B, shared_dim) world embedding  
            math_dir_pred: (B, 2) predicted direction from math encoder
            world_dir_pred: (B, 2) predicted direction from world encoder
            direction_gt: (B, 2) ground truth direction [cos, sin] or [dx, dy]
        """
        # Direction prediction loss
        loss_math_dir = self.mse(math_dir_pred, direction_gt)
        loss_world_dir = self.mse(world_dir_pred, direction_gt)
        direction_loss = (loss_math_dir + loss_world_dir) / 2
        
        # Embedding alignment loss (maximize cosine similarity)
        # cosine_sim is in [-1, 1], we want it to be 1
        cosine_sim = self.cosine(math_emb, world_emb)
        alignment_loss = 1 - cosine_sim.mean()  # Convert to loss (0 when perfect)
        
        total_loss = (self.direction_weight * direction_loss + 
                      self.alignment_weight * alignment_loss)
        
        return total_loss, {
            'direction_loss': direction_loss.item(),
            'alignment_loss': alignment_loss.item(),
            'cosine_similarity': cosine_sim.mean().item()
        }