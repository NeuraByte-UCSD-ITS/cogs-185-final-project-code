"""
Gym-compatible wrapper for the GridWorld environment.
This allows us to use standard RL libraries like Stable Baselines 3.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gridworld_env import GridWorldEnv
from aligned_encoder import AlignedEncoder
from world_encoder import WorldEncoder


class GridWorldGymEnv(gym.Env):
    """
    Gym-compatible wrapper for GridWorld.
    
    Three modes:
    1. 'pixels' - Raw pixel observations (baseline)
    2. 'aligned' - Aligned encoder embeddings (128-dim normalized)
    3. 'encoder' - World encoder latent space (256-dim, better variance)
    """
    
    metadata = {'render_modes': ['rgb_array']}
    
    def __init__(self, observation_mode='pixels', encoder_checkpoint=None, 
                 grid_size=8, cell_size=8, num_obstacles=3, max_steps=50):
        super().__init__()
        
        self.observation_mode = observation_mode
        self.env = GridWorldEnv(
            grid_size=grid_size,
            cell_size=cell_size,
            num_obstacles=num_obstacles,
            max_steps=max_steps
        )
        
        # Action space: 4 discrete actions (Up, Down, Left, Right)
        self.action_space = spaces.Discrete(4)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Observation space depends on mode
        if observation_mode == 'pixels':
            # RGB image in channel-last format (H, W, C) for CnnPolicy
            img_size = grid_size * cell_size
            self.observation_space = spaces.Box(
                low=0, high=255, 
                shape=(img_size, img_size, 3), 
                dtype=np.uint8
            )
            self.encoder = None
        elif observation_mode == 'aligned':
            # 128-dimensional aligned embedding (normalized)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(128,),
                dtype=np.float32
            )
            self.encoder = AlignedEncoder(shared_dim=128, freeze_encoders=True).to(self.device)
            if encoder_checkpoint and os.path.exists(encoder_checkpoint):
                self.encoder.load_state_dict(
                    torch.load(encoder_checkpoint, map_location=self.device, weights_only=True)
                )
            self.encoder.eval()
        elif observation_mode == 'encoder':
            # 256-dimensional world encoder latent (better variance for RL)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(256,),
                dtype=np.float32
            )
            self.encoder = WorldEncoder(img_size=64, latent_dim=256).to(self.device)
            if encoder_checkpoint and os.path.exists(encoder_checkpoint):
                self.encoder.load_state_dict(
                    torch.load(encoder_checkpoint, map_location=self.device, weights_only=True)
                )
            self.encoder.eval()
        else:
            raise ValueError(f"Unknown observation_mode: {observation_mode}")
    
    def _get_obs(self, raw_obs):
        """Convert raw pixel observation to the appropriate format."""
        if self.observation_mode == 'pixels':
            # Return as uint8 channel-last for CnnPolicy
            return raw_obs.astype(np.uint8)
        elif self.observation_mode == 'aligned':
            # Get embedding from aligned encoder
            with torch.no_grad():
                obs = raw_obs.astype(np.float32) / 255.0
                obs = np.transpose(obs, (2, 0, 1))  # (C, H, W)
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                
                embedding, _ = self.encoder.encode_world(obs_tensor)
                return embedding.cpu().numpy().flatten()
        elif self.observation_mode == 'encoder':
            # Get latent from world encoder (Phase 2)
            with torch.no_grad():
                obs = raw_obs.astype(np.float32) / 255.0
                obs = np.transpose(obs, (2, 0, 1))  # (C, H, W)
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                
                latent = self.encoder.encode(obs_tensor)
                return latent.cpu().numpy().flatten()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        raw_obs = self.env.reset()
        obs = self._get_obs(raw_obs)
        info = {
            'agent_pos': self.env.agent_pos,
            'target_pos': self.env.target_pos
        }
        return obs, info
    
    def step(self, action):
        raw_obs, reward, terminated, info = self.env.step(action)
        obs = self._get_obs(raw_obs)
        truncated = False  # We handle max_steps in the env itself
        return obs, reward, terminated, truncated, info
    
    def render(self):
        return self.env._render()
    
    def close(self):
        pass


if __name__ == "__main__":
    # Test pixel mode
    print("Testing pixel mode...")
    env = GridWorldGymEnv(observation_mode='pixels')
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    # Test aligned mode
    print("\nTesting aligned mode...")
    env_aligned = GridWorldGymEnv(
        observation_mode='aligned',
        encoder_checkpoint='checkpoints/phase3_aligned.pth'
    )
    obs, info = env_aligned.reset()
    print(f"Observation shape: {obs.shape}")
    
    # Take a few steps
    for i in range(5):
        action = env_aligned.action_space.sample()
        obs, reward, done, truncated, info = env_aligned.step(action)
        print(f"Step {i+1}: reward={reward:.2f}, done={done}")
        if done:
            break
    
    print("\nGym wrapper test complete!")
