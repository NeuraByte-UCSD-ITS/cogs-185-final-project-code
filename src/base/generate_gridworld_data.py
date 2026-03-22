"""
Generate dataset of gridworld frames for training the World Encoder.
Each sample includes: image, direction to target (dx, dy), distance to target.
"""
import os
import csv
import cv2
import numpy as np
from tqdm import tqdm
import random
import math

from gridworld_env import GridWorldEnv

def generate_gridworld_dataset(output_dir, num_samples, grid_size=8, cell_size=8, num_obstacles=3, split_name="train"):
    """
    Generate a dataset by sampling random states from the gridworld.
    """
    save_dir = os.path.join(output_dir, split_name)
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    csv_path = os.path.join(save_dir, "labels.csv")
    
    env = GridWorldEnv(grid_size=grid_size, cell_size=cell_size, num_obstacles=num_obstacles)
    
    print(f"Generating {num_samples} gridworld samples for '{split_name}'...")
    
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'dx', 'dy', 'distance', 'angle'])
        
        for i in tqdm(range(num_samples)):
            # Reset to a random configuration
            obs = env.reset()
            
            # Get direction info
            dx, dy = env.get_direction_to_target()
            
            # Distance (in grid cells)
            agent_r, agent_c = env.agent_pos
            target_r, target_c = env.target_pos
            distance = math.sqrt((target_r - agent_r)**2 + (target_c - agent_c)**2)
            
            # Angle in radians
            angle = math.atan2(dy, dx)
            
            filename = f"frame_{i:05d}.png"
            filepath = os.path.join(img_dir, filename)
            cv2.imwrite(filepath, obs)
            
            writer.writerow([
                filename,
                f"{dx:.6f}",
                f"{dy:.6f}",
                f"{distance:.6f}",
                f"{angle:.6f}"
            ])
    
    print(f"Dataset saved to {save_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/phase2")
    parser.add_argument("--train_samples", type=int, default=5000)
    parser.add_argument("--val_samples", type=int, default=1000)
    args = parser.parse_args()
    
    generate_gridworld_dataset(args.output_dir, args.train_samples, split_name="train")
    generate_gridworld_dataset(args.output_dir, args.val_samples, split_name="val")
