"""
Training script for Phase 2: World Encoder.
Trains the encoder to predict direction to target from gridworld images.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

from gridworld_dataset import GridworldDataset
from world_encoder import WorldEncoder

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    criterion = nn.MSELoss()
    
    for imgs, direction_targets in loader:
        imgs = imgs.to(device)
        direction_targets = direction_targets.to(device)
        
        optimizer.zero_grad()
        
        _, direction_pred = model(imgs)
        loss = criterion(direction_pred, direction_targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    running_loss = 0.0
    total_angle_error = 0.0
    total_samples = 0
    
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for imgs, direction_targets in loader:
            imgs = imgs.to(device)
            direction_targets = direction_targets.to(device)
            
            _, direction_pred = model(imgs)
            loss = criterion(direction_pred, direction_targets)
            running_loss += loss.item()
            
            # Angle error
            pred_angle = torch.atan2(direction_pred[:, 1], direction_pred[:, 0])
            target_angle = torch.atan2(direction_targets[:, 1], direction_targets[:, 0])
            
            diff = torch.abs(pred_angle - target_angle)
            diff = torch.min(diff, 2 * math.pi - diff)
            total_angle_error += torch.sum(torch.rad2deg(diff)).item()
            total_samples += imgs.size(0)
    
    val_loss = running_loss / len(loader)
    mean_angle_error = total_angle_error / total_samples
    
    return val_loss, mean_angle_error

def main():
    BATCH_SIZE = 64
    LR = 0.001
    EPOCHS = 15
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    train_dir = 'data/phase2/train'
    val_dir = 'data/phase2/val'
    
    train_ds = GridworldDataset(train_dir)
    val_ds = GridworldDataset(val_dir)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = WorldEncoder(img_size=64, latent_dim=256).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_loss = float('inf')
    
    print("Starting World Encoder Training...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_loss, val_angle_err = validate(model, val_loader, DEVICE)
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  > Angle Err: {val_angle_err:.2f} deg")
        
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/phase2_world_encoder.pth')
            print("  * Saved best model *")
    
    print("Training Complete.")

if __name__ == "__main__":
    main()
