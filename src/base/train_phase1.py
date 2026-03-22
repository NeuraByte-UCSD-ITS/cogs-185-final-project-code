import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

from dataset import UnitCircleDataset
from model import MathVisionModel

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    running_geo_loss = 0.0
    running_quad_loss = 0.0
    
    criterion_geo = nn.MSELoss()
    criterion_quad = nn.CrossEntropyLoss()
    
    for imgs, (geo_targets, quad_targets) in loader:
        imgs = imgs.to(device)
        geo_targets = geo_targets.to(device)
        quad_targets = quad_targets.to(device)
        
        optimizer.zero_grad()
        
        geo_pred, quad_pred = model(imgs)
        
        loss_geo = criterion_geo(geo_pred, geo_targets)
        loss_quad = criterion_quad(quad_pred, quad_targets)
        
        # Combined loss: Simple sum for now, can weigh them if needed
        loss = loss_geo + loss_quad
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_geo_loss += loss_geo.item()
        running_quad_loss += loss_quad.item()
        
    return running_loss / len(loader), running_geo_loss / len(loader), running_quad_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    running_loss = 0.0
    
    total_samples = 0
    correct_quad = 0
    total_geo_error = 0.0 # MAE for sin/cos
    total_angle_error = 0.0 # Degrees
    
    criterion_geo = nn.MSELoss()
    criterion_quad = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for imgs, (geo_targets, quad_targets) in loader:
            imgs = imgs.to(device)
            geo_targets = geo_targets.to(device)
            quad_targets = quad_targets.to(device)
            
            geo_pred, quad_pred = model(imgs)
            
            loss_geo = criterion_geo(geo_pred, geo_targets)
            loss_quad = criterion_quad(quad_pred, quad_targets)
            loss = loss_geo + loss_quad
            running_loss += loss.item()
            
            # Metrics
            total_samples += imgs.size(0)
            
            # Quadrant Accuracy
            _, predicted_quad = torch.max(quad_pred, 1)
            correct_quad += (predicted_quad == quad_targets).sum().item()
            
            # Geometry Error (MAE)
            total_geo_error += torch.abs(geo_pred - geo_targets).sum().item()
            
            # Angle Error (Degrees)
            # geo_targets: [sin, cos]
            # pred: [sin, cos]
            # atan2(sin, cos) -> angle
            # Note: atan2 takes (y, x) -> (sin, cos)
            
            pred_sin = geo_pred[:, 0]
            pred_cos = geo_pred[:, 1]
            target_sin = geo_targets[:, 0]
            target_cos = geo_targets[:, 1]
            
            pred_angle = torch.atan2(pred_sin, pred_cos) # [-pi, pi]
            target_angle = torch.atan2(target_sin, target_cos)
            
            diff = torch.abs(pred_angle - target_angle)
            # Handle wrapping: |a - b| > pi -> 2pi - |a - b|
            diff = torch.min(diff, 2*math.pi - diff)
            
            total_angle_error += torch.sum(torch.rad2deg(diff)).item()
            
    val_loss = running_loss / len(loader)
    acc_quad = 100.0 * correct_quad / total_samples
    mae_geo = total_geo_error / (total_samples * 2) # 2 values per sample
    mean_angle_error = total_angle_error / total_samples
    
    return val_loss, acc_quad, mae_geo, mean_angle_error

def main():
    # Config
    BATCH_SIZE = 64
    LR = 0.001
    EPOCHS = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    # Data
    train_dir = 'data/phase1/train'
    val_dir = 'data/phase1/val'
    
    train_ds = UnitCircleDataset(train_dir)
    val_ds = UnitCircleDataset(val_dir)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = MathVisionModel(img_size=64).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_loss = float('inf')
    
    print("Starting Training...")
    for epoch in range(EPOCHS):
        train_loss, train_geo, train_quad = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_loss, val_acc, val_mae, val_angle_err = validate(model, val_loader, DEVICE)
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} (Geo: {train_geo:.4f}, Quad: {train_quad:.4f})")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  > Quad Acc:     {val_acc:.2f}%")
        print(f"  > Sin/Cos MAE:  {val_mae:.4f}")
        print(f"  > Angle Err:    {val_angle_err:.2f} deg")
        
        # Save Best
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/phase1_best.pth')
            print("  * Saved best model *")
            
    print("Training Complete.")

if __name__ == "__main__":
    main()
