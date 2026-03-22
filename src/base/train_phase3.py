"""
Training script for Phase 3: Shared Latent Space Alignment.
Trains the AlignedEncoder to map math and world embeddings to the same space.
"""
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import math

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paired_dataset import PairedDataset
from aligned_encoder import AlignedEncoder, AlignmentLoss


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_dir_loss = 0.0
    running_align_loss = 0.0
    running_cosine = 0.0
    
    for math_imgs, world_imgs, directions in loader:
        math_imgs = math_imgs.to(device)
        world_imgs = world_imgs.to(device)
        directions = directions.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(math_img=math_imgs, world_img=world_imgs)
        
        loss, metrics = criterion(
            outputs['math_embedding'],
            outputs['world_embedding'],
            outputs['math_direction'],
            outputs['world_direction'],
            directions
        )
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_dir_loss += metrics['direction_loss']
        running_align_loss += metrics['alignment_loss']
        running_cosine += metrics['cosine_similarity']
    
    n = len(loader)
    return {
        'total_loss': running_loss / n,
        'direction_loss': running_dir_loss / n,
        'alignment_loss': running_align_loss / n,
        'cosine_similarity': running_cosine / n
    }


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_cosine = 0.0
    total_angle_error = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for math_imgs, world_imgs, directions in loader:
            math_imgs = math_imgs.to(device)
            world_imgs = world_imgs.to(device)
            directions = directions.to(device)
            
            outputs = model(math_img=math_imgs, world_img=world_imgs)
            
            loss, metrics = criterion(
                outputs['math_embedding'],
                outputs['world_embedding'],
                outputs['math_direction'],
                outputs['world_direction'],
                directions
            )
            
            running_loss += loss.item()
            running_cosine += metrics['cosine_similarity']
            
            # Calculate angle error for world predictions
            pred_angle = torch.atan2(outputs['world_direction'][:, 1], outputs['world_direction'][:, 0])
            target_angle = torch.atan2(directions[:, 1], directions[:, 0])
            
            diff = torch.abs(pred_angle - target_angle)
            diff = torch.min(diff, 2 * math.pi - diff)
            total_angle_error += torch.sum(torch.rad2deg(diff)).item()
            total_samples += math_imgs.size(0)
    
    n = len(loader)
    return {
        'val_loss': running_loss / n,
        'cosine_similarity': running_cosine / n,
        'angle_error_deg': total_angle_error / total_samples
    }


def main():
    BATCH_SIZE = 64
    LR = 0.001
    EPOCHS = 15
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    # Data
    train_dir = 'data/phase3/train'
    val_dir = 'data/phase3/val'
    
    train_ds = PairedDataset(train_dir)
    val_ds = PairedDataset(val_dir)
    
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = AlignedEncoder(shared_dim=128, freeze_encoders=True).to(DEVICE)
    
    # Load pretrained encoders
    model.load_pretrained(
        'checkpoints/phase1_best.pth',
        'checkpoints/phase2_world_encoder.pth',
        device=DEVICE
    )
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = AlignmentLoss(direction_weight=1.0, alignment_weight=0.5)
    
    best_loss = float('inf')
    
    print("Starting Alignment Training...")
    for epoch in range(EPOCHS):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics = validate(model, val_loader, criterion, DEVICE)
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_metrics['total_loss']:.4f} (Dir: {train_metrics['direction_loss']:.4f}, Align: {train_metrics['alignment_loss']:.4f})")
        print(f"  Train Cosine Sim: {train_metrics['cosine_similarity']:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"  Val Cosine Sim: {val_metrics['cosine_similarity']:.4f}")
        print(f"  Val Angle Err: {val_metrics['angle_error_deg']:.2f} deg")
        
        if val_metrics['val_loss'] < best_loss:
            best_loss = val_metrics['val_loss']
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/phase3_aligned.pth')
            print("  * Saved best model *")
    
    print("Training Complete.")


if __name__ == "__main__":
    main()