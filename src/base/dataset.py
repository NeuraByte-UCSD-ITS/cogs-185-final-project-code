import os
import csv
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class UnitCircleDataset(Dataset):
    def __init__(self, data_dir):
        """
        Args:
            data_dir (str): Path to data directory (e.g., 'data/phase1/train') containing images/ and labels.csv
        """
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, "images")
        self.labels = []
        
        csv_path = os.path.join(data_dir, "labels.csv")
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.labels.append(row)
                
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        row = self.labels[idx]
        img_name = row['filename']
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image (Grayscale)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Normalize to 0-1 and add channel dim -> (1, H, W)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Targets
        sin_val = float(row['sin'])
        cos_val = float(row['cos'])
        quadrant = int(row['quadrant'])
        
        # Returns: image_tensor, (geo_target, quad_target)
        return torch.from_numpy(img), (torch.tensor([sin_val, cos_val], dtype=torch.float32), torch.tensor(quadrant, dtype=torch.long))
