import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os

MAX_NO_POINTS = 1300

class PointDataset(Dataset):
    def __init__(self, data_dir, split_files, max_points=MAX_NO_POINTS, normalize=True, labels_mode = True):
        self.data_dir = data_dir
        self.max_points = max_points
        self.normalize = normalize
        self.examples = []
        self.labels_mode = labels_mode
        
        # Keep track of min/max coordinates for normalization
        self.min_x = float('inf')
        self.min_y = float('inf')
        self.max_x = float('-inf')
        self.max_y = float('-inf')
        
        # First pass: find min/max coordinates if normalizing
        if normalize:
            for file in split_files:
                points_file = os.path.join(data_dir, f"pg_{file}_points.txt")
                points = np.loadtxt(points_file)
                self.min_x = min(self.min_x, points[:, 0].min())
                self.min_y = min(self.min_y, points[:, 1].min())
                self.max_x = max(self.max_x, points[:, 0].max())
                self.max_y = max(self.max_y, points[:, 1].max())
        
        # Second pass: load and process data
        for file in split_files:
            points_file = os.path.join(data_dir, f"pg_{file}_points.txt")
            points = np.loadtxt(points_file)
            if self.labels_mode == True:
                labels_file = os.path.join(data_dir, f"pg_{file}_labels.txt")
                labels = np.loadtxt(labels_file).astype(int)
            
            
            # Pad if necessary
            if len(points) < max_points:
                pad_length = max_points - len(points)
                points = np.pad(points, ((0, pad_length), (0, 0)), mode='constant')

                if self.labels_mode == True:
                    labels = np.pad(labels, (0, pad_length), mode='constant', constant_values=-1)
            
            # Normalize points if requested
            if normalize:
                points[:, 0] = (points[:, 0] - self.min_x) / (self.max_x - self.min_x)
                #points[:, 1] = (points[:, 1] - self.min_y) / (self.max_y - self.min_y)
                points[:, 1] = 1 - (points[:, 1] - self.min_y) / (self.max_y - self.min_y)
            
            # Shuffle points and labels together
            indices = list(range(len(points)))
            random.shuffle(indices)
            points = points[indices]
            if self.labels_mode == True:
                labels = labels[indices]
                self.examples.append((points, labels))
            else: # just order the points roughly
                self.examples.append((points, np.array([i for i in range(len(points))])))

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        points, labels = self.examples[idx]
        return torch.FloatTensor(points), torch.LongTensor(labels)
    
    def get_normalization_params(self):
        """Return normalization parameters for use in evaluation"""
        return {
            'min_x': self.min_x,
            'max_x': self.max_x,
            'min_y': self.min_y,
            'max_y': self.max_y
        }