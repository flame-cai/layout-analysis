import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os

MAX_NO_POINTS = 1400

class PointDataset(Dataset):
    def __init__(self, data_dir, split_files, max_points=MAX_NO_POINTS, normalize=True, labels_mode=True):
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
        
        # First pass: Find min/max coordinates if normalizing
        if normalize:
            for file in split_files:
                points_file = os.path.join(data_dir, f"{file}__points.txt")
                points = np.loadtxt(points_file)
                if len(points.shape) != 2:
                    continue
                self.min_x = min(self.min_x, points[:, 0].min())
                self.min_y = min(self.min_y, points[:, 1].min())
                self.max_x = max(self.max_x, points[:, 0].max())
                self.max_y = max(self.max_y, points[:, 1].max())
        
        # Second pass: Load, process, and augment data
        for file in split_files:
            points_file = os.path.join(data_dir, f"{file}__points.txt")
            points = np.loadtxt(points_file)
            if len(points.shape) != 2:
                continue
            
            if self.labels_mode:
                labels_file = os.path.join(data_dir, f"{file}__labels.txt")
                labels = np.loadtxt(labels_file).astype(int)
            
            # Pad if necessary
            if len(points) < max_points:
                pad_length = max_points - len(points)
                points = np.pad(points, ((0, pad_length), (0, 0)), mode='constant')
                if self.labels_mode:
                    labels = np.pad(labels, (0, pad_length), mode='constant', constant_values=-1)
            
            # Normalize points if requested
            if normalize:
                points[:, 0] = (points[:, 0] - self.min_x) / (self.max_x - self.min_x)
                # For y, we flip the axis (if needed) to match your desired orientation
                points[:, 1] = 1 - ((points[:, 1] - self.min_y) / (self.max_y - self.min_y))
            
            # Shuffle points (and labels) together
            indices = list(range(len(points)))
            random.shuffle(indices)
            points = points[indices]
            if self.labels_mode:
                labels = labels[indices]
            
            # --- Compute relative angles using vectorized pairwise distance computation ---
            # Convert points to a torch tensor
            points_tensor = torch.tensor(points, dtype=torch.float32)
            # Compute the relative angles for each point (resulting in shape: [max_points, 4])
            angles_tensor = self.compute_relative_angles_tensor(points_tensor, k=4)
            # Convert the angles back to a numpy array for concatenation
            angles = angles_tensor.numpy()
            
            # Concatenate the original 2D points with the 4 relative angles
            # New shape per point: [x, y, angle1, angle2, angle3, angle4]
            points_augmented = np.concatenate([points, angles], axis=1)
            
            # Append the augmented points (and labels) to the dataset
            if self.labels_mode:
                self.examples.append((points_augmented, labels))
            else:
                self.examples.append((points_augmented, np.array(list(range(len(points_augmented))))))
    
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
    
    @staticmethod
    def compute_relative_angles_tensor(points, k=4):
        """
        Computes the relative angles for each point to its k nearest neighbors.
        
        Args:
            points (torch.Tensor): A tensor of shape (N, 2) containing the 2D points.
            k (int): The number of nearest neighbors to compute angles for.
        
        Returns:
            torch.Tensor: A tensor of shape (N, k) where each row contains the angles (in radians)
                          from the point to each of its k nearest neighbors.
        """
        N = points.size(0)
        # Compute pairwise differences: result has shape (N, N, 2)
        diff = points.unsqueeze(1) - points.unsqueeze(0)
        # Compute squared Euclidean distances: shape (N, N)
        dist_sq = torch.sum(diff ** 2, dim=-1)
        # Exclude the distance from a point to itself by setting the diagonal to infinity
        mask = torch.eye(N, dtype=torch.bool, device=points.device)
        dist_sq.masked_fill_(mask, float('inf'))
        # For each point, find the indices of the k smallest distances (i.e., nearest neighbors)
        _, indices = torch.topk(dist_sq, k=k, largest=False)
        
        # Prepare a tensor to hold the angles
        angles = torch.zeros((N, k), dtype=torch.float32, device=points.device)
        
        # For each point, compute the angle (using atan2) to each of its k nearest neighbors
        for i in range(N):
            for j in range(k):
                neighbor_idx = indices[i, j]
                dx = points[neighbor_idx, 0] - points[i, 0]
                dy = points[neighbor_idx, 1] - points[i, 1]
                angles[i, j] = torch.atan2(dy, dx)
        return angles

# Example usage:
if __name__ == '__main__':
    # Assuming you have a directory 'data' and a list of file identifiers
    data_directory = 'data'
    split_files = ['sample1', 'sample2']  # Replace with your actual file identifiers
    dataset = PointDataset(data_directory, split_files, max_points=MAX_NO_POINTS, normalize=True, labels_mode=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Retrieve one batch to see the augmented points
    for points, labels in dataloader:
        print("Points shape:", points.shape)  # Should be [batch, MAX_NO_POINTS, 6]
        print("Labels shape:", labels.shape)
        break
