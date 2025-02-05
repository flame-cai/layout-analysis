import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random

# --- Our original PointDataset with augmented points (precomputing relative angles) ---
MAX_NO_POINTS = 2000

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
        
        c = 0
        cc = len(split_files)
        # Process each file, update normalization parameters if needed, and augment data
        for file in split_files:
            print(f'Processing file {c}/{cc}')
            points_file = os.path.join(data_dir, f"{file}__points.txt")
            points = np.loadtxt(points_file)
            if len(points.shape) != 2:
                continue

            # Update normalization statistics (if enabled)
            if normalize:
                self.min_x = min(self.min_x, points[:, 0].min())
                self.min_y = min(self.min_y, points[:, 1].min())
                self.max_x = max(self.max_x, points[:, 0].max())
                self.max_y = max(self.max_y, points[:, 1].max())
            
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
                # Flip y axis if needed
                points[:, 1] = 1 - ((points[:, 1] - self.min_y) / (self.max_y - self.min_y))
            
            # Shuffle points (and labels) together
            indices = list(range(len(points)))
            random.shuffle(indices)
            points = points[indices]
            if self.labels_mode:
                labels = labels[indices]
            
            # Convert to torch tensor for vectorized angle computation
            points_tensor = torch.tensor(points, dtype=torch.float32)
            angles_tensor = self.compute_relative_angles_tensor(points_tensor, k=4)
            angles = angles_tensor.numpy()  # shape: (max_points, 4)
            
            # Concatenate original 2D points with the 4 relative angles to form 6 features per point
            points_augmented = np.concatenate([points, angles], axis=1)
            
            if self.labels_mode:
                self.examples.append((points_augmented, labels))
            else:
                self.examples.append((points_augmented, np.array(list(range(len(points_augmented))))))
            c += 1

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
        Computes the relative angles for each point to its k nearest neighbors using vectorized operations.

        Args:
            points (torch.Tensor): A tensor of shape (N, 2) containing the 2D points.
            k (int): The number of nearest neighbors to compute angles for.

        Returns:
            torch.Tensor: A tensor of shape (N, k) where each row contains the angles (in radians)
                          from the point to each of its k nearest neighbors.
        """
        N = points.size(0)
        # Compute pairwise differences: shape (N, N, 2)
        diff = points.unsqueeze(1) - points.unsqueeze(0)  # (N, N, 2)
        # Compute squared Euclidean distances: shape (N, N)
        dist_sq = torch.sum(diff ** 2, dim=-1)
        
        # Exclude self-distance by setting diagonal to infinity.
        # If your PyTorch version supports it, you can use fill_diagonal_
        # Otherwise, create a mask.
        diag_indices = torch.arange(N, device=points.device)
        dist_sq[diag_indices, diag_indices] = float('inf')
        
        # Get the indices of k nearest neighbors for each point: shape (N, k)
        _, indices = torch.topk(dist_sq, k=k, largest=False)
        
        # Use advanced indexing to gather neighbor coordinates in one go.
        # neighbor_points shape: (N, k, 2)
        neighbor_points = points[indices]
        
        # Compute differences: subtract each point from its k neighbors (broadcasting over k)
        # differences shape: (N, k, 2)
        differences = neighbor_points - points.unsqueeze(1)
        
        # Compute angles: use arctan2 for vectorized computation over the last dimension.
        # angles shape: (N, k)
        angles = torch.atan2(differences[..., 1], differences[..., 0])
        return angles
    

# --- Precompute and Save the Augmented Dataset ---
def precompute_dataset(original_data_dir, split_files, precomputed_dir,
                       max_points=MAX_NO_POINTS, normalize=True, labels_mode=True):
    # Create the output directory if it doesn't exist
    os.makedirs(precomputed_dir, exist_ok=True)
    
    # Instantiate the dataset (this will precompute the augmented points)
    dataset = PointDataset(original_data_dir, split_files, max_points=max_points,
                           normalize=normalize, labels_mode=labels_mode)
    
    # Save each sample as a separate npz file
    for idx in range(len(dataset)):
        points, labels = dataset[idx]
        # Convert to numpy arrays (if they aren't already)
        points_np = points.numpy()
        labels_np = labels.numpy()
        
        # Save each sample; adjust the naming convention as needed
        sample_filename = f"sample_{idx:04d}.npz"
        save_path = os.path.join(precomputed_dir, sample_filename)
        np.savez_compressed(save_path, points=points_np, labels=labels_np)
        print(f"Saved {save_path}")

    norm_params = {
        'min_x': dataset.min_x,
        'max_x': dataset.max_x,
        'min_y': dataset.min_y,
        'max_y': dataset.max_y,
    }
    np.savez_compressed(os.path.join(precomputed_dir, 'norm_params.npz'), **norm_params)

class PrecomputedPointDataset(Dataset):
    def __init__(self, files, norm_params_path=None):
        """
        Args:
            files (list): List of file paths to the precomputed npz files.
            norm_params_path (str): Path to the normalization parameters file.
        """
        self.files = files
        self.norm_params = None
        if norm_params_path is not None and os.path.exists(norm_params_path):
            params = np.load(norm_params_path)
            self.norm_params = {
                'min_x': float(params['min_x']),
                'max_x': float(params['max_x']),
                'min_y': float(params['min_y']),
                'max_y': float(params['max_y']),
            }
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        sample = np.load(self.files[idx])
        points = torch.tensor(sample['points'], dtype=torch.float32)
        labels = torch.tensor(sample['labels'], dtype=torch.long)
        return points, labels
    
    def get_normalization_params(self):
        if self.norm_params is None:
            raise ValueError("Normalization parameters were not provided.")
        return self.norm_params


if __name__ == '__main__':
    # Example paths and file list; adjust these to match your data
    ORIGINAL_DATA_DIR = '/mnt/cai-data/manuscript-annotation-tool/synthetic-data'
    PRECOMPUTED_DIR = '/mnt/cai-data/manuscript-annotation-tool/synthetic-dataset-precomputed'
    # List of file identifiers (without extensions) for your training samples
    # For example, if you have files named sample1__points.txt and sample1__labels.txt, then:
    #train_files = ['sample1', 'sample2', 'sample3']  # Update this list accordingly
    all_files = [f.split('__')[0] for f in os.listdir(ORIGINAL_DATA_DIR) if f.endswith('__points.txt')]
    
    precompute_dataset(ORIGINAL_DATA_DIR, all_files, PRECOMPUTED_DIR,
                       max_points=MAX_NO_POINTS, normalize=True, labels_mode=True)
