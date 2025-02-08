import torch
from torch.utils.data import Dataset
import numpy as np
import os

MAX_NO_POINTS = 1800

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
                
        # Second pass: load and process data
        for file in split_files:
            points_file = os.path.join(data_dir, f"{file}__points.txt")
            points = np.loadtxt(points_file)
            if len(points.shape)!=2:
                    continue
            if normalize:
                self.min_x = min(self.min_x, points[:, 0].min())
                self.min_y = min(self.min_y, points[:, 1].min())
                self.max_x = max(self.max_x, points[:, 0].max())
                self.max_y = max(self.max_y, points[:, 1].max())
            if self.labels_mode == True:
                labels_file = os.path.join(data_dir, f"{file}__labels.txt")
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
            
            if self.labels_mode == True:
                self.examples.append((points, labels))
            else: # just order the points roughly
                self.examples.append((points, np.array([i for i in range(len(points))])))

    def _add_neighbor_features(self, points: np.ndarray) -> np.ndarray:
        """
        Compute features for each point by concatenating:
        - the 2D point coordinates (columns 0-1), and
        - the raw offset vectors (dx, dy) to each of its 4 nearest neighbors
            among only the ±50 points around it (columns 2-9).

        Assumes points are arranged in an order such that indices correlate with spatial proximity.
        Returns an array of shape (n_points, 10).
        """
        n_points = points.shape[0]
        features = np.zeros((n_points, 10), dtype=points.dtype)
        features[:, :2] = points

        k = 4         # number of neighbors to select
        window = 50   # look at ±50 points
        candidate_count = 2 * window + 1  # total candidates per point

        # For each point, construct candidate indices: i + [-50, -49, ..., 0, ..., 50]
        candidate_offsets = np.arange(-window, window + 1)  # shape (101,)
        # Broadcast: for each index i in [0, n_points), compute candidate indices.
        # We then clip to ensure indices remain within [0, n_points-1].
        all_indices = np.arange(n_points)[:, None] + candidate_offsets[None, :]
        all_indices = np.clip(all_indices, 0, n_points - 1)  # shape: (n_points, candidate_count)

        # Gather the candidate points (shape: n_points x candidate_count x 2)
        candidate_points = points[all_indices]
        # Compute the differences from each candidate to the current point.
        # Each point’s difference: candidate_points[i] - points[i]
        diffs = candidate_points - points[:, None, :]  # shape: (n_points, candidate_count, 2)

        # Compute Euclidean distances for each candidate.
        dists = np.linalg.norm(diffs, axis=2)  # shape: (n_points, candidate_count)

        # Since the candidate window includes the point itself (when offset==0),
        # we set those distances to infinity to exclude self-matches.
        # Because of clipping at boundaries, self might appear in more than one location.
        # Mark every candidate that equals the current index as invalid.
        mask = (all_indices == np.arange(n_points)[:, None])
        dists[mask] = np.inf

        # For each point, choose the indices (within the candidate window) of the k smallest distances.
        # Using argpartition gives the k smallest values in arbitrary order.
        neighbor_pos = np.argpartition(dists, kth=k, axis=1)[:, :k]  # shape: (n_points, k)

        # Use advanced indexing to select the corresponding differences.
        # For each point i, we pick diffs[i, neighbor_pos[i], :]
        i_idx = np.arange(n_points)[:, None]
        selected_diffs = diffs[i_idx, neighbor_pos, :]  # shape: (n_points, k, 2)

        # Flatten the selected differences (k neighbors × 2 coordinates = 8 features)
        offsets_flat = selected_diffs.reshape(n_points, -1)  # shape: (n_points, 8)

        # Concatenate the original coordinates (columns 0-1) with the offsets (columns 2-9)
        features[:, 2:] = offsets_flat

        return features

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        points, labels = self.examples[idx]
        points = self._add_neighbor_features(points)
        return torch.FloatTensor(points), torch.LongTensor(labels)
    
    def get_normalization_params(self):
        """Return normalization parameters for use in evaluation"""
        return {
            'min_x': self.min_x,
            'max_x': self.max_x,
            'min_y': self.min_y,
            'max_y': self.max_y
        }