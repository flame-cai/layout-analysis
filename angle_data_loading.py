import torch
from torch.utils.data import Dataset
import numpy as np
import os
from scipy.spatial import Delaunay

MAX_NO_POINTS = 1800

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
                
        # Second pass: load and process data
        for file in split_files:
            points_file = os.path.join(data_dir, f"{file}__points.txt")
            points = np.loadtxt(points_file)
            if len(points.shape) != 2:
                continue
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
                # For y, flip vertically so that y=0 is at the top
                points[:, 1] = 1 - (points[:, 1] - self.min_y) / (self.max_y - self.min_y)
            
            if self.labels_mode:
                self.examples.append((points, labels))
            else:
                self.examples.append((points, np.arange(len(points))))
    
    def _add_neighbor_features(self, points: np.ndarray) -> np.ndarray:
        """
        Compute features for each point by concatenating:
          - the 2D point coordinates (columns 0-1), and
          - the raw offset vectors (dx, dy) to each of its 4 nearest neighbors,
            selected from only the ±50 points around it (columns 2-9).

        Assumes points are arranged in an order such that indices correlate with spatial proximity.
        Returns an array of shape (n_points, 10).
        """
        n_points = points.shape[0]
        features = np.zeros((n_points, 4), dtype=points.dtype)
        features[:, :2] = points  # columns 0-1: original (x, y)

        k = 4         # number of neighbors to select
        window = 50   # consider candidates in the ±50 index window
        candidate_count = 2 * window + 1  # total candidates per point

        # For each point, construct candidate indices: i + [-50, -49, ..., 0, ..., 50]
        candidate_offsets = np.arange(-window, window + 1)  # shape: (101,)
        all_indices = np.arange(n_points)[:, None] + candidate_offsets[None, :]
        all_indices = np.clip(all_indices, 0, n_points - 1)  # shape: (n_points, candidate_count)

        # Gather candidate points and compute differences relative to each point.
        candidate_points = points[all_indices]  # shape: (n_points, candidate_count, 2)
        diffs = candidate_points - points[:, None, :]  # shape: (n_points, candidate_count, 2)

        # Compute Euclidean distances for each candidate.
        #dists = np.linalg.norm(diffs, axis=2)  # shape: (n_points, candidate_count)
        dists = np.sum(np.abs(diffs), axis=2)  # Manhattan distances

        # Exclude self by setting distances where the candidate index equals the point index.
        mask = (all_indices == np.arange(n_points)[:, None])
        dists[mask] = np.inf

        # For each point, choose the indices (within the candidate window) of the k smallest distances.
        neighbor_pos = np.argpartition(dists, kth=k, axis=1)[:, :k]  # shape: (n_points, k)

        # Use advanced indexing to select the corresponding differences.
        i_idx = np.arange(n_points)[:, None]
        selected_diffs = diffs[i_idx, neighbor_pos, :]  # shape: (n_points, k, 2)

       # Filter: Only add those differences that are approximately horizontal.
        # Define "approximately horizontal" as |dy| < threshold_ratio * |dx|.
        threshold_ratio = 0.3
        horizontal_mask = np.abs(selected_diffs[..., 1]) < threshold_ratio * np.abs(selected_diffs[..., 0])
        # Expand mask dimensions for proper broadcasting.
        horizontal_mask = horizontal_mask[..., None]  # shape: (n_points, k, 1)
        # Zero-out those differences that do not pass the horizontal filter.
        filtered_diffs = selected_diffs * horizontal_mask  # shape: (n_points, k, 2)

        # Sum the filtered differences over the k neighbors to produce a single 2D offset.
        summed_diffs = np.sum(filtered_diffs, axis=1)  # shape: (n_points, 2)

        # Place the summed offsets in columns 2-3.
        features[:, 2:] = summed_diffs

        return features

    def _add_delaunay_features(self, points: np.ndarray) -> np.ndarray:
        """
        Compute a simple graph-based feature using Delaunay triangulation.
        Here, we compute the degree (i.e. number of incident triangles)
        for each point as a measure of local connectivity.
        Returns an array of shape (n_points, 1).
        """
        n_points = points.shape[0]
        degrees = np.zeros(n_points, dtype=np.float32)
        try:
            tri = Delaunay(points)
            # For each triangle (simplex), increment the degree count of its vertices.
            for simplex in tri.simplices:
                for vertex in simplex:
                    degrees[vertex] += 1
        except Exception as e:
            # If Delaunay triangulation fails, simply return zeros.
            degrees = np.zeros(n_points, dtype=np.float32)
        return degrees.reshape(n_points, 1)

    def _add_anchor_features(self, points: np.ndarray) -> np.ndarray:
        """
        Compute anchor-based relative features for each point.
        In this example, five anchors are used: the four corners and the center of the page.
        For each point, compute the offset (dx, dy) to each anchor.
        Returns an array of shape (n_points, 10) [5 anchors × 2 coordinates each].
        """
        n_points = points.shape[0]
        # Define five anchors assuming the page is normalized between 0 and 1.
        anchors = np.array([
            [1.0, 0.0],   # bottom-right
            [0.0, 1.0],   # top-left
            [1.0, 1.0],   # top-right
            [0.5, 0.5]    # center
        ], dtype=points.dtype)
        # Compute offsets for each point: result shape will be (n_points,4, 2)
        offsets = points[:, None, :] - anchors[None, :, :]
        # Flatten the offsets to shape (n_points, 8)
        return offsets.reshape(n_points, -1)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        points, labels = self.examples[idx]
        # Compute neighbor features (2-dim)
        neighbor_feats = self._add_neighbor_features(points)
        # Compute Delaunay (graph-based) features (1-dim)
        delaunay_feats = self._add_delaunay_features(points)
        # Compute anchor-based features (8-dim)
        anchor_feats = self._add_anchor_features(points)
        # Combine all features: 13
        features = np.concatenate([neighbor_feats, delaunay_feats, anchor_feats], axis=1)
        return torch.FloatTensor(features), torch.LongTensor(labels)
    
    def get_normalization_params(self):
        """Return normalization parameters for use in evaluation."""
        return {
            'min_x': self.min_x,
            'max_x': self.max_x,
            'min_y': self.min_y,
            'max_y': self.max_y
        }
