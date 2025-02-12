import torch
from torch.utils.data import Dataset
import numpy as np
import os
from scipy.spatial import Delaunay
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from functools import lru_cache

@dataclass
class NormalizationParams:
    min_x: float
    max_x: float
    min_y: float
    max_y: float

class PointDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split_files: List[str],
        max_points: int = 1800,
        normalize: bool = True,
        labels_mode: bool = True,
        search_window: int = 50,
        neighbor_threshold_ratio: float = 0.3
    ):
        self.data_dir = data_dir
        self.max_points = max_points
        self.normalize = normalize
        self.labels_mode = labels_mode
        self.search_window = search_window
        self.neighbor_threshold_ratio = neighbor_threshold_ratio
        
        # Pre-compute anchor points for feature generation
        self.anchors = torch.tensor([
            [1.0, 0.0],   # bottom-right
            [0.0, 1.0],   # top-left
            [1.0, 1.0],   # top-right
            [0.5, 0.5]    # center
        ], dtype=torch.float32)
        
        # Process all data at initialization
        self.examples, self.norm_params = self._process_data(split_files)

    def _process_data(self, split_files: List[str]) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], NormalizationParams]:
        """Process all data files and compute normalization parameters."""
        examples = []
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        
        # Process each file
        for file in split_files:
            points = np.loadtxt(os.path.join(self.data_dir, f"{file}__points.txt"))
            if len(points.shape) != 2:
                continue
                
            if self.normalize:
                min_x = min(min_x, points[:, 0].min())
                min_y = min(min_y, points[:, 1].min())
                max_x = max(max_x, points[:, 0].max())
                max_y = max(max_y, points[:, 1].max())
            
            if self.labels_mode:
                labels = np.loadtxt(os.path.join(self.data_dir, f"{file}__labels.txt")).astype(np.int64)
            else:
                labels = np.arange(len(points))
            
            # Handle padding
            if len(points) < self.max_points:
                points = np.pad(points, ((0, self.max_points - len(points)), (0, 0)), mode='constant')
                if self.labels_mode:
                    labels = np.pad(labels, (0, self.max_points - len(labels)), mode='constant', constant_values=-1)
            
            examples.append((points, labels))
        
        # Normalize all points if requested
        if self.normalize:
            x_scale = max_x - min_x
            y_scale = max_y - min_y
            for i, (points, _) in enumerate(examples):
                points[:, 0] = (points[:, 0] - min_x) / x_scale
                points[:, 1] = 1 - (points[:, 1] - min_y) / y_scale
                examples[i] = (points.astype(np.float32), examples[i][1])
        
        norm_params = NormalizationParams(min_x, max_x, min_y, max_y)
        return examples, norm_params

    @staticmethod
    @lru_cache(maxsize=128)
    def _compute_neighbor_indices(n_points: int, window: int) -> Tuple[np.ndarray, np.ndarray]:
        """Pre-compute and cache neighbor indices for performance."""
        candidate_offsets = np.arange(-window, window + 1)
        all_indices = np.arange(n_points)[:, None] + candidate_offsets[None, :]
        return np.clip(all_indices, 0, n_points - 1), (all_indices == np.arange(n_points)[:, None])

    def _compute_neighbor_features(self, points: np.ndarray) -> np.ndarray:
        """Optimized neighbor feature computation."""
        n_points = points.shape[0]
        features = np.zeros((n_points, 4), dtype=np.float32)
        features[:, :2] = points
        
        # Get cached indices
        all_indices, mask = self._compute_neighbor_indices(n_points, self.search_window)
        
        # Compute differences and distances
        candidate_points = points[all_indices]
        diffs = candidate_points - points[:, None, :]
        dists = np.sum(np.abs(diffs), axis=2)
        dists[mask] = np.inf
        
        # Select nearest neighbors
        k = 4
        neighbor_pos = np.argpartition(dists, k, axis=1)[:, :k]
        selected_diffs = diffs[np.arange(n_points)[:, None], neighbor_pos]
        
        # Apply horizontal filter
        horizontal_mask = (np.abs(selected_diffs[..., 1]) < 
                         self.neighbor_threshold_ratio * np.abs(selected_diffs[..., 0]))[..., None]
        features[:, 2:] = np.sum(selected_diffs * horizontal_mask, axis=1)
        
        return features

    @staticmethod
    @lru_cache(maxsize=1024)
    def _compute_delaunay_features(points_tuple: Tuple[Tuple[float, float], ...]) -> np.ndarray:
        """Cached Delaunay feature computation."""
        points = np.array(points_tuple)
        n_points = points.shape[0]
        degrees = np.zeros(n_points, dtype=np.float32)
        
        try:
            tri = Delaunay(points)
            np.add.at(degrees, tri.simplices.ravel(), 1)
        except Exception:
            pass
            
        return degrees.reshape(-1, 1) / 10

    def _compute_anchor_features(self, points: torch.Tensor) -> torch.Tensor:
        """Vectorized anchor feature computation."""
        points_tensor = torch.from_numpy(points) if isinstance(points, np.ndarray) else points
        offsets = points_tensor.unsqueeze(1) - self.anchors.to(points_tensor.device)
        return offsets.reshape(points.shape[0], -1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        points, labels = self.examples[idx]
        
        # Compute all features
        neighbor_feats = self._compute_neighbor_features(points)
        delaunay_feats = self._compute_delaunay_features(tuple(map(tuple, points)))
        anchor_feats = self._compute_anchor_features(points).numpy()
        
        # Combine features
        features = np.concatenate([neighbor_feats, delaunay_feats, anchor_feats], axis=1)
        return torch.from_numpy(features), torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.examples)

    def get_normalization_params(self) -> Dict[str, float]:
        """Return normalization parameters."""
        return {
            'min_x': self.norm_params.min_x,
            'max_x': self.norm_params.max_x,
            'min_y': self.norm_params.min_y,
            'max_y': self.norm_params.max_y
        }