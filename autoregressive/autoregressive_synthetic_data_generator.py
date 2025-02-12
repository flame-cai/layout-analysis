import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from dataclasses import dataclass
from typing import List, Tuple
import math
from scipy.spatial import Delaunay

SEARCH_WINDOW = 50

MAX_CLASSES = 15
MAX_BLOCKS = 1
LINE_SHORT_PROBABILITY = 0.3
CHARACTER_SPACING_VARIANCE = 0.1
CHAR_Y_VARIANCE = 0.5
LINE_Y_VAR = 0.15
MAX_CURVE = 13

curve_modes = ['monotonic_up', 'monotonic_down', 'arch_up', 'arch_down', 'no_arch']

@dataclass
class Point:
    x: int
    y: int

class Line:
    def __init__(self, start_x: int, start_y: int, width: int, chars_count: int, 
                 curve_mode: str, curve_scale: float = 0.0, alignment: str = 'left'):
        self.start_x = start_x
        self.start_y = start_y
        self.width = width
        self.chars_count = chars_count
        self.alignment = alignment
        self.points: List[Point] = []
        self.base_spacing = self.width // (self.chars_count - 1)
        self.curve_scale = curve_scale
        self.curve_mode = curve_mode
        
    def generate_points(self) -> List[Point]:
        full_chars_count = self.chars_count
        spacings = np.maximum(1, (self.base_spacing + 
            np.random.normal(0, self.base_spacing * CHARACTER_SPACING_VARIANCE, 
                           full_chars_count - 1)).astype(int))
        x_offsets = np.concatenate(([0], np.cumsum(spacings)))
        x_positions = self.start_x + x_offsets

        if self.curve_mode == 'monotonic_up':
            start_angle, end_angle = -math.pi / 6, math.pi / 6
        elif self.curve_mode == 'monotonic_down':
            start_angle, end_angle = math.pi / 6, -math.pi / 6
        elif self.curve_mode == 'arch_up':
            start_angle, end_angle = 0, math.pi
        elif self.curve_mode == 'arch_down':
            start_angle, end_angle = -math.pi, 0
        else:
            start_angle, end_angle = -math.pi / 6, math.pi / 6

        angles = np.linspace(start_angle, end_angle, full_chars_count)
        y_offsets = np.sin(angles) * self.curve_scale
        y_positions = self.start_y + y_offsets
        noise = np.random.uniform(-CHAR_Y_VARIANCE, CHAR_Y_VARIANCE, size=full_chars_count)
        y_positions = y_positions + noise

        full_line_width = x_positions[-1] - x_positions[0]
        if self.alignment == 'center':
            offset = (self.width - full_line_width) // 2
        elif self.alignment == 'right':
            offset = self.width - full_line_width
        else:
            offset = 0

        x_positions = x_positions + offset

        if random.random() < LINE_SHORT_PROBABILITY:
            new_count = int(2 + full_chars_count * random.random())
            if self.alignment == 'right':
                x_positions = x_positions[-new_count:]
                y_positions = y_positions[-new_count:]
            elif self.alignment == 'center':
                start_index = (full_chars_count - new_count) // 2
                x_positions = x_positions[start_index:start_index + new_count]
                y_positions = y_positions[start_index:start_index + new_count]
            else:
                x_positions = x_positions[:new_count]
                y_positions = y_positions[:new_count]

        points = [Point(int(x), int(y)) for x, y in zip(x_positions, y_positions)]
        self.points = points
        return points

class SyntheticDataset(Dataset):
    def __init__(self, num_samples: int = 1000, max_points: int = 1800, 
                 normalize: bool = True, split: str = 'train'):
        self.num_samples = num_samples
        self.max_points = max_points
        self.normalize = normalize
        self.split = split
        self.rng = np.random.RandomState()  # For thread-safe random generation
        
        # Set different dimensions based on split
        # This creates some variability while keeping dimensions reasonable
        if split == 'train':
            self.width = 1300
            self.height = 500
        elif split == 'val':
            # Validation set: slightly different dimensions
            self.width = 1250
            self.height = 550
        else:  # test
            # Test set: more variation to test generalization
            # Randomly choose between different page formats for each sample
            self.base_formats = [
                (1100, 600),   # Wider format
                (1400, 450),   # Very wide format
                (1000, 700),   # More square format
                (1300, 500),   # Original format
            ]
            # Initial values, will be updated per sample
            self.width, self.height = self.base_formats[0]
        
        # Initialize normalization parameters
        self.min_x = 0
        self.max_x = self.width
        self.min_y = 0
        self.max_y = self.height
        
    def get_page_dimensions(self):
        """Get dimensions for current page, with variation for test set"""
        if self.split == 'test':
            # For test set, randomly choose a format and add some noise
            base_width, base_height = random.choice(self.base_formats)
            # Add up to ±10% variation
            width = int(base_width * random.uniform(0.9, 1.1))
            height = int(base_height * random.uniform(0.9, 1.1))
            return width, height
        return self.width, self.height
    
    def generate_page(self) -> Tuple[np.ndarray, np.ndarray]:
        # Get dimensions for this specific page
        width, height = self.get_page_dimensions()
        
        points = []
        labels = []
        num_blocks = self.rng.randint(1, MAX_BLOCKS + 1)
        
        for _ in range(num_blocks):
            block_width = self.rng.randint(300, min(1200, width - 100))
            block_height = self.rng.randint(300, min(450, height - 50))
            x = self.rng.randint(0, width - block_width)
            y = self.rng.randint(0, height - block_height)
            
            num_lines = self.rng.randint(4, MAX_CLASSES)
            chars_per_line = self.rng.randint(15, 35)
            line_height = self.rng.randint(8, 25)
            
            for line_idx in range(num_lines):
                y_position = y + line_idx * line_height
                y_position += line_height * self.rng.uniform(-LINE_Y_VAR, LINE_Y_VAR)
                
                line = Line(
                    start_x=x,
                    start_y=int(y_position),
                    width=block_width,
                    chars_count=chars_per_line,
                    alignment=random.choice(['left', 'center', 'right']),
                    curve_mode=random.choice(curve_modes),
                    curve_scale=self.rng.randint(0, MAX_CURVE)
                )
                
                line_points = line.generate_points()
                for point in line_points:
                    points.append([point.x, point.y])
                    labels.append(line_idx + 1)
        
        points = np.array(points)
        labels = np.array(labels)
        
        if len(points) > 0:
            sorted_indices = np.argsort(points[:, 1])
            points = points[sorted_indices]
            labels = labels[sorted_indices]
            
            if len(points) < self.max_points:
                pad_length = self.max_points - len(points)
                points = np.pad(points, ((0, pad_length), (0, 0)), mode='constant')
                labels = np.pad(labels, (0, pad_length), mode='constant', constant_values=0)
            
            if self.normalize:
                points = points.astype(float)
                points[:, 0] = points[:, 0] / width
                points[:, 1] = 1 - (points[:, 1] / height)
        
        return points, labels
    
    def get_normalization_params(self):
        """Return normalization parameters for use in evaluation."""
        if self.split == 'test':
            # For test set, return the range of possible dimensions
            widths = [w for w, h in self.base_formats]
            heights = [h for w, h in self.base_formats]
            return {
                'min_x': 0,
                'max_x': max(widths) * 1.1,  # Account for 10% variation
                'min_y': 0,
                'max_y': max(heights) * 1.1
            }
        return {
            'min_x': 0,
            'max_x': self.width,
            'min_y': 0,
            'max_y': self.height
        }
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        points, labels = self.generate_page()
        # [Feature computation methods remain the same]
        neighbor_feats = self._add_neighbor_features(points)
        delaunay_feats = self._add_delaunay_features(points)
        anchor_feats = self._add_anchor_features(points)
        features = np.concatenate([neighbor_feats, delaunay_feats, anchor_feats], axis=1)
        return torch.FloatTensor(features), torch.LongTensor(labels)
    
    # Include the feature computation methods from the original PointDataset
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
        window = SEARCH_WINDOW   # consider candidates in the ±50 index window
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
        return degrees.reshape(n_points, 1)/10

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

# Usage example:
def create_data_loaders(num_samples=1000, batch_size=32):
    # Create datasets with different splits
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    test_size = num_samples - train_size - val_size
    
    train_dataset = SyntheticDataset(num_samples=train_size, split='train')
    val_dataset = SyntheticDataset(num_samples=val_size, split='val')
    test_dataset = SyntheticDataset(num_samples=test_size, split='test')
    
    # Get normalization parameters
    train_norm_params = train_dataset.get_normalization_params()
    val_norm_params = val_dataset.get_normalization_params()
    test_norm_params = test_dataset.get_normalization_params()
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, 
                           num_workers=4, pin_memory=True)
    
    return (train_loader, val_loader, test_loader), \
           (train_norm_params, val_norm_params, test_norm_params)