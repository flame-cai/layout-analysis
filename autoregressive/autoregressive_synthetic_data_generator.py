import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from dataclasses import dataclass
from typing import List, Tuple
import math
from scipy.spatial import Delaunay

# Constants
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
        # Compute randomized spacings (with a minimum of 1)
        spacings = np.maximum(
            1,
            (self.base_spacing + np.random.normal(
                0, self.base_spacing * CHARACTER_SPACING_VARIANCE, full_chars_count - 1
            )).astype(int)
        )
        x_offsets = np.concatenate(([0], np.cumsum(spacings)))
        x_positions = self.start_x + x_offsets

        # Choose curve parameters
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
        # Add uniform noise
        y_positions += np.random.uniform(-CHAR_Y_VARIANCE, CHAR_Y_VARIANCE, size=full_chars_count)

        # Adjust positions based on alignment
        full_line_width = x_positions[-1] - x_positions[0]
        if self.alignment == 'center':
            offset = (self.width - full_line_width) // 2
        elif self.alignment == 'right':
            offset = self.width - full_line_width
        else:
            offset = 0

        x_positions += offset

        # Possibly shorten the line with a given probability
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
        self.rng = np.random.RandomState()  # thread-safe random generation

        # Precompute the candidate offsets used in neighbor feature extraction.
        self._candidate_offsets = np.arange(-SEARCH_WINDOW, SEARCH_WINDOW + 1)

        # Set page dimensions based on split
        if split == 'train':
            self.width = 1300
            self.height = 500
        elif split == 'val':
            self.width = 1250
            self.height = 550
        else:  # test split: choose from multiple formats per sample.
            self.base_formats = [
                (1100, 600),   # Wider format
                (1400, 450),   # Very wide format
                (1000, 700),   # More square format
                (1300, 500),   # Original format
            ]
            self.width, self.height = self.base_formats[0]

        self.min_x = 0
        self.max_x = self.width
        self.min_y = 0
        self.max_y = self.height

    def get_page_dimensions(self) -> Tuple[int, int]:
        if self.split == 'test':
            base_width, base_height = random.choice(self.base_formats)
            # Add up to ±10% variation
            width = int(base_width * random.uniform(0.9, 1.1))
            height = int(base_height * random.uniform(0.9, 1.1))
            return width, height
        return self.width, self.height

    def generate_page(self) -> Tuple[np.ndarray, np.ndarray]:
        width, height = self.get_page_dimensions()
        points_list = []
        labels_list = []
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
                # Calculate y with jitter
                y_position = y + line_idx * line_height + line_height * self.rng.uniform(-LINE_Y_VAR, LINE_Y_VAR)
                line = Line(
                    start_x=x,
                    start_y=int(y_position),
                    width=block_width,
                    chars_count=chars_per_line,
                    alignment=self.rng.choice(['left', 'center', 'right']),
                    curve_mode=self.rng.choice(curve_modes),
                    curve_scale=self.rng.randint(0, MAX_CURVE)
                )
                # Use list comprehensions to quickly extend the lists
                line_points = line.generate_points()
                points_list.extend([[p.x, p.y] for p in line_points])
                labels_list.extend([line_idx + 1] * len(line_points))

        points = np.array(points_list)
        labels = np.array(labels_list)

        if points.size > 0:
            # Sort points by y-coordinate (which helps with the neighbor feature computation)
            sorted_indices = np.argsort(points[:, 1])
            points = points[sorted_indices]
            labels = labels[sorted_indices]

            # Pad to max_points
            if len(points) < self.max_points:
                pad_length = self.max_points - len(points)
                points = np.pad(points, ((0, pad_length), (0, 0)), mode='constant')
                labels = np.pad(labels, (0, pad_length), mode='constant', constant_values=0)

            # Normalize coordinates if required
            if self.normalize:
                points = points.astype(np.float32)
                points[:, 0] /= width
                points[:, 1] = 1 - (points[:, 1] / height)

        return points, labels

    def get_normalization_params(self):
        if self.split == 'test':
            widths = [w for w, h in self.base_formats]
            heights = [h for w, h in self.base_formats]
            return {
                'min_x': 0,
                'max_x': max(widths) * 1.1,
                'min_y': 0,
                'max_y': max(heights) * 1.1
            }
        return {
            'min_x': 0,
            'max_x': self.width,
            'min_y': 0,
            'max_y': self.height
        }

    def _add_neighbor_features(self, points: np.ndarray) -> np.ndarray:
        """
        For each point, return a 4D feature vector containing:
          - The (x,y) coordinates, and
          - The summed (dx, dy) offsets (over up to 4 nearest neighbors that are roughly horizontal)
        """
        n_points = points.shape[0]
        features = np.zeros((n_points, 4), dtype=points.dtype)
        features[:, :2] = points  # Original coordinates

        k = 4
        window = SEARCH_WINDOW
        candidate_count = 2 * window + 1

        # Use the precomputed candidate offsets.
        candidate_offsets = self._candidate_offsets  # shape: (candidate_count,)
        all_indices = np.arange(n_points)[:, None] + candidate_offsets[None, :]
        all_indices = np.clip(all_indices, 0, n_points - 1)

        candidate_points = points[all_indices]  # (n_points, candidate_count, 2)
        diffs = candidate_points - points[:, None, :]  # (n_points, candidate_count, 2)
        # Use Manhattan distance (can be replaced with Euclidean if desired)
        dists = np.sum(np.abs(diffs), axis=2)
        # Exclude the self-point by setting its distance to infinity.
        mask = (all_indices == np.arange(n_points)[:, None])
        dists[mask] = np.inf

        # Get indices of the k nearest candidates
        neighbor_pos = np.argpartition(dists, kth=k, axis=1)[:, :k]
        i_idx = np.arange(n_points)[:, None]
        selected_diffs = diffs[i_idx, neighbor_pos, :]  # (n_points, k, 2)

        # Keep only neighbors that are approximately horizontal (|dy| < 0.3 * |dx|)
        threshold_ratio = 0.3
        horizontal_mask = np.abs(selected_diffs[..., 1]) < threshold_ratio * np.abs(selected_diffs[..., 0])
        horizontal_mask = horizontal_mask[..., None]  # For broadcasting
        filtered_diffs = selected_diffs * horizontal_mask
        summed_diffs = np.sum(filtered_diffs, axis=1)
        features[:, 2:] = summed_diffs

        return features

    def _add_delaunay_features(self, points: np.ndarray) -> np.ndarray:
        """
        Compute a connectivity feature based on Delaunay triangulation:
        the degree of each point is computed via a fast vectorized bincount.
        """
        n_points = points.shape[0]
        try:
            tri = Delaunay(points)
            # Vectorized counting of how many triangles each point is part of.
            counts = np.bincount(tri.simplices.ravel(), minlength=n_points)
            degrees = counts.astype(np.float32) / 10  # scale factor
        except Exception:
            degrees = np.zeros(n_points, dtype=np.float32)
        return degrees.reshape(n_points, 1)

    def _add_anchor_features(self, points: np.ndarray) -> np.ndarray:
        """
        For each point, compute the offset to four fixed anchors:
          - bottom-right, top-left, top-right, and center.
        The result is flattened to an 8-dimensional feature.
        """
        n_points = points.shape[0]
        anchors = np.array([
            [1.0, 0.0],   # bottom-right
            [0.0, 1.0],   # top-left
            [1.0, 1.0],   # top-right
            [0.5, 0.5]    # center
        ], dtype=points.dtype)
        offsets = points[:, None, :] - anchors[None, :, :]
        return offsets.reshape(n_points, -1)

    def __getitem__(self, idx):
        points, labels = self.generate_page()
        neighbor_feats = self._add_neighbor_features(points)
        delaunay_feats = self._add_delaunay_features(points)
        anchor_feats = self._add_anchor_features(points)
        # Concatenate all features along the last axis.
        features = np.concatenate([neighbor_feats, delaunay_feats, anchor_feats], axis=1)
        # Use torch.as_tensor to avoid an extra copy.
        return torch.as_tensor(features, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.num_samples

# DataLoader factory function
def create_data_loaders(num_samples=1000, batch_size=32):
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    test_size = num_samples - train_size - val_size

    train_dataset = SyntheticDataset(num_samples=train_size, split='train')
    val_dataset = SyntheticDataset(num_samples=val_size, split='val')
    test_dataset = SyntheticDataset(num_samples=test_size, split='test')

    train_norm_params = train_dataset.get_normalization_params()
    val_norm_params = val_dataset.get_normalization_params()
    test_norm_params = test_dataset.get_normalization_params()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1,
        num_workers=4, pin_memory=True
    )

    return (train_loader, val_loader, test_loader), (train_norm_params, val_norm_params, test_norm_params)

# Example usage:
if __name__ == '__main__':
    (train_loader, val_loader, test_loader), norm_params = create_data_loaders()
    for features, labels in train_loader:
        print("Features shape:", features.shape)  # Expected: (batch_size, max_points, 13)
        print("Labels shape:", labels.shape)
        break
