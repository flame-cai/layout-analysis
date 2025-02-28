import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor

MAX_CLASSES = 15  # a text block can have max 15 lines
MAX_BLOCKS = 2

LINE_SHORT_PROBABILITY= 0.3
CHARACTER_SPACING_VARIANCE = 0.1

CHAR_Y_VARIANCE = 0.5 #Character level
LINE_Y_VAR = 0.15 #Line level 

curve_modes = ['monotonic_up', 'monotonic_down', 'arch_up', 'arch_down','no_arch']
MAX_CURVE = 13

@dataclass
class Point:
    x: int
    y: int

class Line:
    def __init__(self, start_x: int, start_y: int, width: int, chars_count: int, curve_mode: str, curve_scale: float = 0.0,
                 alignment: str = 'left'):
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
        """Generate character points along the line with a curve, optimized for speed.

        The full curve is generated using the original char_count, then aligned,
        and finally truncated (by slicing) depending on the alignment (left, center, right).
        """
        # Save the original full character count.
        full_chars_count = self.chars_count

        # --- Generate x positions ---
        # Create randomized spacings (vectorized) and compute cumulative x offsets.
        spacings = np.maximum(
            1,
            (self.base_spacing +
            np.random.normal(0, self.base_spacing * CHARACTER_SPACING_VARIANCE, full_chars_count - 1)
            ).astype(int)
        )
        x_offsets = np.concatenate(([0], np.cumsum(spacings)))
        x_positions = self.start_x + x_offsets

        # --- Compute the sine-based y positions ---
        # Determine the start and end angles based on the desired curve mode.
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

        # Generate angles for the full line (vectorized).
        angles = np.linspace(start_angle, end_angle, full_chars_count)
        # Compute y offsets using the sine function.
        y_offsets = np.sin(angles) * self.curve_scale
        y_positions = self.start_y + y_offsets

        # --- Add vertical randomness in a vectorized manner ---
        # Note: np.random.randint's high is exclusive, so add 1 to include CHAR_Y_VARIANCE.
        noise = np.random.uniform(-CHAR_Y_VARIANCE, CHAR_Y_VARIANCE, size=full_chars_count)
        y_positions = y_positions + noise

        # --- Compute alignment offset using the full line width ---
        full_line_width = x_positions[-1] - x_positions[0]
        if self.alignment == 'center':
            offset = (self.width - full_line_width) // 2
        elif self.alignment == 'right':
            offset = self.width - full_line_width
        else:
            offset = 0

        # Apply the offset to the entire x_positions array.
        x_positions = x_positions + offset

        # --- Possibly truncate the line ---
        # Use slicing on the NumPy arrays based on the alignment.
        if random.random() < LINE_SHORT_PROBABILITY:
            new_count = int(2 + full_chars_count * random.random())  # enforce a minimum number of points
            if self.alignment == 'right':
                # Keep the last new_count points.
                x_positions = x_positions[-new_count:]
                y_positions = y_positions[-new_count:]
            elif self.alignment == 'center':
                start_index = (full_chars_count - new_count) // 2
                x_positions = x_positions[start_index:start_index + new_count]
                y_positions = y_positions[start_index:start_index + new_count]
            else:  # left alignment (default)
                x_positions = x_positions[:new_count]
                y_positions = y_positions[:new_count]

        # --- Convert the NumPy arrays into a list of Point objects ---
        points = [Point(int(x), int(y)) for x, y in zip(x_positions, y_positions)]
        self.points = points
        return points
    
class TextBlock:
    def __init__(self, x: int, y: int, width: int, height: int, 
                 lines_count: int, chars_per_line: int, curve_mode: str, curve_scale: str,
                 alignment: str = 'left', allow_half_lines: bool = True):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.lines_count = lines_count
        self.base_chars_per_line = chars_per_line
        self.alignment = alignment
        self.allow_half_lines = allow_half_lines
        self.curve_mode = curve_mode
        self.curve_scale = curve_scale
        # Apply a slight random curve to the line
        self.lines: List[Line] = []

    
        
    def generate_lines(self) -> List[Line]:
        """Generate lines for the text block"""
        line_height = np.random.randint(8,25) #self.height // self.lines_count

        for i in range(self.lines_count):                
            # Vary the y-position for each line slightly
            y_position = self.y + i * line_height
            y_position += line_height*np.random.uniform(-LINE_Y_VAR, LINE_Y_VAR)
   
            line = Line(
                start_x=self.x,
                start_y=y_position,
                width=self.width,
                chars_count=self.base_chars_per_line,
                alignment=self.alignment,
                curve_mode=self.curve_mode,
                curve_scale=self.curve_scale
            )
            line.generate_points()
            self.lines.append(line)
            
        return self.lines

class Page:
    def __init__(self, width: int = 1300, height: int = 500):
        self.width = width
        self.height = height
        self.text_blocks: List[TextBlock] = []
        
        
    def add_text_block(self, block: TextBlock):
        """Add a text block after checking for overlaps using bounding boxes"""
        for existing in self.text_blocks:
            if (block.x < existing.x + existing.width and
                block.x + block.width > existing.x and
                block.y < existing.y + existing.height and
                block.y + block.height > existing.y):
                raise ValueError("Text blocks overlap!")
        self.text_blocks.append(block)
        
    def generate_random_layout(self, num_blocks: int = 8):
        """Generate a random layout with a specified number of text blocks"""
        self.text_blocks = []
        # Create main text blocks
        for _ in range(num_blocks):
            attempts = 0
            while attempts < 100:
                block_width = random.randint(300, 1200)
                block_height = random.randint(300, 450)
                x = random.randint(0, self.width - block_width)
                y = random.randint(0, self.height - block_height)
                
                block = TextBlock(
                    x=x,
                    y=y,
                    width=block_width,
                    height=block_height,
                    lines_count=random.randint(4, MAX_CLASSES),
                    chars_per_line=random.randint(15, 35),
                    alignment=random.choice(['left','center','right']),
                    curve_mode=random.choice(curve_modes),
                    curve_scale=random.randint(0,MAX_CURVE)
                )
        
                try:
                    self.add_text_block(block)
                    block.generate_lines()
                    break
                except ValueError:
                    attempts += 1
                    
        # Optionally add marginalia (small text blocks at the margins)
        if random.random() < 0.3:
            margin_width = 150
            margin_height = 100
            positions = [
                (0, 0),  # top
                (0, self.height - margin_height),  # bottom
                (0, (self.height - margin_height) // 2),  # left
                (self.width - margin_width, (self.height - margin_height) // 2)  # right
            ]
            pos = random.choice(positions)
            marginalia = TextBlock(
                x=pos[0],
                y=pos[1],
                width=margin_width,
                height=margin_height,
                lines_count=random.randint(1, 4),
                chars_per_line=random.randint(2, 5),
                alignment='left',
                curve_mode='no curve',
                curve_scale=0
            )
            try:
                self.add_text_block(marginalia)
                marginalia.generate_lines()
            except ValueError:
                pass  # Skip if overlapping
                
    def get_points_and_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """Aggregate all points and associated labels from text blocks"""
        points = []
        labels = []
        
        for block_idx,block in enumerate(self.text_blocks):
            for line_idx, line in enumerate(block.lines):
                len_line = len(line.points)
                for point_idx,point in enumerate(line.points):
                    # if point_idx == 0: #left 
                    #     label = line_idx + MAX_CLASSES 
                    # elif point_idx == len_line - 1: #right
                    #     label = line_idx + MAX_CLASSES*2
                    # elif point_idx == int((len_line - 1)/2): #center
                    #     label = line_idx + MAX_CLASSES*3#+ 300
                    # else:
                    label = line_idx
                    # 0 padding
                    # 1-15 labels
                    # 16 start token
                    points.append([point.x, point.y])
                    labels.append(label)
                    
        # Convert to NumPy arrays and sort by y-coordinate
        points = np.array(points)
        labels = np.array(labels)
        sorted_indices = np.argsort(points[:, 1]) # 1 -> sort along y
        points = points[sorted_indices]
        labels = labels[sorted_indices]
        
        return points, labels

def process_page(page_idx: int, base_path: str) -> None:
    """
    Generate one page with a random layout and write its points and labels
    to files in the specified base path.
    """
    page = Page()
    # Random number of text blocks per page between 1 and 8
    page.generate_random_layout(num_blocks=random.randint(1, MAX_BLOCKS))
    points, labels = page.get_points_and_labels()
    
    # Use os.path.join for file paths
    points_file = os.path.join(base_path, f"{page_idx}__points.txt")
    labels_file = os.path.join(base_path, f"{page_idx}__labels.txt")
    
    # Save as text files (consider switching to np.save for binary if needed)
    np.savetxt(points_file, points, fmt='%d', delimiter=' ')
    np.savetxt(labels_file, labels, fmt='%d')

def generate_dataset_parallel(num_pages: int = 100, 
                              base_path: str = "/home/kartik/layout-analysis/data/synthetic-data/"):
    """
    Generate multiple pages of synthetic data in parallel.
    Each page is processed in a separate process.
    """
    os.makedirs(base_path, exist_ok=True)
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_page, idx, base_path) for idx in range(num_pages)]
        # Wait for all pages to be processed
        for future in futures:
            future.result()

def visualize_sample_page():
    """Create and visualize a sample page using Matplotlib."""
    page = Page()
    # Create a random layout with 1 to 8 text blocks.
    page.generate_random_layout(num_blocks=random.randint(1, 3))
    points, labels = page.get_points_and_labels()
    
    plt.figure(figsize=(10, 5))
    scatter = plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis', s=20)
    plt.title("Sample Page Visualization")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.gca().invert_yaxis()  # Invert y-axis if you want to mimic image coordinate systems.
    plt.colorbar(scatter, label='Line Index')
    plt.savefig('/home/kartik/layout-analysis/analysis_images/sample.png')
    #plt.show()

if __name__ == '__main__':
    generate_dataset_parallel(2000000)
    
