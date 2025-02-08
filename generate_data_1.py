import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor

MAX_CLASSES = 15
MAX_BLOCKS = 3

@dataclass
class Point:
    x: int
    y: int

class Line:
    def __init__(self, start_x: int, start_y: int, width: int, chars_count: int, 
                 alignment: str = 'left', curve_factor: float = 0.0):
        self.start_x = start_x
        self.start_y = start_y
        self.width = width
        self.chars_count = chars_count
        self.alignment = alignment
        self.curve_factor = curve_factor
        self.points: List[Point] = []
        self.base_spacing = self.width // (self.chars_count - 1)
        
    def generate_points(self) -> List[Point]:
        """Generate character points along the line with possible curve using vectorized operations"""
        if random.random() < 0.3:
            self.chars_count = int(self.chars_count*random.random())
        if self.chars_count <= 1:
            self.points = [Point(self.start_x, self.start_y)]
            return self.points

        # Determine a base spacing between characters
        #base_spacing = self.width // (self.chars_count - 1)
        
        # Generate randomized spacings with a minimum of 1 using NumPy
        spacings = np.maximum(1, (self.base_spacing + np.random.normal(0, self.base_spacing * 0.1, self.chars_count - 1)).astype(int))
        
        # Adjust the spacings so that the total matches the target width
        # total_spacing = spacings.sum()
        # scaling_factor = self.width / total_spacing
        # spacings = (spacings * scaling_factor).astype(int)
        
        # Compute cumulative x-offsets and positions
        x_offsets = np.concatenate(([0], np.cumsum(spacings)))
        x_positions = self.start_x + x_offsets
        
        # Compute y offsets using a sine function (vectorized)
        i_values = np.arange(self.chars_count)
        angles = (i_values / self.chars_count) * math.pi
        y_offsets = (np.sin(angles) * self.curve_factor).astype(int)
        y_positions = self.start_y + y_offsets 
        
        # Create the list of points
        
        points = [Point(int(x), int(y+random.uniform(-1.5, 1.5))) for x, y in zip(x_positions, y_positions)]
        
        # Apply alignment adjustments
        if self.alignment == 'center':
            offset = (self.width - (points[-1].x - points[0].x)) // 2
            for p in points:
                p.x += offset
        elif self.alignment == 'right':
            offset = self.width - (points[-1].x - points[0].x)
            for p in points:
                p.x += offset

        self.points = points
        return points

class TextBlock:
    def __init__(self, x: int, y: int, width: int, height: int, 
                 lines_count: int, chars_per_line: int,
                 alignment: str = 'left', allow_half_lines: bool = True):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.lines_count = lines_count
        self.base_chars_per_line = chars_per_line
        self.alignment = alignment
        self.allow_half_lines = allow_half_lines
        self.lines: List[Line] = []
        
    def generate_lines(self) -> List[Line]:
        """Generate lines for the text block"""
        line_height = np.random.randint(8,25) #self.height // self.lines_count
        
        for i in range(self.lines_count):
            # Possibly reduce the number of characters in some lines
                
            # Vary the y-position for each line slightly
            y_position = self.y + i * line_height
            y_position += random.randint(-line_height // 3, line_height // 3)
            
            # Apply a slight random curve to the line
            curve_factor = random.uniform(0, 3)
            line = Line(
                start_x=self.x,
                start_y=y_position,
                width=self.width,
                chars_count=self.base_chars_per_line,
                alignment=self.alignment,
                curve_factor=curve_factor
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
                    alignment=random.choice(['left', 'center', 'right'])
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
                alignment='left'
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
                for point_idx,point in enumerate(line.points):
                    if point_idx == 0:
                        label = MAX_CLASSES
                    elif point_idx == len(line.points) - 1:
                        label = MAX_CLASSES+1
                    else:
                        label = line_idx   
                    points.append([point.x, point.y])
                    labels.append(label)
                    
        # Convert to NumPy arrays and sort by y-coordinate
        points = np.array(points)
        labels = np.array(labels)
        sorted_indices = np.argsort(points[:, 1])
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
                              base_path: str = "/mnt/cai-data/manuscript-annotation-tool/synthetic-data/"):
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
    # Generate 10,000 pages in parallel.
    # visualize_sample_page()
    generate_dataset_parallel(200000)
    
