import numpy as np
from typing import List, Tuple, Optional
import random
from dataclasses import dataclass
import math

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
        
    def generate_points(self) -> List[Point]:
        """Generate character points along the line with possible curve"""
        if self.chars_count <= 1:
            return [Point(self.start_x, self.start_y)]
            
        # Calculate spacing between characters
        base_spacing = self.width // (self.chars_count - 1)
        
        # Add some randomness to spacing
        spacings = [max(1, int(base_spacing + random.gauss(0, base_spacing * 0.1))) 
                   for _ in range(self.chars_count - 1)]
        
        # Adjust total width to match target width
        total_spacing = sum(spacings)
        scaling_factor = self.width / total_spacing
        spacings = [int(s * scaling_factor) for s in spacings]
        
        # Generate points with curve
        points = []
        current_x = self.start_x
        
        for i in range(self.chars_count):
            # Add slight curve using sine wave
            y_offset = int(math.sin(i / self.chars_count * math.pi) * self.curve_factor)
            points.append(Point(current_x, self.start_y + y_offset))
            
            if i < len(spacings):
                current_x += spacings[i]
        
        # Apply alignment
        if self.alignment == 'center':
            offset = (self.width - (points[-1].x - points[0].x)) // 2
            for point in points:
                point.x += offset
        elif self.alignment == 'right':
            offset = self.width - (points[-1].x - points[0].x)
            for point in points:
                point.x += offset
                
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
        line_height = self.height // self.lines_count
        
        for i in range(self.lines_count):
            # Randomize number of characters
            chars_count = self.base_chars_per_line

            if chars_count>2 and self.allow_half_lines and random.random() < 0.3:
                print('in the goddamn IF')
                chars_count = math.ceil(chars_count / 2)
                
            # Add some variance to line spacing
            y_position = self.y + i * line_height
            y_position += random.randint(-line_height//3, line_height//3)
            
            # Create line with slight random curve
            curve_factor = random.uniform(0, 3)
            print(f'initializing line with character count: {chars_count}')
            line = Line(
                start_x=self.x,
                start_y=y_position,
                width=self.width,
                chars_count=chars_count,
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
        """Add a text block after checking for overlaps"""
        # Simple overlap check using bounding boxes
        for existing_block in self.text_blocks:
            if (block.x < existing_block.x + existing_block.width and
                block.x + block.width > existing_block.x and
                block.y < existing_block.y + existing_block.height and
                block.y + block.height > existing_block.y):
                raise ValueError("Text blocks overlap!")
        self.text_blocks.append(block)
        
    def generate_random_layout(self, num_blocks: int = 8):
        """Generate a random layout with the specified number of text blocks"""
        self.text_blocks = []
        
        # Main text blocks
        for _ in range(num_blocks):
            attempts = 0
            while attempts < 100:
                width = random.randint(80, 1000)
                height = random.randint(80, 430)
                x = random.randint(0, self.width - width)
                y = random.randint(0, self.height - height)
                
                block = TextBlock(
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    lines_count=random.randint(1, 15),
                    chars_per_line=random.randint(2, 35),
                    alignment=random.choice(['left', 'center', 'right'])
                )
                
                try:
                    self.add_text_block(block)
                    block.generate_lines()
                    break
                except ValueError:
                    attempts += 1
                    
        # Maybe add marginalia
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
                chars_per_line=5,
                alignment='left'
            )
            
            try:
                self.add_text_block(marginalia)
                marginalia.generate_lines()
            except ValueError:
                pass  # Skip if overlap
                
    def get_points_and_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert all points to numpy arrays with labels"""
        points = []
        labels = []
        
        for block_idx, block in enumerate(self.text_blocks):
            print(f"block no: {block_idx}")
            label = block_idx
            for line_idx, line in enumerate(block.lines):
                print(f"line no: {line_idx}")
                #label = block_idx * 15 + line_idx
                for point in line.points:
                    points.append([point.x, point.y])
                    labels.append(label)
                    print(f'points: {[point.x, point.y]}')
                
        return np.array(points), np.array(labels)

def generate_dataset(num_pages: int = 100, base_path: str = "/mnt/cai-data/manuscript-annotation-tool/synthetic-data/"):
    """Generate multiple pages of synthetic data"""
    for page_idx in range(num_pages):
        # Create page with random layout
        page = Page()
        page.generate_random_layout(num_blocks=random.randint(1, 8))
        
        # Get points and labels 
        points, labels = page.get_points_and_labels()
        if len(points.shape)!=2:
            print('in generate dataset..')
            print(points.shape)
            print(points)
            print('stopping...')
            break
        
        # Save to files
        points_file = f"{page_idx}__points.txt"
        labels_file = f"{page_idx}__labels.txt"
        
        np.savetxt(base_path + points_file, points, fmt='%d', delimiter=' ')
        np.savetxt(base_path + labels_file, labels, fmt='%d')

generate_dataset(100000)