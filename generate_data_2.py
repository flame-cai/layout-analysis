import numpy as np
import random

CHARS_PER_LINE = 34  # max possible characters per line for single column

def generate_text_layout(page_width=1250, page_height=532, num_lines=20, chars_per_line_range=(30, 50),
                        footnotes_prob=0.2, footnote_length_range=(10, 30), layout_type='single'):
    """
    Generate text layout with support for single and double columns
    layout_type: 'single' or 'double' for column layout
    """
    points = []
    labels = []
    current_label = 0
    
    # Adjust parameters based on layout type
    if layout_type == 'double':
        # For double column, split the content area in half
        column_width = (page_width * 0.7) / 2  # 70% of page width split into two columns
        left_margins = [
            page_width * 0.15,  # First column starts at 15% margin
            page_width * 0.15 + column_width + (page_width * 0.1)  # Second column starts after first + gap
        ]
        right_margins = [
            left_margins[0] + column_width,  # First column end
            left_margins[1] + column_width   # Second column end
        ]
        chars_per_column = CHARS_PER_LINE // 2  # Reduce characters per line for each column
    else:  # single column
        left_margins = [page_width * 0.15]  # 15% margin from left
        right_margins = [page_width * 0.85]  # 15% margin from right
        chars_per_column = CHARS_PER_LINE
    
    line_height = page_height / (num_lines * 1.5)  # Allow space for footnotes
    
    # Generate text for each column
    for col_idx, (left_margin, right_margin) in enumerate(zip(left_margins, right_margins)):
        char_spacing = (right_margin - left_margin) / chars_per_column
        y_position = line_height  # Start from top with some margin
        
        # Adjust number of lines for double column to fill page
        col_lines = num_lines * 2 if layout_type == 'double' else num_lines
        
        # Generate main text lines for this column
        for line_num in range(col_lines):
            # Adjust character range for column width
            adjusted_range = (
                min(chars_per_line_range[0], chars_per_column),
                min(chars_per_line_range[1], chars_per_column)
            )
            num_chars = random.randint(*adjusted_range)
            
            # Generate character positions for this line
            for char_num in range(num_chars):
                x = left_margin + char_num * char_spacing
                # Add small random variation to make it look more natural
                x += random.uniform(-1, 1)
                y = y_position + random.uniform(-2, 2)
                
                points.append([int(x), int(y)])
                labels.append(current_label)
            
            # Handle footnotes
            if random.random() < footnotes_prob:
                footnote_length = random.randint(*footnote_length_range)
                footnote_position = random.choice(['left', 'right', 'below'])
                
                # Adjust footnote positioning for columns
                if footnote_position == 'left':
                    footnote_x_start = left_margin - (page_width * 0.1)
                    footnote_y = y_position
                    char_spacing_factor = 0.8
                elif footnote_position == 'right':
                    footnote_x_start = right_margin + (page_width * 0.02)
                    footnote_y = y_position
                    char_spacing_factor = 0.8
                else:  # below
                    footnote_x_start = left_margin
                    footnote_y = y_position + line_height * 0.7
                    char_spacing_factor = 1
                
                # Generate footnote characters
                for char_num in range(footnote_length):
                    x = footnote_x_start + char_num * char_spacing * char_spacing_factor
                    x += random.uniform(-1, 1)
                    y = footnote_y + random.uniform(-1, 1)
                    
                    # Handle footnote wrapping
                    if footnote_position in ['left', 'right']:
                        if footnote_position == 'left':
                            max_width = (left_margin - (page_width * 0.1)) * 0.9
                        else:  # right
                            max_width = (page_width - right_margin - (page_width * 0.02))
                        
                        line_offset = int(char_num * char_spacing * char_spacing_factor / max_width)
                        if line_offset > 0:
                            x = footnote_x_start + (char_num * char_spacing * char_spacing_factor) % max_width
                            y = footnote_y + line_offset * line_height * 0.7
                    
                    points.append([int(x), int(y)])
                    labels.append(current_label)
            
            current_label += 1
            y_position += line_height

    return np.array(points), np.array(labels)

def generate_realistic_parameters():
    """Generate realistic variations in page layout parameters"""
    page_width = random.randint(1200, 1300)
    page_height = random.randint(500, 550)
    num_lines = random.randint(5, 12)
    
    # Adjust character ranges based on layout type
    layout_type = random.choice(['single', 'double'])
    if layout_type == 'double':
        min_chars = random.randint(10, 12)  # Smaller range for double column
        max_chars = random.randint(15, CHARS_PER_LINE // 2)
    else:
        min_chars = random.randint(20, 24)
        max_chars = random.randint(30, CHARS_PER_LINE)
    
    chars_per_line_range = (min_chars, max_chars)
    footnotes_prob = random.uniform(0.10, 0.20)
    
    min_footnote = random.randint(1, 4)
    max_footnote = random.randint(8, 10)
    footnote_length_range = (min_footnote, max_footnote)
    
    return {
        'page_width': page_width,
        'page_height': page_height,
        'num_lines': num_lines,
        'chars_per_line_range': chars_per_line_range,
        'footnotes_prob': footnotes_prob,
        'footnote_length_range': footnote_length_range,
        'layout_type': layout_type
    }



def save_points_and_labels(points, labels, points_file="points.txt", labels_file="labels.txt"):
    """Save points and labels to separate filess"""
    path = '/home/kartik/layout-analysis/data/synthetic-data/'
    
    # Save original points and labels
    np.savetxt(path+points_file, points, fmt='%d', delimiter=' ')
    np.savetxt(path+labels_file, labels, fmt='%d')

    
# Generate the dataset
for i in range(20000):
    params = generate_realistic_parameters()
    points, labels = generate_text_layout(**params)
    
    # Save to files with sorted points
    points_file_name = f"pg_{i}_points.txt"
    labels_file_name = f"pg_{i}_labels.txt"
    save_points_and_labels(points, labels, points_file_name, labels_file_name)