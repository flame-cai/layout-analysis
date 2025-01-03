import numpy as np
import random

CHARS_PER_LINE = 34  #max possible

def generate_text_layout(page_width=1250, page_height=532, num_lines=20, chars_per_line_range=(30, 50), 
                        footnotes_prob=0.2, footnote_length_range=(10, 30)):
    points = []
    labels = []
    current_label = 0
    
    # Parameters for text positioning
    left_margin = page_width * 0.15  # 15% margin from left
    right_margin = page_width * 0.85  # 15% margin from right
    line_height = page_height / (num_lines * 1.5)  # Allow space for footnotes
    char_spacing = (right_margin - left_margin) / CHARS_PER_LINE  # Maximum characters per line
    
    y_position = line_height  # Start from top with some margin

    # Generate main text lines
    for line_num in range(num_lines):
        # Randomly decide number of characters in this line
        num_chars = random.randint(*chars_per_line_range)
        
        # Generate character positions for this line
        for char_num in range(num_chars):
            x = left_margin + char_num * char_spacing
            # Add small random variation to make it look more natural
            x += random.uniform(-1, 1)
            y = y_position + random.uniform(-2, 2)
            
            points.append([int(x), int(y)])
            labels.append(current_label)
            current_label += 1
        
        # Randomly decide if this line has a footnote
        if random.random() < footnotes_prob:
            footnote_length = random.randint(*footnote_length_range)
            
            # Randomly choose footnote position (left, right, or below)
            footnote_position = random.choice(['left', 'right', 'below'])
            
            if footnote_position == 'left':
                # Position footnote to the left of the line
                footnote_x_start = left_margin * 0.3  # Start in left margin
                footnote_y = y_position
                char_spacing_factor = 1  # Denser spacing for side footnotes
            
            elif footnote_position == 'right':
                # Position footnote to the right of the line
                footnote_x_start = right_margin + (left_margin * 0.2)  # Start after right margin
                footnote_y = y_position
                char_spacing_factor = 1  # Denser spacing for side footnotes
            
            else:  # below
                # Position footnote below the line
                footnote_x_start = left_margin
                footnote_y = y_position + line_height * 0.7
                char_spacing_factor = 1  # Slightly denser spacing for below footnotes
            
            # Generate footnote characters
            for char_num in range(footnote_length):
                x = footnote_x_start + char_num * char_spacing * char_spacing_factor
                x += random.uniform(-1, 1)
                y = footnote_y + random.uniform(-1, 1)
                
                # For side footnotes, wrap to next line if too long
                if footnote_position in ['left', 'right']:
                    if footnote_position == 'left':
                        max_width = left_margin * 0.9
                    else:  # right
                        max_width = page_width - right_margin - (left_margin * 0.2)
                    
                    line_offset = int(char_num * char_spacing * char_spacing_factor / max_width)
                    if line_offset > 0:
                        x = footnote_x_start + (char_num * char_spacing * char_spacing_factor) % max_width
                        y = footnote_y + line_offset * line_height * 0.7
                
                points.append([int(x), int(y)])
                labels.append(current_label)
                current_label += 1
        
        y_position += line_height

    return np.array(points), np.array(labels)

def save_points_and_labels(points, labels, points_file="points.txt", labels_file="labels.txt"):
    """Save points and labels to separate filess"""
    path = '/home/kartik/layout-analysis/data/synthetic-data/'
    
    # Save original points and labels
    np.savetxt(path+points_file, points, fmt='%d', delimiter=' ')
    np.savetxt(path+labels_file, labels, fmt='%d')
    

def generate_realistic_parameters():
    """Generate realistic variations in page layout parameters"""
    # Page dimensions (slight variations around standard values)
    page_width = random.randint(1200, 1300)  # Varying around 1250
    page_height = random.randint(500, 550)   # Varying around 532
    
    # Number of lines varies by content density
    num_lines = random.randint(5, 12)  # Varying around 20
    
    # Character length varies by content style
    min_chars = random.randint(20, 24)
    max_chars = random.randint(30, CHARS_PER_LINE)
    chars_per_line_range = (min_chars, max_chars)
    
    # Footnote probability varies by document type
    footnotes_prob = random.uniform(0.10, 0.20)  # Varying around 0.2
    
    # Footnote length varies by content
    min_footnote = random.randint(1, 4)
    max_footnote = random.randint(8, 10)
    footnote_length_range = (min_footnote, max_footnote)
    
    return {
        'page_width': page_width,
        'page_height': page_height,
        'num_lines': num_lines,
        'chars_per_line_range': chars_per_line_range,
        'footnotes_prob': footnotes_prob,
        'footnote_length_range': footnote_length_range
    }

# Generate the dataset
for i in range(20000):
    params = generate_realistic_parameters()
    points, labels = generate_text_layout(**params)
    
    # Save to files with sorted points
    points_file_name = f"pg_{i}_points.txt"
    labels_file_name = f"pg_{i}_labels.txt"
    save_points_and_labels(points, labels, points_file_name, labels_file_name)