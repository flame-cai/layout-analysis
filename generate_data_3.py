import numpy as np
import random
import math

CHARS_PER_LINE = 50  # max possible characters per line for single column

def generate_text_layout(
    page_width=1250, page_height=532, num_lines=20, chars_per_line_range=(30, 50),
    footnotes_prob=0.2, footnote_length_range=(10, 30), layout_type='single',
    text_direction='ltr', rotation_angle=0, text_curve=0.0, 
    line_spacing_variation=0.0, char_spacing_variation=0.0,
    paragraph_style='block', has_header=False, has_footer=False,
    handwriting_variation=0.0, start_y_variation=0.0):
    """
    Enhanced text layout generator with additional variations
    
    New Parameters:
    paragraph_style: 'block', 'indented', or 'spaced'
    has_header: Boolean for header presence
    has_footer: Boolean for footer presence
    handwriting_variation: Amount of random variation in character placement
    start_y_variation: Variation in where the first line starts (0.0 to 0.3)
    """
    points = []
    labels = []
    current_label = 1
    
    # Calculate effective page height accounting for headers and footers
    effective_height = page_height
    top_margin = page_height * 0.1  # Default top margin
    
    if has_header:
        header_height = page_height * 0.08
        effective_height -= header_height
        # Add header text
        header_y = page_height - (header_height / 2)
        header_chars = random.randint(20, 40)
        header_start = page_width * 0.15
        header_spacing = (page_width * 0.7) / header_chars
        for i in range(header_chars):
            x = header_start + i * header_spacing + random.uniform(-1, 1)
            y = header_y + random.uniform(-2, 2)
            points.append([int(x), int(y)])
            labels.append(0)  # Header gets label 0
    else:
        header_height = 0
    
    if has_footer:
        footer_height = page_height * 0.08
        effective_height -= footer_height
        # Add footer text
        footer_y = footer_height / 2
        footer_chars = random.randint(20, 40)
        footer_start = page_width * 0.15
        footer_spacing = (page_width * 0.7) / footer_chars
        for i in range(footer_chars):
            x = footer_start + i * footer_spacing + random.uniform(-1, 1)
            y = footer_y + random.uniform(-2, 2)
            points.append([int(x), int(y)])
            labels.append(0)  # Footer gets label 0
    else:
        footer_height = 0
    
    # Apply start_y_variation to top margin
    top_margin = page_height * (0.1 + random.uniform(0, start_y_variation))
    
    # Adjust parameters based on layout type
    if layout_type == 'double':
        # Random column widths for variety
        left_col_width = (page_width * 0.7) * random.uniform(0.45, 0.55)
        right_col_width = (page_width * 0.7) - left_col_width
        
        left_margins = [
            page_width * 0.15,  # First column
            page_width * 0.15 + left_col_width + (page_width * 0.1)  # Second column
        ]
        right_margins = [
            left_margins[0] + left_col_width,  # First column end
            left_margins[1] + right_col_width  # Second column end
        ]
        chars_per_column = CHARS_PER_LINE // 2
    else:
        left_margins = [page_width * 0.15]
        right_margins = [page_width * 0.85]
        chars_per_column = CHARS_PER_LINE
    
    base_line_height = (effective_height - top_margin - footer_height) / (num_lines * 1.5)
    
    def rotate_point(x, y, angle, center_x, center_y):
        angle_rad = math.radians(angle)
        dx = x - center_x
        dy = y - center_y
        rotated_x = dx * math.cos(angle_rad) - dy * math.sin(angle_rad) + center_x
        rotated_y = dx * math.sin(angle_rad) + dy * math.cos(angle_rad) + center_y
        return rotated_x, rotated_y
    
    # Generate text for each column
    for col_idx, (left_margin, right_margin) in enumerate(zip(left_margins, right_margins)):
        y_position = page_height - top_margin
        
        col_lines = num_lines * 2 if layout_type == 'double' else num_lines
        
        # Track paragraphs
        current_paragraph = 0
        lines_in_paragraph = 0
        paragraph_length = random.randint(3, 8)  # Random paragraph length
        
        for line_num in range(col_lines):
            # Handle paragraph breaks
            if lines_in_paragraph >= paragraph_length:
                current_paragraph += 1
                lines_in_paragraph = 0
                paragraph_length = random.randint(3, 8)
                if paragraph_style == 'spaced':
                    y_position -= base_line_height * 0.5  # Extra space between paragraphs
            
            line_height = base_line_height * (1 + random.uniform(-line_spacing_variation, line_spacing_variation))
            curve_offset = lambda x: math.sin(x * math.pi / (right_margin - left_margin)) * text_curve * base_line_height
            
            # Adjust starting position based on paragraph style
            current_left_margin = left_margin
            if paragraph_style == 'indented' and lines_in_paragraph == 0:
                current_left_margin += random.uniform(20, 40)  # Paragraph indentation
            
            adjusted_range = (
                min(chars_per_line_range[0], chars_per_column),
                min(chars_per_line_range[1], chars_per_column)
            )
            num_chars = random.randint(*adjusted_range)
            
            # Generate character positions
            char_spacing = (right_margin - current_left_margin) / num_chars
            for char_num in range(num_chars):
                char_space = char_spacing * (1 + random.uniform(-char_spacing_variation, char_spacing_variation))
                
                if text_direction == 'ltr':
                    x = current_left_margin + char_num * char_space
                else:
                    x = right_margin - char_num * char_space
                
                # Add handwriting variation
                x += random.uniform(-handwriting_variation * 5, handwriting_variation * 5)
                y = y_position + random.uniform(-handwriting_variation * 5, handwriting_variation * 5)
                
                y += curve_offset(x - current_left_margin)
                
                if rotation_angle != 0:
                    x, y = rotate_point(x, y, rotation_angle, page_width/2, page_height/2)
                
                points.append([int(x), int(y)])
                labels.append(current_label)
            
            # Handle footnotes with fixed positions
            if random.random() < footnotes_prob:
                footnote_length = random.randint(*footnote_length_range)
                
                # In double column layout, only allow footnotes at outer margins
                if layout_type == 'double':
                    if col_idx == 0:  # First column
                        footnote_position = random.choice(['left', 'below'])
                    else:  # Second column
                        footnote_position = random.choice(['right', 'below'])
                else:
                    footnote_position = random.choice(['left', 'right', 'below'])
                
                if footnote_position == 'left':
                    footnote_x_start = left_margins[0] - (page_width * random.uniform(0.08, 0.12))
                    footnote_y = y_position
                    char_spacing_factor = random.uniform(0.7, 0.9)
                elif footnote_position == 'right':
                    footnote_x_start = right_margins[-1] + (page_width * random.uniform(0.01, 0.03))
                    footnote_y = y_position
                    char_spacing_factor = random.uniform(0.7, 0.9)
                else:
                    footnote_x_start = current_left_margin
                    footnote_y = y_position + line_height * random.uniform(0.6, 0.8)
                    char_spacing_factor = random.uniform(0.9, 1.1)
                
                for char_num in range(footnote_length):
                    x = footnote_x_start + char_num * char_spacing * char_spacing_factor
                    x += random.uniform(-1, 1)
                    y = footnote_y + random.uniform(-1, 1)
                    
                    if footnote_position in ['left', 'right']:
                        if footnote_position == 'left':
                            max_width = (left_margins[0] - (page_width * 0.1)) * 0.9
                        else:
                            max_width = (page_width - right_margins[-1] - (page_width * 0.02))
                        
                        line_offset = int(char_num * char_spacing * char_spacing_factor / max_width)
                        if line_offset > 0:
                            x = footnote_x_start + (char_num * char_spacing * char_spacing_factor) % max_width
                            y = footnote_y + line_offset * line_height * random.uniform(0.65, 0.75)
                    
                    if rotation_angle != 0:
                        x, y = rotate_point(x, y, rotation_angle, page_width/2, page_height/2)
                    
                    points.append([int(x), int(y)])
                    labels.append(0)
            
            current_label += 1
            y_position -= line_height
            lines_in_paragraph += 1

    return np.array(points), np.array(labels)

def generate_realistic_parameters():
    """Generate realistic variations in page layout parameters with enhanced options"""
    layout_type = random.choice(['single', 'double'])
    
    # Basic layout parameters
    page_width = random.randint(1000, 1500)
    page_height = random.randint(450, 600)
    num_lines = random.randint(5, 15)
    
    # Character ranges based on layout type
    if layout_type == 'double':
        min_chars = random.randint(8, 15)
        max_chars = random.randint(15, CHARS_PER_LINE // 2)
    else:
        min_chars = random.randint(8, 15)
        max_chars = random.randint(25, CHARS_PER_LINE)
    
    chars_per_line_range = (min_chars, max_chars)
    
    # Enhanced variation parameters
    text_direction = random.choice(['ltr', 'rtl'])
    rotation_angle = random.choice([0, 0, 0, random.uniform(-5, 5)])
    text_curve = random.choice([0, 0, 0, random.uniform(0, 0.5)])
    
    # Spacing variations
    line_spacing_variation = random.uniform(0, 0.2)
    char_spacing_variation = random.uniform(0, 0.15)
    
    # New style parameters
    paragraph_style = random.choice(['block', 'indented', 'spaced'])
    has_header = random.choice([True, False])
    has_footer = random.choice([True, False])
    handwriting_variation = random.uniform(0, 0.3)
    start_y_variation = random.uniform(0, 0.3)
    
    # Footnote parameters
    footnotes_prob = random.uniform(0.05, 0.25)
    min_footnote = random.randint(1, 5)
    max_footnote = random.randint(8, 15)
    footnote_length_range = (min_footnote, max_footnote)
    
    return {
        'page_width': page_width,
        'page_height': page_height,
        'num_lines': num_lines,
        'chars_per_line_range': chars_per_line_range,
        'footnotes_prob': footnotes_prob,
        'footnote_length_range': footnote_length_range,
        'layout_type': layout_type,
        'text_direction': text_direction,
        'rotation_angle': rotation_angle,
        'text_curve': text_curve,
        'line_spacing_variation': line_spacing_variation,
        'char_spacing_variation': char_spacing_variation,
        'paragraph_style': paragraph_style,
        'has_header': has_header,
        'has_footer': has_footer,
        'handwriting_variation': handwriting_variation,
        'start_y_variation': start_y_variation
    }


def save_points_and_labels(points, labels, points_file="points.txt", labels_file="labels.txt"):
    """Save points and labels to separate filess"""
    path = '/mnt/cai-data/layout-analysis/synthetic-data/'
    
    # Save original points and labels
    np.savetxt(path+points_file, points, fmt='%d', delimiter=' ')
    np.savetxt(path+labels_file, labels, fmt='%d')

    
# Generate the dataset
for i in range(100000):
    params = generate_realistic_parameters()
    points, labels = generate_text_layout(**params)
    
    # Save to files with sorted points
    points_file_name = f"pg_{i}_points.txt"
    labels_file_name = f"pg_{i}_labels.txt"
    save_points_and_labels(points, labels, points_file_name, labels_file_name)
