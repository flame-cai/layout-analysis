import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def visualize_text_layout(points_file, labels_file=None):
    """
    Visualize 2D points representing text layout with reading order numbers.
    
    Args:
        points_file (str): Path to the file containing x,y coordinates
        labels_file (str, optional): Path to the file containing labels
    """
    # Read points
    points = np.loadtxt(points_file)
    
    # Read labels if provided, otherwise create sequential labels
    if labels_file:
        labels = np.loadtxt(labels_file)
    else:
        labels = np.arange(len(points))
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Scatter points
    plt.scatter(points[:, 0], points[:, 1], c='lightgray', s=30, alpha=0.3)
    
    # Add reading order numbers
    # To avoid overcrowding, we can show every Nth number
    N = 1 # Show ~100 numbers
    for i in range(0, len(points), N):
        plt.annotate(f'{int(labels[i])}', 
                    (points[i, 0], points[i, 1]),
                    fontsize=8,
                    alpha=0.9)
    
    plt.title('Text Layout with Reading Order')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    
    # Add some stats in the corner
    stats_text = f'Total points: {len(points)}\n'
    stats_text += f'X range: {points[:, 0].min():.0f} to {points[:, 0].max():.0f}\n'
    stats_text += f'Y range: {points[:, 1].min():.0f} to {points[:, 1].max():.0f}'
    
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    plt.savefig('/home/kartik/layout-analysis/analysis_images/ordering')
   
    

# Example usage with your data
labels_file = "/home/kartik/layout-analysis/data/synthetic-data/pg_19_labels.txt"
points_file = "/home/kartik/layout-analysis/data/synthetic-data/pg_19_points.txt"  # Update with your actual file path
visualize_text_layout(points_file, labels_file)


# points_file = "/home/kartik/layout-analysis/data/test-data/points_3976_0001.jpg.txt"  # Update with your actual file path
# visualize_text_layout(points_file)