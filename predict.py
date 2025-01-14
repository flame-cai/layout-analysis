import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from architecture import ReadingOrderTransformer


MAX_NO_POINTS = 1200


class PointDataset(Dataset):
    def __init__(self, data_dir, split_files, max_points=MAX_NO_POINTS, normalize=True, prediction_mode = True):
        self.data_dir = data_dir
        self.max_points = max_points
        self.normalize = normalize
        self.examples = []
        self.prediction_mode = prediction_mode
        
        # Keep track of min/max coordinates for normalization
        self.min_x = float('inf')
        self.min_y = float('inf')
        self.max_x = float('-inf')
        self.max_y = float('-inf')
        
        # First pass: find min/max coordinates if normalizing
        if normalize:
            for file in split_files:
                points_file = os.path.join(data_dir, f"pg_{file}_points.txt")
                points = np.loadtxt(points_file)
                self.min_x = min(self.min_x, points[:, 0].min())
                self.min_y = min(self.min_y, points[:, 1].min())
                self.max_x = max(self.max_x, points[:, 0].max())
                self.max_y = max(self.max_y, points[:, 1].max())
        
        # Second pass: load and process data
        for file in split_files:
            points_file = os.path.join(data_dir, f"pg_{file}_points.txt")
            if self.prediction_mode == False:
                labels_file = os.path.join(data_dir, f"pg_{file}_labels.txt")
            
            points = np.loadtxt(points_file)

            if self.prediction_mode == False:
                labels = np.loadtxt(labels_file).astype(int)

                        # Pad if necessary
            if len(points) < max_points:
                pad_length = max_points - len(points)
                points = np.pad(points, ((0, pad_length), (0, 0)), mode='constant')

                if self.prediction_mode == False:
                    labels = np.pad(labels, (0, pad_length), mode='constant', constant_values=-1)
            
            # Normalize points if requested
            if normalize:
                points[:, 0] = (points[:, 0] - self.min_x) / (self.max_x - self.min_x)
                #points[:, 1] = (points[:, 1] - self.min_y) / (self.max_y - self.min_y)
                points[:, 1] = 1 - (points[:, 1] - self.min_y) / (self.max_y - self.min_y)
            
            # Shuffle points and labels together
            indices = list(range(len(points)))
            random.shuffle(indices)
            points = points[indices]
            if self.prediction_mode == False:
                labels = labels[indices]
                            
            if self.prediction_mode == False:
                self.examples.append((points, labels))
            else: # just order the points roughly
                self.examples.append((points, np.array([i for i in range(len(points))])))
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        points, labels = self.examples[idx]
        return torch.FloatTensor(points), torch.LongTensor(labels)
    
    def get_normalization_params(self):
        """Return normalization parameters for use in evaluation"""
        return {
            'min_x': self.min_x,
            'max_x': self.max_x,
            'min_y': self.min_y,
            'max_y': self.max_y
        }
    
def evaluate_and_visualize(model, test_loader, device='cuda', num_pages=10, norm_params=None):
    """
    Evaluate the model and create visualizations for multiple test pages.
    
    Args:
        model: The trained transformer model
        test_loader: DataLoader containing test data
        device: Device to run the model on ('cuda' or 'cpu')
        num_pages: Number of test pages to visualize
        norm_params: Dictionary containing normalization parameters {min_x, max_x, min_y, max_y}
    """
    # Ensure model is on correct device and in evaluation mode
    model = model.to(device)
    model.eval()
    
    # Create iterator for test loader
    test_iter = iter(test_loader)
    
    
    for page_idx in range(num_pages):
        # Get next test page
        try:
            points, _= next(test_iter)
        except StopIteration:
            test_iter = iter(test_loader)
            points, _ = next(test_iter)
            
        # Move data to appropriate device
        points = points.to(device)
        points_first = points[0]  # Get first example since batch_size=1
        _first = _[0]
        
        # Get model predictions
        with torch.no_grad():
            output = model(points_first.unsqueeze(0))
            pred_labels = output[0].argmax(dim=-1).cpu()
        
        # Move points back to CPU
        points_first = points_first.cpu()
        
        # Denormalize points if normalization parameters are provided
        if norm_params is not None:
            points_first_denorm = points_first.clone()
            points_first_denorm[:, 0] = (points_first[:, 0] * 
                (norm_params['max_x'] - norm_params['min_x']) + norm_params['min_x'])
            # points_first_denorm[:, 1] = (points_first[:, 1] * 
            #     (norm_params['max_y'] - norm_params['min_y']) + norm_params['min_y'])
            points_first_denorm[:, 1] = ((1 - points_first[:, 1]) * 
                (norm_params['max_y'] - norm_params['min_y']) + norm_params['min_y'])
        else:
            points_first_denorm = points_first
        
        # Create figure with two subplots
        fig, ax1 = plt.subplots(1, 1, figsize=(30, 15))
        fig.suptitle(f'Predicted Reading order - Page {page_idx + 1}', 
                    fontsize=16, y=1.05)
        
        # Helper function to create consistent point and label plotting
        def plot_points_and_labels(ax, points, labels, title):
            # Plot only valid points (not padding)
            valid_mask = (labels != -1) & (points[:, 0] >= 0) & (points[:, 1] >= 0)
            valid_points = points[valid_mask]
            valid_labels = labels[valid_mask]
            
            # Plot points with color mapping
            scatter = ax.scatter(valid_points[:, 0], valid_points[:, 1],
                               c=valid_labels, cmap='viridis',
                               s=100, zorder=5)
            
            # Add labels with white background for better visibility
            for i, (x, y) in enumerate(valid_points):
                label = valid_labels[i]
                ax.annotate(str(label.item()),
                           (x, y),
                           xytext=(5, 5),
                           textcoords='offset points',
                           ha='left',
                           va='bottom',
                           bbox=dict(boxstyle='round,pad=0.5',
                                   fc='white',
                                   ec='gray',
                                   alpha=0.8),
                           zorder=6)
            
            # Customize subplot
            ax.set_title(title, pad=20, fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xlabel('X Coordinate', fontsize=12)
            ax.set_ylabel('Y Coordinate', fontsize=12)
            
            return scatter
        
        # Create visualizations for both original and predicted orders
        scatter1 = plot_points_and_labels(ax1, points_first_denorm, pred_labels,
                                        'Predicted Reading Order')

 
        # Add colorbars
        plt.colorbar(scatter1, ax=ax1, label='Reading Order Index')

        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f'/home/kartik/layout-analysis/analysis_images/prediction_page_{page_idx + 1}.png',
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        # Print prediction accuracy for this page
        # valid_mask = true_labels_first != -1
        # accuracy = (pred_labels[valid_mask] == true_labels_first[valid_mask]).float().mean()
        # print(f'Page {page_idx + 1} Accuracy: {accuracy:.2%}')


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    all_files = [f.split('_')[1] for f in os.listdir('/home/kartik/layout-analysis/data/test-data') if f.endswith('points.txt')]
    random.shuffle(all_files)
    
    test_files = all_files
    test_dataset = PointDataset('/home/kartik/layout-analysis/data/test-data', test_files)
    test_norm_params = test_dataset.get_normalization_params()
    
    # Create data loaders

    test_loader = DataLoader(test_dataset, batch_size=1)
    
    # Create and train model
    model = ReadingOrderTransformer()
    
    # Load best model and evaluate
    model.load_state_dict(torch.load('/home/kartik/layout-analysis/models/best_model.pt'))
    evaluate_and_visualize(model, test_loader, device=device, norm_params=test_norm_params)

if __name__ == "__main__":
    main()