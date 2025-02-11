import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import random
from angle_architecture import ReadingOrderTransformer
from angle_data_loading import *

MANUSCRIPT_NAME = 'clean'
DATA_PATH = f'/mnt/cai-data/manuscript-annotation-tool/manuscripts/{MANUSCRIPT_NAME}/points-2D'

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

        # Sort by the second column (y-coordinates)
        sorted_tensor, indices = torch.sort(points_first, dim=0)
        # Get the sorted tensor based on the y-coordinates
        points_first = points_first[indices[:, 1]]

        _first = _[0]
        
        # Get model predictions
        # with torch.no_grad():
        #     output = model(points_first.unsqueeze(0))
        #     print(output[0].argmax(dim=-1).shape)
        #     pred_labels = output[0].argmax(dim=-1).cpu()

        with torch.no_grad():
            output = model(points_first.unsqueeze(0))
            probabilities = torch.softmax(output[0], dim=-1)  # Convert logits to probabilities
            topk_probs, topk_indices = probabilities.topk(k=5, dim=-1)  # Get top 3 predictions
            # Example: Choose the one with the highest probability among top-k
            print(topk_indices[500:530,:])
            pred_labels = topk_indices[:, 0].cpu()
            print(pred_labels[500:530])

        # with torch.no_grad():
        #     output = model(points_first.unsqueeze(0))
        #     probabilities = torch.softmax(output[0], dim=-1)  # Convert logits to probabilities
        #     entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=-1)
        #     low_entropy_mask = entropy < 1.0  # Example threshold for low entropy
        #     pred_labels = torch.where(
        #         low_entropy_mask,
        #         probabilities.argmax(dim=-1),
        #         torch.tensor(-1, device=output.device)  # Use -1 for uncertain predictions
        #     ).cpu()



        # Convert to a tensor if needed
        pred_labels = torch.tensor(pred_labels)
        


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
            for i, (x, y) in enumerate(valid_points[:,:2]):
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
            x_ticks = list(range(0, 1301, 100))
            y_ticks = list(range(0, 551, 100))
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)

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
    #all_files = [f.split('_')[1] for f in os.listdir(DATA_PATH) if f.endswith('.txt')]
    all_files = [f.split('__')[0] for f in os.listdir(DATA_PATH) if f.endswith('__points.txt')]
    random.shuffle(all_files)
    
    test_files = all_files
    test_dataset = PointDataset(DATA_PATH, test_files, labels_mode=False)
    test_norm_params = test_dataset.get_normalization_params()
    
    # Create data loaders

    test_loader = DataLoader(test_dataset, batch_size=1)
    
    # Create and train model
    model = ReadingOrderTransformer()
    
    # Load best model and evaluate
    model.load_state_dict(torch.load('/mnt/cai-data/manuscript-annotation-tool/models/segmentation/graph-models/best_model.pt'))
    evaluate_and_visualize(model, test_loader, device=device, norm_params=test_norm_params)

if __name__ == "__main__":
    main()