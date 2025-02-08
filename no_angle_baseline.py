import os
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from no_angle_architecture import ReadingOrderTransformer
from no_angle_data_loading import *

#DATA_PATH = "/home/kartik/layout-analysis/data/synthetic-data"
DATA_PATH = "/mnt/cai-data/manuscript-annotation-tool/synthetic-data"


def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    # Use AdamW with recommended hyperparameters
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Determine total steps and warmup steps for scheduling
    total_steps = len(train_loader) * num_epochs
    warmup_steps = min(5000, total_steps // 10)  # e.g., use 5000 or 10% of total steps if lower
    
    # Define a lambda function for linear warmup then cosine annealing decay
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine annealing after warmup:
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    model = model.to(device)
    best_val_loss = float('inf')
    
    # Set up mixed precision training if using CUDA
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    global_step = 0  # global step counter for scheduler updates
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (points, labels) in enumerate(train_loader):
            points, labels = points.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Use mixed precision autocast if available
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                output = model(points)
                loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))
            
            if scaler is not None:
                scaler.scale(loss).backward()
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            train_loss += loss.item()
            
            # Update scheduler each batch
            scheduler.step()
            global_step += 1
        
        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for points, labels in val_loader:
                points, labels = points.to(device), labels.to(device)
                output = model(points)
                val_loss += criterion(output.view(-1, output.size(-1)), labels.view(-1)).item()
        
        print(f'Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, '
              f'Val Loss = {val_loss/len(val_loader):.4f}')
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '/mnt/cai-data/manuscript-annotation-tool/models/segmentation/graph-models/best_model.pt')


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
    model = model.to(device)
    model.eval()
    
    test_iter = iter(test_loader)
    
    for page_idx in range(num_pages):
        try:
            points, true_labels = next(test_iter)
        except StopIteration:
            test_iter = iter(test_loader)
            points, true_labels = next(test_iter)
            
        points = points.to(device)
        points_first = points[0]  # Assuming batch_size=1 for test
        true_labels_first = true_labels[0]
        
        with torch.no_grad():
            output = model(points_first.unsqueeze(0))
            pred_labels = output[0].argmax(dim=-1).cpu()
        
        points_first = points_first.cpu()
        
        if norm_params is not None:
            points_first_denorm = points_first.clone()
            points_first_denorm[:, 0] = (points_first[:, 0] * 
                (norm_params['max_x'] - norm_params['min_x']) + norm_params['min_x'])
            points_first_denorm[:, 1] = ((1 - points_first[:, 1]) * 
                (norm_params['max_y'] - norm_params['min_y']) + norm_params['min_y'])
        else:
            points_first_denorm = points_first
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(27, 27))
        fig.suptitle(f'Reading Order Analysis - Page {page_idx + 1}', 
                     fontsize=16, y=1.05)
        
        def plot_points_and_labels(ax, points, labels, title):
            valid_mask = (labels != -1)
            valid_points = points[valid_mask]
            valid_labels = labels[valid_mask]
            
            scatter = ax.scatter(valid_points[:, 0], valid_points[:, 1],
                                 c=valid_labels, cmap='viridis',
                                 s=100, zorder=5)
            
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
            
            ax.set_title(title, pad=20, fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xlabel('X Coordinate', fontsize=12)
            ax.set_ylabel('Y Coordinate', fontsize=12)
            x_ticks = list(range(0, 1301, 100))
            y_ticks = list(range(0, 551, 100))
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            
            return scatter
        
        scatter1 = plot_points_and_labels(ax1, points_first_denorm, true_labels_first,
                                          'Original Reading Order')
        scatter2 = plot_points_and_labels(ax2, points_first_denorm, pred_labels,
                                          'Predicted Reading Order')
        
        plt.colorbar(scatter1, ax=ax1, label='Reading Order Index')
        plt.colorbar(scatter2, ax=ax2, label='Reading Order Index')
        
        plt.tight_layout()
        plt.savefig(f'/home/kartik/layout-analysis/analysis_images/reading_order_comparison_page_{page_idx + 1}.png',
                    bbox_inches='tight', dpi=300)
        plt.close()
        
        valid_mask = true_labels_first != -1
        accuracy = (pred_labels[valid_mask] == true_labels_first[valid_mask]).float().mean()
        print(f'Page {page_idx + 1} Accuracy: {accuracy:.2%}')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    all_files = [f.split('__')[0] for f in os.listdir(DATA_PATH) if f.endswith('__points.txt')]
    random.shuffle(all_files)
    
    train_files = all_files[:int(0.7*len(all_files))]
    val_files = all_files[int(0.7*len(all_files)):int(0.85*len(all_files))]
    test_files = all_files[int(0.85*len(all_files)):]
    
    train_dataset = PointDataset(DATA_PATH, train_files, labels_mode=True)
    val_dataset = PointDataset(DATA_PATH, val_files, labels_mode=True)
    test_dataset = PointDataset(DATA_PATH, test_files, labels_mode=True)

    train_norm_params = train_dataset.get_normalization_params()
    val_norm_params = val_dataset.get_normalization_params()
    test_norm_params = test_dataset.get_normalization_params()
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, pin_memory=True)
    
    # Create and train model (keeping architecture unchanged)
    model = ReadingOrderTransformer()
    train_model(model, train_loader, val_loader, device=device, num_epochs=20)
    
    # Load the best saved model and evaluate
    model.load_state_dict(torch.load('/mnt/cai-data/manuscript-annotation-tool/models/segmentation/graph-models/best_model.pt'))
    evaluate_and_visualize(model, test_loader, device=device, norm_params=test_norm_params)


if __name__ == "__main__":
    main()
