import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import random
import math

MAX_LENGTH = 1000

class PointDataset(Dataset):
    def __init__(self, data_dir, split_files, max_points=MAX_LENGTH, normalize=True):
        self.data_dir = data_dir
        self.max_points = max_points
        self.normalize = normalize
        self.examples = []
        
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
            labels_file = os.path.join(data_dir, f"pg_{file}_labels.txt")
            
            points = np.loadtxt(points_file)
            labels = np.loadtxt(labels_file).astype(int)
            
            # Normalize points if requested
            if normalize:
                points[:, 0] = (points[:, 0] - self.min_x) / (self.max_x - self.min_x)
                points[:, 1] = (points[:, 1] - self.min_y) / (self.max_y - self.min_y)
            
            # Shuffle points and labels together
            indices = list(range(len(points)))
            random.shuffle(indices)
            points = points[indices]
            labels = labels[indices]
            
            # Pad if necessary
            if len(points) < max_points:
                pad_length = max_points - len(points)
                points = np.pad(points, ((0, pad_length), (0, 0)), mode='constant')
                labels = np.pad(labels, (0, pad_length), mode='constant', constant_values=-1)
                
            self.examples.append((points, labels))
    
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LENGTH):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class ReadingOrderTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_encoder_layers=3, num_classes=MAX_LENGTH+2):  # MAX_LENGTH + start/end tokens
        super().__init__()
        
        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU()
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward=256)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # Output layer
        self.output = nn.Linear(d_model, num_classes)
        
    def forward(self, src):
        # src shape: [batch_size, seq_len, 2]
        
        # Embed input
        src = self.input_embed(src)  # [batch_size, seq_len, d_model]
        src = src.transpose(0, 1)  # [seq_len, batch_size, d_model]
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Transform
        output = self.transformer_encoder(src)
        
        # Output layer
        output = output.transpose(0, 1)  # [batch_size, seq_len, d_model]
        output = self.output(output)  # [batch_size, seq_len, num_classes] [32, 100, 100]
        
        return output

def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters())
    
    model = model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (points, labels) in enumerate(train_loader):
            points, labels = points.to(device), labels.to(device)
            
            optimizer.zero_grad()
            output = model(points)
            
            loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for points, labels in val_loader:
                points, labels = points.to(device), labels.to(device)
                output = model(points)
                val_loss += criterion(output.view(-1, output.size(-1)), labels.view(-1)).item()
        
        print(f'Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, Val Loss = {val_loss/len(val_loader):.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '/home/kartik/layout-analysis/models/best_model.pt')

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
            points, true_labels = next(test_iter)
        except StopIteration:
            test_iter = iter(test_loader)
            points, true_labels = next(test_iter)
            
        # Move data to appropriate device
        points = points.to(device)
        points_first = points[0]  # Get first example since batch_size=1
        true_labels_first = true_labels[0]
        
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
            points_first_denorm[:, 1] = (points_first[:, 1] * 
                (norm_params['max_y'] - norm_params['min_y']) + norm_params['min_y'])
        else:
            points_first_denorm = points_first
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(f'Reading Order Analysis - Page {page_idx + 1}', 
                    fontsize=16, y=1.05)
        
        # Helper function to create consistent point and label plotting
        def plot_points_and_labels(ax, points, labels, title):
            # Plot only valid points (not padding)
            valid_mask = labels != -1
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
        scatter1 = plot_points_and_labels(ax1, points_first_denorm, true_labels_first,
                                        'Original Reading Order')
        scatter2 = plot_points_and_labels(ax2, points_first_denorm, pred_labels,
                                        'Predicted Reading Order')
        
        # Add colorbars
        plt.colorbar(scatter1, ax=ax1, label='Reading Order Index')
        plt.colorbar(scatter2, ax=ax2, label='Reading Order Index')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f'/home/kartik/layout-analysis/analysis_images/reading_order_comparison_page_{page_idx + 1}.png',
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        # Print prediction accuracy for this page
        valid_mask = true_labels_first != -1
        accuracy = (pred_labels[valid_mask] == true_labels_first[valid_mask]).float().mean()
        print(f'Page {page_idx + 1} Accuracy: {accuracy:.2%}')

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    all_files = [f.split('_')[1] for f in os.listdir('/home/kartik/layout-analysis/data/synthetic-data') if f.endswith('points.txt')]
    random.shuffle(all_files)
    
    # Split files
    train_files = all_files[:int(0.7*len(all_files))]
    val_files = all_files[int(0.7*len(all_files)):int(0.85*len(all_files))]
    test_files = all_files[int(0.85*len(all_files)):]
    
    train_dataset = PointDataset('/home/kartik/layout-analysis/data/synthetic-data', train_files)
    val_dataset = PointDataset('/home/kartik/layout-analysis/data/synthetic-data', val_files)
    test_dataset = PointDataset('/home/kartik/layout-analysis/data/synthetic-data', test_files)

    train_norm_params = train_dataset.get_normalization_params()
    val_norm_params = val_dataset.get_normalization_params()
    test_norm_params = test_dataset.get_normalization_params()
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    # Create and train model
    model = ReadingOrderTransformer()
    train_model(model, train_loader, val_loader, device=device, num_epochs=10)
    
    # Load best model and evaluate
    model.load_state_dict(torch.load('/home/kartik/layout-analysis/models/best_model.pt'))
    evaluate_and_visualize(model, test_loader, device=device, norm_params=test_norm_params)

if __name__ == "__main__":
    main()


# TODO
# fix the generator code and make it mimic the manuscript well (top to bottom)
# fi
# normalize for page size
# ordering is pretty wrog
# make a nice evaluation metric
# ensure decoding does not repeat a class !! important
# how does transformer encoder work
# train longer