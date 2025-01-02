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

class PointDataset(Dataset):
    def __init__(self, data_dir, split_files, max_points=100):
        self.data_dir = data_dir
        self.max_points = max_points
        self.examples = []
        
        for file in split_files:
            points_file = os.path.join(data_dir, f"pg_{file}_points.txt")
            labels_file = os.path.join(data_dir, f"pg_{file}_labels.txt")
            
            points = np.loadtxt(points_file)
            labels = np.loadtxt(labels_file).astype(int)
            
            # Shuffle points and labels together
            indices = list(range(len(points)))
            random.shuffle(indices)
            points = points[indices]
            labels = labels[indices]
            # confirmed that these are shuffled
            
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
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
    def __init__(self, d_model=64, nhead=4, num_encoder_layers=3, num_classes=102):  # 20 + start/end tokens
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
        output = self.output(output)  # [batch_size, seq_len, num_classes]
        
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

def evaluate_and_visualize(model, test_loader, device='cuda', num_pages=10):
    model = model.to(device)
    model.eval()
    
    # Create a single iterator for the test loader
    test_iter = iter(test_loader)
    
    # Process multiple test pages
    for page_idx in range(num_pages):
        try:
            # Get next batch using the iterator
            points, true_labels = next(test_iter)
        except StopIteration:
            # If we run out of data, create new iterator
            test_iter = iter(test_loader)
            points, true_labels = next(test_iter)
            
        # Move points to the same device as model
        points = points.to(device)
        
        # Since batch size is 1, we can directly use index 0
        points_first = points[0]
        true_labels_first = true_labels[0]
        
        # Get predictions
        with torch.no_grad():
            output = model(points_first.unsqueeze(0))
            pred_labels = output[0].argmax(dim=-1).cpu()
        
        # Move points back to CPU for visualization
        points_first = points_first.cpu()
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Original order
        ax1.scatter(points_first[:, 0], points_first[:, 1])
        for i, (x, y) in enumerate(points_first):
            if true_labels_first[i] != -1:  # Skip padding
                ax1.annotate(str(true_labels_first[i].item()), (x, y))
        ax1.set_title(f'Original Reading Order - Page {page_idx + 1}')
        
        # Predicted order
        ax2.scatter(points_first[:, 0], points_first[:, 1])
        for i, (x, y) in enumerate(points_first):
            if true_labels_first[i] != -1:  # Skip padding
                ax2.annotate(str(pred_labels[i].item()), (x, y))
        ax2.set_title(f'Predicted Reading Order - Page {page_idx + 1}')
        
        plt.savefig(f'/home/kartik/layout-analysis/analysis_images/reading_order_comparison_page_{page_idx + 1}.png')
        plt.close()

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
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    # Create and train model
    model = ReadingOrderTransformer()
    train_model(model, train_loader, val_loader, device=device)
    
    # Load best model and evaluate
    model.load_state_dict(torch.load('/home/kartik/layout-analysis/models/best_model.pt'))
    evaluate_and_visualize(model, test_loader, device=device)

if __name__ == "__main__":
    main()