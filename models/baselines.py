import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split

class LayoutDataset(Dataset):
    def __init__(self, points, labels, max_len=800):
        """
        Dataset for layout points and their reading order.

        Args:
            points: List of numpy arrays containing points for each page
            labels: List of numpy arrays containing labels for each page
            max_len: Maximum length of the sequences.
        """
        self.points = points
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        points = torch.FloatTensor(self.points[idx])
        labels = torch.LongTensor(self.labels[idx])

        # Pad or truncate sequences to max_len
        if len(points) > self.max_len:
            points = points[:self.max_len]
            labels = labels[:self.max_len]
        elif len(points) < self.max_len:
            padding_len = self.max_len - len(points)
            points = torch.cat((points, torch.zeros((padding_len, points.shape[1]))), dim=0)
            labels = torch.cat((labels, torch.zeros(padding_len, dtype=torch.long)), dim=0)

        print(points.shape)
        print(labels.shape)
        return points, labels

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class LayoutTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, 
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # Input embedding layers
        self.point_embedding = nn.Sequential(
            nn.Linear(2, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, 1)  # Predicting single value (position in sequence)
        
    def forward(self, src, tgt):
        # Embed the input points
        src = self.point_embedding(src)
        tgt = self.point_embedding(tgt)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        # Create attention mask for decoder
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Transform
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        
        # Predict position
        output = self.output_layer(output)
        
        return output.squeeze(-1)

def load_data(data_dir):
    """Load all pages from the synthetic data directory."""
    points_files = sorted(glob.glob(os.path.join(data_dir, "*_points.txt")))
    labels_files = sorted(glob.glob(os.path.join(data_dir, "*_labels.txt")))
    
    points_list = []
    labels_list = []
    
    for p_file, l_file in zip(points_files, labels_files):
        points = np.loadtxt(p_file)
        labels = np.loadtxt(l_file)
        
        # Normalize points
        points = (points - points.mean(axis=0)) / points.std(axis=0)
        
        points_list.append(points)
        labels_list.append(labels)

    return points_list, labels_list

def train_model(model, train_loader, val_loader, num_epochs=100, device='cuda'):
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=0.0001)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

  best_val_loss = float('inf')

  for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    for points, labels in train_loader:
      points, labels = points.to(device), labels.to(device)

      # Check and ensure batch sizes are equal
      if points.shape[0] != labels.shape[0]:
        raise ValueError("Batch sizes of points and labels don't match!")

      # Use points as both source and target, shifted by one position
      tgt = points[:, :-1, :]  # Input sequence

      optimizer.zero_grad()
      output = model(points, tgt)
      loss = criterion(output.float(), labels.float())
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
      optimizer.step()

      train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
      for points, labels in val_loader:
        points, labels = points.to(device), labels.to(device)
        if points.shape[0] != labels.shape[0]:
          raise ValueError("Batch sizes of points and labels don't match!")
        tgt = points[:, :-1, :]
        output = model(points, tgt)
        loss = criterion(output.float(), labels.float())
        val_loss += loss.item()

    # Print progress
    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Training Loss: {train_loss/len(train_loader):.4f}')
    print(f'Validation Loss: {val_loss/len(val_loader):.4f}')

    # Learning rate scheduling
    scheduler.step(val_loss)

    # Save best model
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      torch.save(model.state_dict(), 'best_model.pth')

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    points_list, labels_list = load_data('/home/kartik/layout-analysis/data/synthetic-data')
    
    print(len(points_list))
    print(len(labels_list))

    # Split data at page level
    train_points, val_points, train_labels, val_labels = train_test_split(
        points_list, labels_list, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = LayoutDataset(train_points, train_labels)
    val_dataset = LayoutDataset(val_points, val_labels)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Initialize model
    model = LayoutTransformer().to(device)
    
    # Train model
    train_model(model, train_loader, val_loader, num_epochs=100, device=device)

if __name__ == "__main__":
    main()