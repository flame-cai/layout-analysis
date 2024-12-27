import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import glob
import os
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from models import *
# nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9

#TODO
# does the ordering of the points matter? translation invariant
# optimize model hyperparameters
# hidden dime, dim feed forward, max points


class PageDataset(Dataset):
    def __init__(self, data_dir, page_indices, max_points=500):
        self.data_dir = data_dir
        self.page_indices = page_indices
        self.max_points = max_points
        
    def __len__(self):
        return len(self.page_indices)
    
    def __getitem__(self, idx):
        page_idx = self.page_indices[idx]
        points = np.loadtxt(os.path.join(self.data_dir, f'pg_{page_idx}_points.txt'))
        labels = np.loadtxt(os.path.join(self.data_dir, f'pg_{page_idx}_labels.txt'))
        
        # Convert to tensors
        points = torch.tensor(points, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)
        
        # Store original sequence length
        seq_len = len(points)
        
        # Pad if necessary
        if seq_len < self.max_points:
            pad_points = torch.zeros((self.max_points - seq_len, 2))
            pad_labels = torch.ones((self.max_points - seq_len)) * -1  # Use -1 for padding
            points = torch.cat([points, pad_points], dim=0)
            labels = torch.cat([labels, pad_labels], dim=0)
        
        return points, labels, seq_len

def collate_fn(batch):
    points, labels, seq_lens = zip(*batch)
    points = torch.stack(points)
    labels = torch.stack(labels)
    seq_lens = torch.tensor(seq_lens)
    return points, labels, seq_lens

def create_dataloaders(data_dir, batch_size=10, train_ratio=0.8, val_ratio=0.1):
    # Get all page indices
    all_files = glob.glob(os.path.join(data_dir, 'pg_*_points.txt'))
    page_indices = [int(f.split('_')[1]) for f in all_files]
    
    # Split indices at page level
    train_indices, temp_indices = train_test_split(page_indices, train_size=train_ratio)
    val_indices, test_indices = train_test_split(temp_indices, train_size=val_ratio/(1-train_ratio))
    
    # Create datasets
    train_dataset = PageDataset(data_dir, train_indices)
    val_dataset = PageDataset(data_dir, val_indices)
    test_dataset = PageDataset(data_dir, test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader

# Training loop for Sequence-to-Sequence model
def train_seq2seq(model, train_loader, val_loader, num_epochs=5, device='cuda', 
                  batch_size=32, gradient_accumulation_steps=4):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    
    # Enable automatic mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        print(len(train_loader))
        for batch_idx, (points, labels, seq_lens) in enumerate(train_loader):
            # Use a smaller batch size by processing partial batches
            batch_start_idx = 0
            num_samples = points.size(0)
            print(num_samples)
            
            while batch_start_idx < num_samples:
                batch_end_idx = min(batch_start_idx + batch_size, num_samples)
                
                batch_points = points[batch_start_idx:batch_end_idx].to(device)
                batch_labels = labels[batch_start_idx:batch_end_idx].to(device)
                batch_seq_lens = seq_lens[batch_start_idx:batch_end_idx].to(device)
                
                # Normalize points
                batch_points[:,:,0] = batch_points[:,:,0] / 1250
                batch_points[:,:,1] = batch_points[:,:,1] / 532
                
                # Create mask for padding - moved to device
                mask = (torch.arange(batch_points.size(1), device=device)[None, :] < batch_seq_lens[:, None])
                
                # Use automatic mixed precision
                with torch.amp.autocast('cuda'):
                    # Forward pass
                    predicted_positions = model(batch_points)
                    
                    # Normalize labels to [0,1] and mask padding
                    normalized_labels = batch_labels.float() / batch_seq_lens.unsqueeze(1)
                    normalized_labels[~mask] = 0
                    predicted_positions[~mask] = 0
                    
                    # Calculate loss only on non-padded elements
                    loss = criterion(predicted_positions[mask], normalized_labels[mask])
                    loss = loss / gradient_accumulation_steps  # Scale loss
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Free up memory
                del batch_points, batch_labels, batch_seq_lens, predicted_positions, normalized_labels
                torch.cuda.empty_cache()
                
                batch_start_idx += batch_size
                train_loss += loss.item() * gradient_accumulation_steps
            
            # Update weights after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for points, labels, seq_lens in val_loader:
                # Process validation data in smaller batches too
                batch_start_idx = 0
                num_samples = points.size(0)
                
                while batch_start_idx < num_samples:
                    batch_end_idx = min(batch_start_idx + batch_size, num_samples)
                    
                    batch_points = points[batch_start_idx:batch_end_idx].to(device)
                    batch_labels = labels[batch_start_idx:batch_end_idx].to(device)
                    batch_seq_lens = seq_lens[batch_start_idx:batch_end_idx].to(device)
                    
                    batch_points[:,:,0] = batch_points[:,:,0] / 1250
                    batch_points[:,:,1] = batch_points[:,:,1] / 532
                    
                    mask = (torch.arange(batch_points.size(1), device=device)[None, :] < batch_seq_lens[:, None])
                    
                    with torch.amp.autocast('cuda'):
                        predicted_positions = model(batch_points)
                        
                        normalized_labels = batch_labels.float() / batch_seq_lens.unsqueeze(1)
                        normalized_labels[~mask] = 0
                        predicted_positions[~mask] = 0
                        
                        loss = criterion(predicted_positions[mask], normalized_labels[mask])
                    
                    val_loss += loss.item()
                    
                    # Free up memory
                    del batch_points, batch_labels, batch_seq_lens, predicted_positions, normalized_labels
                    torch.cuda.empty_cache()
                    
                    batch_start_idx += batch_size
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'models/best_seq2seq_model.pt')
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')


# Training loop for GNN model
def train_gnn(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for points, labels, seq_lens in train_loader:
            batch_graphs = []
            batch_labels = []
            
            # Process each page in the batch
            for i in range(len(points)):
                valid_points = points[i][:seq_lens[i]]
                valid_labels = labels[i][:seq_lens[i]]
                
                # Build graph
                edge_index = build_graph_from_points(valid_points.cpu().numpy())  # Convert to numpy
                graph_data = Data(
                    x=valid_points.to(device),
                    edge_index=edge_index.to(device),
                    y=valid_labels.float().to(device) / seq_lens[i]
)
                
                batch_graphs.append(graph_data)
            
            # Batch the graphs
            batch = torch_geometric.data.Batch.from_data_list(batch_graphs)
            
            # Forward pass
            pred = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(pred, batch.y.view(-1, 1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for points, labels, seq_lens in val_loader:
                batch_graphs = []
                
                for i in range(len(points)):
                    valid_points = points[i][:seq_lens[i]]
                    valid_labels = labels[i][:seq_lens[i]]
                    
                    edge_index = build_graph_from_points(valid_points.numpy())
                    graph_data = Data(
                        x=valid_points,
                        edge_index=edge_index,
                        y=valid_labels.float() / seq_lens[i]
                    ).to(device)
                    
                    batch_graphs.append(graph_data)
                
                batch = torch_geometric.data.Batch.from_data_list(batch_graphs)
                pred = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(pred, batch.y.view(-1, 1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'models/best_gnn_model.pt')
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')

# Usage example
if __name__ == "__main__":

    data_dir = "/home/kartik/layout-analysis/data/synthetic-data"
    train_loader, val_loader, test_loader = create_dataloaders(data_dir, batch_size=4)
    
    # # Train Seq2Seq model
    # Train Seq2Seq model with memory optimizations
    seq2seq_model = ReadingOrderPredictor()
    train_seq2seq(seq2seq_model, train_loader, val_loader, 
                  batch_size=2,  # Process 4 samples at a time
                  gradient_accumulation_steps=4)  # Accumulate gradients for 4 steps
    
    # Train GNN model
    # gnn_model = ReadingOrderGNN()
    # train_gnn(gnn_model, train_loader, val_loader)