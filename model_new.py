import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import random
from pathlib import Path
import math
from torch.nn.utils.rnn import pad_sequence

# Special tokens
PAD_IDX = 0
START_IDX = 1
END_IDX = 2
NUM_SPECIAL_TOKENS = 3  # PAD, START, END
MAX_SEQ_LENGTH = 70  # Maximum sequence length including special tokens


def tokens_to_points(tokens, num_special_tokens=NUM_SPECIAL_TOKENS):
    """Convert token sequence back to points"""
    points = []
    current_point = []
    
    # Skip START token and iterate until END token or padding
    for token in tokens[1:]:  # Skip START token
        if token == END_IDX or token == PAD_IDX:
            break
        
        # Convert token back to digit
        digit = token.item() - num_special_tokens
        if digit >= 0:  # Ignore special tokens
            current_point.append(str(digit))
            
        # If we have collected enough digits for a point
        if len(current_point) == 6:  # Assuming 3 digits each for x and y
            x = int(''.join(current_point[:3]))
            y = int(''.join(current_point[3:]))
            points.append([x, y])
            current_point = []
    
    return np.array(points)

def predict_sequence(model, input_tokens, device='cuda'):
    """Generate point sequence using the trained model"""
    model.eval()
    with torch.no_grad():
        # Prepare input
        src = input_tokens.unsqueeze(0).to(device)  # Add batch dimension
        
        # Initialize target sequence with START token
        tgt = torch.zeros(1, 1, dtype=torch.long, device=device)
        tgt[0, 0] = START_IDX
        
        max_len = 500
        
        # Generate sequence
        for i in range(max_len - 1):
            # Get prediction
            output = model(src, tgt)
            next_token = output[0, -1].argmax()
            
            # Append prediction to target sequence
            next_tgt = torch.zeros(1, tgt.size(1) + 1, dtype=torch.long, device=device)
            next_tgt[0, :-1] = tgt[0]
            next_tgt[0, -1] = next_token
            tgt = next_tgt
            
            # Stop if END token is predicted
            if next_token == END_IDX:
                break
    
    return tgt.squeeze()


class PointDataset(Dataset):
    def __init__(self, data_dir, page_indices):
        self.data_dir = Path(data_dir)
        self.page_indices = page_indices
        
    def __len__(self):
        return len(self.page_indices)
    
    def __getitem__(self, idx):
        page_idx = self.page_indices[idx]
        
        # Load points and sorted points
        points = np.loadtxt(self.data_dir / f"pg_{page_idx}_points.txt", dtype=np.int32)
        sorted_points = np.loadtxt(self.data_dir / f"pg_{page_idx}_sorted_points.txt", dtype=np.int32)
        
        # Shuffle input points
        shuffle_idx = list(range(len(points)))
        random.shuffle(shuffle_idx)
        points = points[shuffle_idx]
        
        # Truncate if necessary to ensure we don't exceed max length after tokenization
        max_points = (MAX_SEQ_LENGTH - 2) // 8  # Each point takes ~8 tokens (digits) + START/END
        if len(points) > max_points:
            points = points[:max_points]
            sorted_points = sorted_points[:max_points]
        
        # Convert coordinates to tokens
        input_tokens = self.points_to_tokens(points)
        target_tokens = self.points_to_tokens(sorted_points)
        
        return {
            'input_tokens': torch.LongTensor(input_tokens),
            'target_tokens': torch.LongTensor(target_tokens),
            'input_points': torch.FloatTensor(points),
            'target_points': torch.FloatTensor(sorted_points)
        }
    
    def points_to_tokens(self, points):
        tokens = []
        tokens.append(START_IDX)
        
        for x, y in points:
            # Convert each coordinate digit to a token
            x_tokens = [int(d) + NUM_SPECIAL_TOKENS for d in str(x)]
            y_tokens = [int(d) + NUM_SPECIAL_TOKENS for d in str(y)]
            tokens.extend(x_tokens)
            tokens.extend(y_tokens)
        
        tokens.append(END_IDX)
        
        # Pad sequence if needed
        if len(tokens) < MAX_SEQ_LENGTH:
            tokens.extend([PAD_IDX] * (MAX_SEQ_LENGTH - len(tokens)))
        
        return tokens[:MAX_SEQ_LENGTH]  # Ensure we don't exceed max length

def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    input_tokens = [item['input_tokens'] for item in batch]
    target_tokens = [item['target_tokens'] for item in batch]
    input_points = [item['input_points'] for item in batch]
    target_points = [item['target_points'] for item in batch]
    
    # Pad sequences
    input_tokens = pad_sequence(input_tokens, batch_first=True, padding_value=PAD_IDX)
    target_tokens = pad_sequence(target_tokens, batch_first=True, padding_value=PAD_IDX)
    
    # Pad points tensors
    max_points = max(p.size(0) for p in input_points)
    input_points_padded = torch.zeros(len(input_points), max_points, 2)
    target_points_padded = torch.zeros(len(target_points), max_points, 2)
    
    for i, (inp, tgt) in enumerate(zip(input_points, target_points)):
        input_points_padded[i, :inp.size(0)] = inp
        target_points_padded[i, :tgt.size(0)] = tgt
    
    return {
        'input_tokens': input_tokens,
        'target_tokens': target_tokens,
        'input_points': input_points_padded,
        'target_points': target_points_padded
    }

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_SEQ_LENGTH):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class PointSortingTransformer(nn.Module):
    def __init__(self, num_tokens, d_model=64, nhead=8, num_encoder_layers=6, 
                 num_decoder_layers=6, dim_feedforward=256):
        super().__init__()
        
        self.embedding = nn.Embedding(num_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward
        )
        
        self.fc_out = nn.Linear(d_model, num_tokens)
        
    def create_mask(self, src, tgt):
        src_padding_mask = (src == PAD_IDX)
        tgt_padding_mask = (tgt == PAD_IDX)
        
        seq_len = tgt.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(seq_len).to(tgt.device)
        
        return src_padding_mask, tgt_padding_mask, tgt_mask
        
    def forward(self, src, tgt):
        src_padding_mask, tgt_padding_mask, tgt_mask = self.create_mask(src, tgt)
        
        src_embed = self.pos_encoder(self.embedding(src).transpose(0, 1))
        tgt_embed = self.pos_encoder(self.embedding(tgt).transpose(0, 1))
        
        output = self.transformer(
            src_embed, 
            tgt_embed, 
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        return self.fc_out(output.transpose(0, 1))

def create_data_splits(num_pages, train_ratio=0.7, val_ratio=0.15):
    """Create train/val/test splits at page level"""
    indices = list(range(num_pages))
    random.shuffle(indices)
    
    train_size = int(num_pages * train_ratio)
    val_size = int(num_pages * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return train_indices, val_indices, test_indices

    

def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            src = batch['input_tokens'].to(device)
            tgt = batch['target_tokens'].to(device)
            
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                src = batch['input_tokens'].to(device)
                tgt = batch['target_tokens'].to(device)
                
                output = model(src, tgt[:, :-1])
                loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}")
        print("------------------------")

    return model


def tokens_to_points(tokens, num_special_tokens=NUM_SPECIAL_TOKENS):
    """Convert token sequence back to points"""
    points = []
    current_point = []
    
    # Skip START token and iterate until END token or padding
    for token in tokens[1:]:  # Skip START token
        if token == END_IDX or token == PAD_IDX:
            break
        
        # Convert token back to digit
        digit = token.item() - num_special_tokens
        if digit >= 0:  # Ignore special tokens
            current_point.append(str(digit))
            
        # If we have collected enough digits for a point
        if len(current_point) == 6:  # Assuming 3 digits each for x and y
            x = int(''.join(current_point[:3]))
            y = int(''.join(current_point[3:]))
            points.append([x, y])
            current_point = []
    
    return np.array(points)

def predict_sequence(model, input_tokens, device='cuda'):
    """Generate point sequence using the trained model"""
    model.eval()
    with torch.no_grad():
        # Prepare input
        src = input_tokens.unsqueeze(0).to(device)  # Add batch dimension
        
        # Initialize target sequence with START token
        tgt = torch.zeros(1, 1, dtype=torch.long, device=device)
        tgt[0, 0] = START_IDX
        
        max_len = MAX_SEQ_LENGTH

        # Generate sequence
        for i in range(max_len - 1):
            # Get prediction
            output = model(src, tgt)
            next_token = output[0, -1].argmax()
            # print(f"{i}/{MAX_SEQ_LENGTH}")
            # Append prediction to target sequence
            next_tgt = torch.zeros(1, tgt.size(1) + 1, dtype=torch.long, device=device)
            next_tgt[0, :-1] = tgt[0]
            next_tgt[0, -1] = next_token
            tgt = next_tgt
            
            # Stop if END token is predicted
            if next_token == END_IDX:
                break
    
    return tgt.squeeze()

def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model performance on test set"""
    model.eval()
    
    total_correct = 0
    total_points = 0
    
    results = []
    
    with torch.no_grad():
        for batch in test_loader:
            print("batch batch")
            src = batch['input_tokens'].to(device)
            target_points = batch['target_points']
            
            # Generate predictions for each sequence in batch
            for i in range(src.size(0)):
                input_seq = src[i]
                pred_tokens = predict_sequence(model, input_seq, device)
                pred_points = tokens_to_points(pred_tokens)

                # print("___")
                # print(input_seq)
                # print(pred_points)
                # print("___")
                
                # Get actual points (remove padding)
                target = target_points[i]
                target = target[~torch.all(target == 0, dim=1)].numpy()
                
                # Calculate accuracy (points in correct order)
                min_len = min(len(pred_points), len(target))
                correct = np.sum(np.all(pred_points[:min_len] == target[:min_len], axis=1))
                
                total_correct += correct
                total_points += min_len
                
                results.append({
                    'input_points': batch['input_points'][i].numpy(),
                    'target_points': target,
                    'pred_points': pred_points
                })
    
    accuracy = total_correct / total_points if total_points > 0 else 0
    return accuracy, results

def visualize_points(input_points, target_points, pred_points, page_idx=0):
    """Visualize input, target, and predicted point orderings"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Remove padding (zero points)
    input_points = input_points[~np.all(input_points == 0, axis=1)]
    
    # Plot input points
    ax1.scatter(input_points[:, 0], input_points[:, 1], c='blue', alpha=0.6)
    for i, (x, y) in enumerate(input_points):
        ax1.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
    ax1.set_title('Input Points')
    ax1.invert_yaxis()  # Invert y-axis to match document coordinates
    
    # Plot target points
    ax2.scatter(target_points[:, 0], target_points[:, 1], c='green', alpha=0.6)
    for i, (x, y) in enumerate(target_points):
        ax2.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
    ax2.set_title('Target Order')
    ax2.invert_yaxis()
    
    # Plot predicted points
    ax3.scatter(pred_points[:, 0], pred_points[:, 1], c='red', alpha=0.6)
    for i, (x, y) in enumerate(pred_points):
        ax3.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
    ax3.set_title('Predicted Order')
    ax3.invert_yaxis()
    
    plt.tight_layout()
    return fig


# Usage example
if __name__ == "__main__":
    data_dir = "/home/kartik/layout-analysis/data/synthetic-data/"
    num_pages = 5000
    
    # Create data splits
    train_indices, val_indices, test_indices = create_data_splits(num_pages)
    
    # Create datasets
    train_dataset = PointDataset(data_dir, train_indices)
    val_dataset = PointDataset(data_dir, val_indices)
    test_dataset = PointDataset(data_dir, test_indices)
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn)
    
    # Calculate number of tokens (0-9 + special tokens)
    num_tokens = 10 + NUM_SPECIAL_TOKENS
    
    # Initialize model
    model = PointSortingTransformer(num_tokens)
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train_model(model, train_loader, val_loader, num_epochs=10, device=device)


    accuracy, results = evaluate_model(model, test_loader, device)
    print(f"Test Set Accuracy: {accuracy:.4f}")
    
    # Visualize a few examples
    num_examples = 20
    for i in range(min(num_examples, len(results))):
        
        fig = visualize_points(
            results[i]['input_points'],
            results[i]['target_points'],
            results[i]['pred_points'],
            i
        )
        # print("vizzz")
        # print(results[i]['pred_points'])

        plt.show()
        plt.savefig(f'/home/kartik/layout-analysis/analysis_images/sorting_{i}.png')
        plt.close()
        

#input and target do not match
#prediction always constant? when margin is bigger..
#when width small - the output range is crazy
#check all individual functions

    