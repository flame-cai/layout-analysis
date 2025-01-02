import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import random
from pathlib import Path
import math
from torch.nn.utils.rnn import pad_sequence
from transformers import ByT5Tokenizer

# Initialize ByT5 tokenizer and constants
tokenizer = ByT5Tokenizer.from_pretrained("google/byt5-small")
PAD_IDX = tokenizer.pad_token_id
START_IDX = tokenizer.convert_tokens_to_ids('<')
END_IDX = tokenizer.convert_tokens_to_ids('>')
MAX_SEQ_LENGTH = 70



def points_to_tokens(points):
    """Convert points to ByT5 tokens"""
    # Convert points to string format: "x1,y1;x2,y2;..."
    point_strings = [f"{x},{y};" for x, y in points]
    point_sequence = "".join(point_strings)
    
    # Add special tokens
    point_sequence = f"<{point_sequence}>"
    
    # Tokenize using ByT5
    tokens = tokenizer.encode(point_sequence, add_special_tokens=False)
    
    # Pad sequence if needed
    if len(tokens) < MAX_SEQ_LENGTH:
        tokens.extend([PAD_IDX] * (MAX_SEQ_LENGTH - len(tokens)))
    
    return tokens[:MAX_SEQ_LENGTH]

def tokens_to_points(tokens):
    """Convert ByT5 tokens back to points"""
    # Convert tokens to string
    sequence = tokenizer.decode(tokens)
    
    # Remove special tokens and split into point strings
    sequence = sequence.strip('<>')
    if not sequence:
        return np.array([])
    
    point_strings = sequence.strip(';').split(';')
    points = []
    
    for point_str in point_strings:
        if ',' not in point_str:
            continue
        try:
            x, y = map(int, point_str.split(','))
            points.append([x, y])
        except (ValueError, IndexError):
            continue
    
    return np.array(points)

class PointDataset(Dataset):
    def __init__(self, data_dir, page_indices):
        self.data_dir = Path(data_dir)
        self.page_indices = page_indices
        
    def __len__(self):
        return len(self.page_indices)
    
    def __getitem__(self, idx):
        page_idx = self.page_indices[idx]
        
        # Load points
        points = np.loadtxt(self.data_dir / f"pg_{page_idx}_points.txt", dtype=np.int32)
        assert len(points.shape) == 2 and points.shape[1] == 2, f"Invalid points shape: {points.shape}"
        
        # Create sorted points first (original reading order)
        sorted_points = points.copy()
        
        # Create shuffled version for input using the same points
        shuffle_idx = np.arange(len(points))
        np.random.shuffle(shuffle_idx)
        shuffled_points = points[shuffle_idx].copy()
        
        # Verify point sets are identical before truncation
        # assert np.array_equal(np.sort(shuffled_points, axis=0), np.sort(sorted_points, axis=0)), \
        #     "Shuffled and sorted points don't match before truncation"
        
        # # Truncate if necessary
        # max_points = (MAX_SEQ_LENGTH - 2) // 8
        # if len(points) > max_points:
        #     # First determine which points to keep
        #     keep_indices = np.arange(max_points)
        #     sorted_points = sorted_points[:max_points]
        #     # Then truncate shuffled points to the same set
        #     shuffled_points = shuffled_points[:max_points]
        
        # Verify point sets are identical after truncation
        # assert len(shuffled_points) == len(sorted_points), \
        #     f"Length mismatch: {len(shuffled_points)} != {len(sorted_points)}"
        # assert np.array_equal(np.sort(shuffled_points, axis=0), np.sort(sorted_points, axis=0)), \
        #     "Shuffled and sorted points don't match after truncation"
        
        # Convert coordinates to tokens
        input_tokens = points_to_tokens(shuffled_points)
        target_tokens = points_to_tokens(sorted_points)
        
        # Additional verification
        assert len(input_tokens) == len(target_tokens), \
            f"Token length mismatch: {len(input_tokens)} != {len(target_tokens)}"
        
        return {
            'input_tokens': torch.LongTensor(input_tokens),
            'target_tokens': torch.LongTensor(target_tokens),
            'input_points': torch.FloatTensor(shuffled_points),
            'target_points': torch.FloatTensor(sorted_points)
        }

def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    input_tokens = [item['input_tokens'] for item in batch]
    target_tokens = [item['target_tokens'] for item in batch]
    input_points = [item['input_points'] for item in batch]
    target_points = [item['target_points'] for item in batch]
    
    # Verify point consistency for each item in batch
    for i, (inp, tgt) in enumerate(zip(input_points, target_points)):
        assert inp.size() == tgt.size(), \
            f"Size mismatch in batch item {i}: {inp.size()} != {tgt.size()}"
        assert torch.equal(torch.sort(inp, dim=0)[0], torch.sort(tgt, dim=0)[0]), \
            f"Point sets don't match in batch item {i}"
    
    # Pad sequences
    input_tokens = pad_sequence(input_tokens, batch_first=True, padding_value=PAD_IDX)
    target_tokens = pad_sequence(target_tokens, batch_first=True, padding_value=PAD_IDX)
    
    # Verify token padding
    assert input_tokens.size() == target_tokens.size(), \
        f"Padded token size mismatch: {input_tokens.size()} != {target_tokens.size()}"
    
    # Get max points for padding
    max_points = max(p.size(0) for p in input_points)
    input_points_padded = torch.zeros(len(input_points), max_points, 2)
    target_points_padded = torch.zeros(len(target_points), max_points, 2)
    
    # Pad points tensors
    for i, (inp, tgt) in enumerate(zip(input_points, target_points)):
        input_points_padded[i, :inp.size(0)] = inp
        target_points_padded[i, :tgt.size(0)] = tgt
        
        # Verify padding hasn't corrupted the points
        assert torch.equal(input_points_padded[i, :inp.size(0)], inp), \
            f"Input points corrupted during padding for batch item {i}"
        assert torch.equal(target_points_padded[i, :tgt.size(0)], tgt), \
            f"Target points corrupted during padding for batch item {i}"
    
    collated_batch = {
        'input_tokens': input_tokens,
        'target_tokens': target_tokens,
        'input_points': input_points_padded,
        'target_points': target_points_padded
    }
    
    # Final verification
    assert torch.is_tensor(collated_batch['input_points']) and torch.is_tensor(collated_batch['target_points']), \
        "Output points must be tensors"
    assert collated_batch['input_points'].size() == collated_batch['target_points'].size(), \
        "Final output size mismatch"
    
    return collated_batch

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
        return x #+ self.pe[:, :x.size(1)] disabling positional encoding

class PointSortingTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=8, num_encoder_layers=6, 
                 num_decoder_layers=6, dim_feedforward=256):
        super().__init__()
        
        num_tokens = tokenizer.vocab_size
        
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

def create_data_splits(num_pages, train_ratio=0.8, val_ratio=0.19):
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
    """Train the model"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters())
    
    best_val_loss = float('inf')
    best_model = None
    
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
        
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict().copy()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print("------------------------")
    
    # Load best model
    model.load_state_dict(best_model)
    return model

def predict_sequence(model, input_tokens, device='cuda'):
    """Generate point sequence using the trained model"""
    model.eval()
    with torch.no_grad():
        src = input_tokens.unsqueeze(0).to(device)
        
        # Initialize target sequence with START token
        tgt = torch.zeros(1, 1, dtype=torch.long, device=device)
        tgt[0, 0] = START_IDX
        
        for _ in range(MAX_SEQ_LENGTH - 1):
            output = model(src, tgt)
            next_token = output[0, -1].argmax()
            
            next_tgt = torch.zeros(1, tgt.size(1) + 1, dtype=torch.long, device=device)
            next_tgt[0, :-1] = tgt[0]
            next_tgt[0, -1] = next_token
            tgt = next_tgt
            
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
            src = batch['input_tokens'].to(device)
            target_points = batch['target_points']
            input_points = batch['input_points']
            
            for i in range(src.size(0)):
                input_seq = src[i]
                pred_tokens = predict_sequence(model, input_seq, device)
                pred_points = tokens_to_points(pred_tokens)
                
                target = target_points[i]
                target = target[~torch.all(target == 0, dim=1)].numpy()
                input_pts = input_points[i][~torch.all(input_points[i] == 0, dim=1)].numpy()
                
                # Verify point sets match
                input_set = {tuple(p) for p in input_pts}
                target_set = {tuple(p) for p in target}
                
                if input_set != target_set:
                    print("Warning: Point sets don't match")
                    continue
                
                min_len = min(len(pred_points), len(target))
                correct = np.sum(np.all(pred_points[:min_len] == target[:min_len], axis=1))
                
                total_correct += correct
                total_points += min_len
                
                results.append({
                    'input_points': input_pts,
                    'target_points': target,
                    'pred_points': pred_points
                })
    
    accuracy = total_correct / total_points if total_points > 0 else 0
    return accuracy, results

def visualize_points(input_points, target_points, pred_points, save_path=None):
    """Visualize input, target, and predicted point orderings"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Remove padding (zero points)
    input_mask = ~np.all(input_points == 0, axis=1)
    input_points = input_points[input_mask]
    target_points = target_points[~np.all(target_points == 0, axis=1)]
    
    # Verify points sets match
    input_set = {tuple(p) for p in input_points}
    target_set = {tuple(p) for p in target_points}
    pred_set = {tuple(p) for p in pred_points}
    
    assert input_set == target_set, "Input and target point sets don't match"
    print(f"Number of points: Input={len(input_points)}, Target={len(target_points)}, Predicted={len(pred_points)}")
    
    # Plot input points with arrows showing order
    ax1.scatter(input_points[:, 0], input_points[:, 1], c='blue', alpha=0.6)
    for i, (x, y) in enumerate(input_points):
        ax1.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
    ax1.set_title('Input Order')
    ax1.invert_yaxis()
    
    # Plot target points with arrows showing order
    ax2.scatter(target_points[:, 0], target_points[:, 1], c='green', alpha=0.6)
    for i, (x, y) in enumerate(target_points):
        ax2.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
    ax2.set_title('Target Order')
    ax2.invert_yaxis()
    
    # Plot predicted points with arrows showing order
    ax3.scatter(pred_points[:, 0], pred_points[:, 1], c='red', alpha=0.6)
    for i, (x, y) in enumerate(pred_points):
        ax3.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
    ax3.set_title('Predicted Order')
    ax3.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        return fig

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    data_dir = "/home/kartik/layout-analysis/data/synthetic-data"
    num_pages = 20000
    batch_size = 32
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create data splits
    train_indices, val_indices, test_indices = create_data_splits(num_pages)
    
    # Create datasets
    train_dataset = PointDataset(data_dir, train_indices)
    val_dataset = PointDataset(data_dir, val_indices)
    test_dataset = PointDataset(data_dir, test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    # Initialize model
    model = PointSortingTransformer(
        d_model=64,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=256
    )
    /home/kartik/layout-analysis/data/synthetic-data
    # Train model
    print("Starting training...")
    model = train_model(model, train_loader, val_loader, num_epochs=num_epochs, device=device)
    
    # Save trained model
    torch.save(model.state_dict(), '/home/kartik/layout-analysis/models/point_sorting_model.pth')
    print("Model saved")
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    accuracy, results = evaluate_model(model, test_loader, device)
    print(f"Test Set Accuracy: {accuracy:.4f}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    output_dir = Path("/home/kartik/layout-analysis/analysis_images/2d_sorting_viz")
    output_dir.mkdir(exist_ok=True)
    
    num_examples = min(20, len(results))
    for i in range(num_examples):
        save_path = output_dir / f"sorting_example_{i}.png"
        visualize_points(
            results[i]['input_points'],
            results[i]['target_points'],
            results[i]['pred_points'],
            save_path=str(save_path)
        )
    
    print(f"Generated {num_examples} visualizations in {output_dir}")
    
    # # Example of loading and using the trained model
    # print("\nExample of model usage:")
    # # Load model
    # loaded_model = PointSortingTransformer()
    # loaded_model.load_state_dict(torch.load('/home/kartik/layout-analysis/models/point_sorting_model.pth'))
    # loaded_model = loaded_model.to(device)
    
    # # Example points
    # example_points = np.array([
    #     [34, 168],
    #     [38, 168],
    #     [42, 169],
    #     [35, 338],
    #     [38, 340]
    # ])
    
    # # Convert to tokens
    # input_tokens = torch.LongTensor(points_to_tokens(example_points)).to(device)
    
    # # Get prediction
    # pred_tokens = predict_sequence(loaded_model, input_tokens, device)
    # pred_points = tokens_to_points(pred_tokens)
    
    # print("Input points:")
    # print(example_points)
    # print("\nPredicted sorted points:")
    # print(pred_points)