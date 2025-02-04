import torch
import torch.nn as nn

from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


NUM_CLASSES = 9

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=NUM_CLASSES):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x #+ self.pe[:x.size(0)]

class ReadingOrderTransformer(nn.Module):
    def __init__(self, d_model=72, nhead=6, num_encoder_layers=6, num_classes=NUM_CLASSES+2):  # NUM_CLASSES + start/end tokens
        super().__init__()
        
        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # Output layer
        self.output = nn.Linear(d_model, num_classes)
        
    def forward(self, src):
        # src shape: [batch_size, seq_len, 2]
        
        # Embed input
        src = self.input_embed(src)  # [batch_size, seq_len, d_model]
        src = src.transpose(0, 1)  # [seq_len, batch_size, d_model]
        
        # Add positional encoding
        # src = self.pos_encoder(src)
        
        # Transform
        output = self.transformer_encoder(src)
        
        # Output layer
        output = output.transpose(0, 1)  # [batch_size, seq_len, d_model]
        output = self.output(output)  # [batch_size, seq_len, num_classes] 
        
        return output