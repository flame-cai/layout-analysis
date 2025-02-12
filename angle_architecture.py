
import torch.nn as nn

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from generate_data_1 import MAX_CLASSES

NUM_CLASSES = MAX_CLASSES # MAX_BLOCKS
TOTAL_CLASSES = NUM_CLASSES#*4 # + for each line, there are left, right, center anchors


class ReadingOrderTransformer(nn.Module):
    def __init__(self, d_model=120, nhead=8, num_encoder_layers=6, num_classes=TOTAL_CLASSES):  # NUM_CLASSES + start/end tokens
        super().__init__()
        
        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(13, d_model),   # here instead of 2, it will be 6 or 8.
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

        self.embed_norm = nn.LayerNorm(d_model)
        
        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*2)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # Output layer
        self.output = nn.Linear(d_model, num_classes)
        
    def forward(self, src):
        # src shape: [batch_size, seq_len, 2]
        
        # Embed input
        src = self.input_embed(src)  # [batch_size, seq_len, d_model]
        src = src.transpose(0, 1)  # [seq_len, batch_size, d_model]
        
        # Transform
        output = self.transformer_encoder(src)
        
        # Output layer
        output = output.transpose(0, 1)  # [batch_size, seq_len, d_model]
        output = self.output(output)  # [batch_size, seq_len, num_classes] 
        
        return output