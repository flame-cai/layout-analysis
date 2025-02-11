
import torch.nn as nn

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from generate_data_1 import MAX_CLASSES
#from angle_data_loading import MAX_NO_POINTS

#NUM_CLASSES = MAX_NO_POINTS
NUM_CLASSES = MAX_CLASSES # MAX_BLOCKS
TOTAL_CLASSES = NUM_CLASSES*4 # + for each line, there are left, right, center anchors

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=NUM_CLASSES):
#         super().__init__()
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return x + self.pe[:x.size(0)]
    
# class RoPEEncoding(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         if d_model % 2 != 0:
#             raise ValueError("d_model must be even for RoPE encoding")
#         self.d_model = d_model
        
#         # Pre-compute division terms for efficiency
#         self.div_term = torch.exp(
#             torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
#         )

#     def forward(self, x, seq_dim=0):
#         """
#         Args:
#             x: Input tensor of shape (seq_len, batch_size, d_model) or (batch_size, seq_len, d_model)
#             seq_dim: Dimension containing sequence length (0 for seq-first, 1 for batch-first)
#         Returns:
#             Tensor with rotary position encodings applied
#         """
#         # Convert to seq-first if needed
#         if seq_dim == 1:
#             x = x.transpose(0, 1)
        
#         seq_len = x.shape[0]
#         device = x.device
        
#         # Move div_term to the correct device
#         div_term = self.div_term.to(device)
        
#         # Compute position angles
#         position = torch.arange(seq_len, device=device).unsqueeze(1)
#         angles = position * div_term  # (seq_len, d_model/2)
        
#         # Compute sin and cos values
#         sin = torch.sin(angles).unsqueeze(1)  # (seq_len, 1, d_model/2)
#         cos = torch.cos(angles).unsqueeze(1)  # (seq_len, 1, d_model/2)
        
#         # Split embedding into even and odd dimensions
#         x_even = x[..., ::2]  # (seq_len, batch_size, d_model/2)
#         x_odd = x[..., 1::2]  # (seq_len, batch_size, d_model/2)
        
#         # Apply rotation
#         rotated_x = torch.empty_like(x)
#         rotated_x[..., ::2] = x_even * cos - x_odd * sin
#         rotated_x[..., 1::2] = x_odd * cos + x_even * sin
        
#         # Restore original format if needed
#         if seq_dim == 1:
#             rotated_x = rotated_x.transpose(0, 1)
            
#         return rotated_x

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
        # Positional encoding
        #self.pos_encoder = RoPEEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*2)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # Output layer
        self.output = nn.Linear(d_model, num_classes)
        
    def forward(self, src):
        # src shape: [batch_size, seq_len, 2]
        
        # Embed input
        src = self.input_embed(src)  # [batch_size, seq_len, d_model]
        #src = self.embed_norm(src)    # Normalize the embedded features
        src = src.transpose(0, 1)  # [seq_len, batch_size, d_model]
        
        # Add positional encoding
        #src = self.pos_encoder(src, seq_dim=0)

        
        # Transform
        output = self.transformer_encoder(src)
        
        # Output layer
        output = output.transpose(0, 1)  # [batch_size, seq_len, d_model]
        output = self.output(output)  # [batch_size, seq_len, num_classes] 
        
        return output