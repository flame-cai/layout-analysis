
import torch.nn as nn

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from generate_data_1 import MAX_CLASSES

# MAX_BLOCKS
TOTAL_CLASSES = MAX_CLASSES#*4 # + for each line, there are left, right, center anchors


class ReadingOrderTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=32, num_encoder_layers=6, num_classes=TOTAL_CLASSES):  # NUM_CLASSES + start/end tokens
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
        encoder_layers_1 = TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*2)
        self.transformer_encoder_1 = TransformerEncoder(encoder_layers_1, num_encoder_layers,enable_nested_tensor=True)

        # encoder_layers_2 = TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*2)
        # self.transformer_encoder_2 = TransformerEncoder(encoder_layers_2, num_encoder_layers,enable_nested_tensor=True)

        # Output layer
        self.output = nn.Linear(d_model, num_classes)
        
    def forward(self, src):
        # src shape: [batch_size, seq_len, 2]
        
        # Embed input
        src = self.input_embed(src)  # [batch_size, seq_len, d_model]
        src = src.transpose(0, 1)  # [seq_len, batch_size, d_model]


        # Encoders
        output = self.transformer_encoder_1(src)
        # Create a square subsequent mask for causal attention
        # seq_len = src.size(0)
        # mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(src.device)
        # output = self.transformer_encoder_2(output,mask=mask, is_causal=True)


        output = output.transpose(0, 1)  # [batch_size, seq_len, d_model]
        output = self.output(output)  # [batch_size, seq_len, num_classes] 
        
        return output
    
