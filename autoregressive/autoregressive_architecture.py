import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
from autoregressive_synthetic_data_generator import MAX_CLASSES

class AutoregressiveReadingOrderTransformer(nn.Module):
    def __init__(self, d_model=120, nhead=8, num_layers=6):
        super().__init__()
        
        # Encoder embedding for point features (continuous input)
        self.src_embed = nn.Sequential(
            nn.Linear(13, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.src_norm = nn.LayerNorm(d_model)
        
        # Decoder embedding for target tokens (discrete indices)
        # Assume we have num_classes tokens plus one extra for start token.
        VOCAB_SIZE = MAX_CLASSES + 2  # start + padding
        NUM_OUTPUT_CLASSES = VOCAB_SIZE - 1  # only the real labels

        self.tgt_norm = nn.LayerNorm(d_model)
        self.tgt_embed = nn.Embedding(VOCAB_SIZE, d_model)
        # Save the start token
        self.start_token = MAX_CLASSES + 1

        
        
        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 2)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        # Transformer decoder (autoregressive)
        decoder_layers = TransformerDecoderLayer(d_model, nhead, dim_feedforward=d_model * 2)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_layers)
        
        # Output layer mapping decoder output to the class distribution
        self.output = nn.Linear(d_model, NUM_OUTPUT_CLASSES)
    
    def forward(self, src, tgt):
        """
        Args:
            src: [batch_size, src_seq_len, feature_dim] -- continuous point features.
            tgt: [batch_size, tgt_seq_len] -- token indices for teacher forcing.
        Returns:
            logits: [batch_size, tgt_seq_len, num_classes]
        """
        batch_size, src_seq_len, _ = src.shape
        
        # Encode source points.
        src_emb = self.src_embed(src)  # [batch_size, src_seq_len, d_model]
        src_emb = self.src_norm(src_emb)
        src_emb = src_emb.transpose(0, 1)  # [src_seq_len, batch_size, d_model]
        
        memory = self.transformer_encoder(src_emb)
        
        # Embed target tokens.
        tgt_emb = self.tgt_embed(tgt)  # [batch_size, tgt_seq_len, d_model]
        tgt_emb = self.tgt_norm(tgt_emb)
        tgt_emb = tgt_emb.transpose(0, 1)  # [tgt_seq_len, batch_size, d_model]
        
        # Create causal mask for autoregressive decoding.
        tgt_mask = self._generate_square_subsequent_mask(tgt_emb.size(0),tgt_emb.device)
        
        # Decode autoregressively.
        output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        output = output.transpose(0, 1)  # [batch_size, tgt_seq_len, d_model]
        logits = self.output(output)  # [batch_size, tgt_seq_len, num_classes]
        
        return logits
    
    def _generate_square_subsequent_mask(self, size, device):
        # Create the mask directly on the correct device with explicit dtype float32.
        mask = torch.triu(torch.ones(size, size, dtype=torch.float32, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def generate(self, src, max_len):
        """
        Autoregressive inference.
        Args:
            src: [batch_size, src_seq_len, feature_dim]
            max_len: int, maximum length to generate.
        Returns:
            preds: [batch_size, max_len] predicted token indices.
        """
        batch_size = src.shape[0]
        device = src.device
        
        # Encode the source.
        src_emb = self.src_embed(src)
        src_emb = self.src_norm(src_emb).transpose(0, 1)
        memory = self.transformer_encoder(src_emb)
        
        # Start with the start token.
        tgt = torch.full((batch_size, 1), self.start_token, dtype=torch.long, device=device)
        preds = []
        
        for _ in range(max_len):
            # Embed the current target sequence.
            tgt_emb = self.tgt_embed(tgt)
            tgt_emb = self.tgt_norm(tgt_emb).transpose(0, 1)
            tgt_mask = self._generate_square_subsequent_mask(tgt_emb.size(0),tgt_emb.device)
            out = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            out = out.transpose(0, 1)  # [batch_size, seq_len, d_model]
            logits = self.output(out)  # [batch_size, seq_len, num_classes]
            
            # Get the last token prediction.
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [batch_size, 1]
            preds.append(next_token)
            tgt = torch.cat([tgt, next_token], dim=1)  # Append predicted token.
        
        return torch.cat(preds, dim=1)
