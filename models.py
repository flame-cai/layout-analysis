import torch
import torch.nn as nn
import torch.nn.functional as F

class PointEncoder(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        # Input: (batch_size, num_points, 2) - x,y coordinates
        self.embedding = nn.Linear(2, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=512
            ),
            num_layers=6
        )
        
    def forward(self, points):
        # points shape: (batch_size, num_points, 2)
        embedded = self.embedding(points)  # (batch_size, num_points, hidden_dim)
        encoded = self.transformer(embedded.transpose(0, 1)).transpose(0, 1)
        return encoded

class ReadingOrderPredictor(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.encoder = PointEncoder(hidden_dim)
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=512
            ),
            num_layers=6
        )
        self.output_projection = nn.Linear(hidden_dim, 1)  # Project to scalar position value
        
    def forward(self, points, max_length=None):
        # points shape: (batch_size, num_points, 2)
        if max_length is None:
            max_length = points.shape[1]
            
        encoded = self.encoder(points)
        batch_size = points.shape[0]
        
        # Initialize decoder input
        decoder_input = torch.zeros(max_length, batch_size, encoded.shape[-1]).to(points.device)
        
        # Decode positions one by one
        outputs = []
        for i in range(max_length):
            tgt_mask = self._generate_square_subsequent_mask(i+1).to(points.device)
            decoder_output = self.decoder(
                decoder_input[:i+1], 
                encoded.transpose(0, 1),
                tgt_mask=tgt_mask
            )
            position_scores = self.output_projection(decoder_output[-1])  # Get last position
            outputs.append(position_scores)
            
        return torch.cat(outputs, dim=1)
    
    @staticmethod
    def _generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    

import torch
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

class ReadingOrderGNN(torch.nn.Module):
    def __init__(self, hidden_channels=64):
        super().__init__()
        self.conv1 = GCNConv(2, hidden_channels)  # 2D coordinates to hidden
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)  # Output single value for position
        )
        
    def forward(self, x, edge_index, batch):
        # x: Node features (coordinates)
        # edge_index: Graph connectivity
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        x = self.conv3(x, edge_index)
        x = self.mlp(x)  # Project to position values
        return x

def build_graph_from_points(points, k=8):
    """Build a k-nearest neighbors graph from points"""
    from sklearn.neighbors import NearestNeighbors
    
    # Find k nearest neighbors for each point
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # Create edge indices
    edge_index = []
    for i in range(len(points)):
        for j in indices[i][1:]:  # Skip self-loop
            edge_index.append([i, j])
            edge_index.append([j, i])  # Add both directions
            
    return torch.tensor(edge_index).t().contiguous()