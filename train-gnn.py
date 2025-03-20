import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import networkx as nx
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns

GRAPH_DATA_PATH = "/home/kartik/layout-analysis/graph-data/"
GRAPH_RESULTS_PATH = "/home/kartik/layout-analysis/graph-results/"
MODEL_PATH = "/home/kartik/layout-analysis/graph-models/"

# Define GCN model for link prediction
class GCNLinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNLinkPredictor, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.link_predictor = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, 1)
        )
        
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        row, col = edge_index
        # Get node embeddings for both source and target nodes
        z_src = z[row]
        z_dst = z[col]
        # Concatenate node embeddings
        z_combined = torch.cat([z_src, z_dst], dim=1)
        # Predict link
        return self.link_predictor(z_combined)
    
    def forward(self, x, edge_index, pred_edge_index):
        z = self.encode(x, edge_index)
        link_logits = self.decode(z, pred_edge_index)
        return link_logits

# Function to load and preprocess data from a single page
def load_page_data(page_num, data_dir='.'):
    # Load node features
    node_features_file = os.path.join(data_dir, f'node_features_page_{page_num}.npy')
    node_features = np.load(node_features_file)
    
    # Load adjacency matrix for graph structure
    adj_matrix_file = os.path.join(data_dir, f'adjacency_matrix_page_{page_num}.npy')
    adj_matrix = np.load(adj_matrix_file)
    
    # Load train and test edges
    train_edges_file = os.path.join(data_dir, f'train_edges_page_{page_num}.txt')
    test_edges_file = os.path.join(data_dir, f'test_edges_page_{page_num}.txt')
    
    # Parse edge files
    train_edges = []
    train_labels = []
    with open(train_edges_file, 'r') as f:
        for line in f:
            src, dst, label = map(int, line.strip().split())
            train_edges.append((src, dst))
            train_labels.append(label)
    
    test_edges = []
    test_labels = []
    with open(test_edges_file, 'r') as f:
        for line in f:
            src, dst, label = map(int, line.strip().split())
            test_edges.append((src, dst))
            test_labels.append(label)
    
    # Convert to PyTorch tensors
    train_edge_index = torch.tensor(train_edges, dtype=torch.long).t().contiguous()
    train_edge_label = torch.tensor(train_labels, dtype=torch.float)
    
    test_edge_index = torch.tensor(test_edges, dtype=torch.long).t().contiguous()
    test_edge_label = torch.tensor(test_labels, dtype=torch.float)
    
    # Create edge_index from adjacency matrix (for message passing)
    edge_index = torch.tensor(np.array(np.nonzero(adj_matrix)), dtype=torch.long)
    
    tobe_normalized_features = node_features
    
    # Preserve aspect ratio if node features represent coordinates
    if node_features.shape[1] == 2:  # If these are 2D coordinates
        print("here")
        # Calculate aspect ratio
        x_min, x_max = np.min(node_features[:, 0]), np.max(node_features[:, 0])
        y_min, y_max = np.min(node_features[:, 1]), np.max(node_features[:, 1])
        
        width = x_max - x_min
        height = y_max - y_min
        aspect_ratio = width / height if height != 0 else 1.0
        
        # Normalize while preserving aspect ratio
        tobe_normalized_features[:, 0] = (node_features[:, 0] - x_min) / width
        tobe_normalized_features[:, 1] = (node_features[:, 1] - y_min) / height * aspect_ratio
    
    #If we need to augment features (optional)
    if node_features.shape[1] < 8:  # If features are low-dimensional
        # Add degree as a feature
        degrees = np.sum(adj_matrix, axis=1).reshape(-1, 1)
        #normalized_degrees = scaler.fit_transform(degrees)
        
        # Create node embeddings with more features
        additional_features = np.random.normal(0, 0.1, (node_features.shape[0], 8 - node_features.shape[1]))
        tobe_normalized_features = np.hstack([tobe_normalized_features, degrees, additional_features])
    
    x = torch.tensor(tobe_normalized_features, dtype=torch.float)
    
    # Create PyG data object
    data = Data(x=x, edge_index=edge_index)
    data.train_edge_index = train_edge_index
    data.train_edge_label = train_edge_label
    data.test_edge_index = test_edge_index
    data.test_edge_label = test_edge_label
    
    return data

# Function to combine data from multiple pages
def load_all_data(data_dir='.', num_pages=1):
    all_data = []
    for page_num in range(0, num_pages):
        try:
            page_data = load_page_data(page_num, data_dir)
            all_data.append(page_data)
            print(f"Successfully loaded data for page {page_num}")
        except Exception as e:
            print(f"Could not load data for page {page_num}: {e}")
    
    return all_data

# Training function
def train_model(model, data, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    link_logits = model(data.x.to(device), 
                      data.edge_index.to(device), 
                      data.train_edge_index.to(device))
    
    loss = criterion(link_logits.squeeze(), data.train_edge_label.to(device))
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Evaluation function
def evaluate_model(model, data, criterion, device):
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        link_logits = model(data.x.to(device), 
                          data.edge_index.to(device), 
                          data.test_edge_index.to(device))
        
        loss = criterion(link_logits.squeeze(), data.test_edge_label.to(device))
        
        # Convert logits to predictions
        predictions = torch.sigmoid(link_logits).squeeze()
        preds = (predictions > 0.5).float()
        
        # Move to CPU for evaluation
        preds = preds.cpu().numpy()
        labels = data.test_edge_label.cpu().numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        
    return loss.item(), accuracy, precision, recall, f1, preds, labels

# Function to visualize test predictions
def visualize_predictions(data, pred_labels, true_labels, page_num, data_dir='.'):
    try:
        # Load original coordinate data
        node_features_file = os.path.join(data_dir, f'node_features_page_{page_num}.npy')
        node_positions = np.load(node_features_file)
        
        # Create a networkx graph for visualization
        G = nx.Graph()
        
        # Add nodes with positions
        for i in range(node_positions.shape[0]):
            G.add_node(i, pos=(node_positions[i, 0], node_positions[i, 1]))
        
        # Get test edges
        test_edges = data.test_edge_index.t().numpy()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Get positions
        pos = nx.get_node_attributes(G, 'pos')
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=30, node_color='skyblue', ax=ax1)
        nx.draw_networkx_nodes(G, pos, node_size=30, node_color='skyblue', ax=ax2)
        
        # Split edges by true labels
        true_pos_edges = test_edges[true_labels == 1]
        true_neg_edges = test_edges[true_labels == 0]
        
        # Draw edges by ground truth
        nx.draw_networkx_edges(G, pos, edgelist=[tuple(e) for e in true_pos_edges], 
                              edge_color='green', width=1.5, alpha=0.7, ax=ax1, label='True Links')
        nx.draw_networkx_edges(G, pos, edgelist=[tuple(e) for e in true_neg_edges], 
                              edge_color='red', width=1.5, alpha=0.7, ax=ax1, label='True No-Links')
        
        # Split edges by predicted labels
        pred_pos_edges = test_edges[pred_labels == 1]
        pred_neg_edges = test_edges[pred_labels == 0]
        
        # Draw edges by prediction
        nx.draw_networkx_edges(G, pos, edgelist=[tuple(e) for e in pred_pos_edges], 
                              edge_color='green', width=1.5, alpha=0.7, ax=ax2, label='Predicted Links')
        nx.draw_networkx_edges(G, pos, edgelist=[tuple(e) for e in pred_neg_edges], 
                              edge_color='red', width=1.5, alpha=0.7, ax=ax2, label='Predicted No-Links')
        
        # Set titles and legends
        ax1.set_title(f'Ground Truth - Page {page_num}')
        ax2.set_title(f'Predictions - Page {page_num}')
        
        # Create custom legends
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1, labels1, loc='upper right')
        ax2.legend(handles2, labels2, loc='upper right')
        
        # Adjust layout
        plt.tight_layout()
        plt.savefig(GRAPH_RESULTS_PATH+f'prediction_visualization_page_{page_num}.png', dpi=300)
        plt.close()
        
        # Create confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Link', 'Link'], 
                    yticklabels=['No Link', 'Link'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - Page {page_num}')
        plt.tight_layout()
        plt.savefig(GRAPH_RESULTS_PATH+f'confusion_matrix_page_{page_num}.png', dpi=300)
        plt.close()
        
        return True
    except Exception as e:
        print(f"Error visualizing predictions for page {page_num}: {e}")
        return False

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set parameters
    data_dir = GRAPH_DATA_PATH
    num_epochs = 1500
    hidden_channels = 64
    out_channels = 32
    learning_rate = 0.001
    
    # Load data
    all_data = load_all_data(data_dir=data_dir, num_pages=11)
    
    if not all_data:
        print("No data was loaded. Exiting.")
        return
    
    # Results storage
    all_results = []
    
    # Check dimensions of node features to initialize model once
    max_feature_dim = max([data.x.shape[1] for data in all_data])
    
    # Initialize model once
    model = GCNLinkPredictor(
        in_channels=max_feature_dim,
        hidden_channels=hidden_channels,
        out_channels=out_channels
    ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    
    # Initialize loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Process each page with the same model
    for page_idx, data in enumerate(all_data):
        page_num = page_idx
        print(f"\nTraining model for page {page_num}...")
        
        # Ensure feature dimensions match the model's expected input
        if data.x.shape[1] < max_feature_dim:
            # Pad features if needed
            padding = torch.zeros((data.x.shape[0], max_feature_dim - data.x.shape[1]), 
                                 dtype=torch.float)
            data.x = torch.cat([data.x, padding], dim=1)
        
        # Training loop
        for epoch in range(num_epochs):
            loss = train_model(model, data, optimizer, criterion, device)
            
            if (epoch + 1) % 20 == 0:
                test_loss, accuracy, precision, recall, f1, _, _ = evaluate_model(model, data, criterion, device)
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Final evaluation
        test_loss, accuracy, precision, recall, f1, preds, labels = evaluate_model(model, data, criterion, device)
        
        print(f"\nResults for page {page_num}:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        # Store results
        all_results.append({
            'page': page_num,
            'test_loss': test_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        # Visualize predictions
        visualize_predictions(data, preds, labels, page_num, data_dir)
    
    # Save the final model
    torch.save(model.state_dict(), MODEL_PATH+f'gcn_link_predictor_shared.pt')
    
    # Print overall results
    print("\nOverall Results:")
    avg_accuracy = np.mean([res['accuracy'] for res in all_results])
    avg_precision = np.mean([res['precision'] for res in all_results])
    avg_recall = np.mean([res['recall'] for res in all_results])
    avg_f1 = np.mean([res['f1'] for res in all_results])
    
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    
    # Visualization code remains the same
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    pages = [res['page'] for res in all_results]
    
    plt.figure(figsize=(12, 8))
    
    for i, metric in enumerate(metrics):
        values = [res[metric.lower()] for res in all_results]
        plt.subplot(2, 2, i+1)
        plt.bar(pages, values)
        plt.title(f'{metric} by Page')
        plt.xlabel('Page Number')
        plt.ylabel(metric)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('overall_metrics.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    main()