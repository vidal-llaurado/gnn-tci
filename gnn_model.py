"""
GNN Model for Predicting Repayment Edges in Financial Transaction Networks

This module implements a Graph Neural Network to predict repayment edges given
only goods_service edges. The model learns to:
1. Predict which repayment edges should exist (binary classification)
2. Optionally predict repayment edge features (regression)

Architecture:
- Node encoder: GNN layers (GCN/GAT/GraphSAGE) to learn node embeddings
- Edge encoder: Processes goods_service edge features
- Edge predictor: Decoder that predicts repayment edge existence and features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
from torch_geometric.utils import negative_sampling, to_undirected
import numpy as np
import networkx as nx
from typing import Optional, Tuple, Dict, List
import pickle
from pathlib import Path


class RepaymentEdgePredictor(nn.Module):
    """
    GNN model for predicting repayment edges from goods_service edges.
    
    The model consists of:
    1. Node encoder: Learns node embeddings from node features and graph structure
    2. Edge encoder: Processes goods_service edge features
    3. Edge decoder: Predicts repayment edge existence and features
    """
    
    def __init__(
        self,
        node_feat_dim: int = 5,  # company_size, credit_rating, industry_sector, revenue, num_employees
        edge_feat_dim: int = 4,  # transaction_amount, transaction_type, days_until_due, interest_rate
        hidden_dim: int = 64,
        num_layers: int = 2,
        gnn_type: str = 'GCN',  # 'GCN', 'GAT', or 'SAGE'
        dropout: float = 0.2,
        predict_edge_features: bool = True,
        edge_feat_output_dim: int = 3,  # repayment_amount, payment_status, days_overdue
    ):
        """
        Initialize the RepaymentEdgePredictor model.
        
        Args:
            node_feat_dim: Dimension of node features
            edge_feat_dim: Dimension of edge features (goods_service)
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            gnn_type: Type of GNN layer ('GCN', 'GAT', or 'SAGE')
            dropout: Dropout rate
            predict_edge_features: Whether to predict repayment edge features
            edge_feat_output_dim: Dimension of predicted edge features
        """
        super(RepaymentEdgePredictor, self).__init__()
        
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.predict_edge_features = predict_edge_features
        
        # Node feature projection
        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)
        
        # GNN layers for node encoding
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if gnn_type == 'GCN':
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == 'GAT':
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            elif gnn_type == 'SAGE':
                self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        # Edge feature encoder (for goods_service edges)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge decoder for predicting repayment edge existence
        # Uses concatenated node embeddings + edge features
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),  # [src_emb, dst_emb, edge_emb]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Binary classification: repayment exists or not
        )
        
        # Edge feature predictor (for repayment edge features)
        if predict_edge_features:
            self.edge_feat_predictor = nn.Sequential(
                nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, edge_feat_output_dim)
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_label_index: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            x: Node features [num_nodes, node_feat_dim]
            edge_index: Edge indices for goods_service edges [2, num_edges]
            edge_attr: Edge features for goods_service edges [num_edges, edge_feat_dim]
            edge_label_index: Optional edge indices to predict [2, num_predictions]
                             If None, uses all possible edges
        
        Returns:
            edge_logits: Logits for repayment edge existence [num_predictions, 1]
            edge_features: Predicted repayment edge features [num_predictions, edge_feat_output_dim] or None
        """
        # Project node features
        x = self.node_proj(x)
        
        # Apply GNN layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            x = gnn_layer(x, edge_index)
            if i < len(self.gnn_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        # Encode edge features
        edge_emb = self.edge_encoder(edge_attr)
        
        # Determine which edges to predict
        if edge_label_index is None:
            # Predict for all possible edges (can be expensive for large graphs)
            # In practice, you'd want to sample candidate edges
            num_nodes = x.size(0)
            # Create all possible directed edges
            src_nodes = torch.arange(num_nodes).repeat_interleave(num_nodes)
            dst_nodes = torch.arange(num_nodes).repeat(num_nodes)
            edge_label_index = torch.stack([src_nodes, dst_nodes], dim=0)
        
        # Get source and target node embeddings
        src_emb = x[edge_label_index[0]]  # [num_predictions, hidden_dim]
        dst_emb = x[edge_label_index[1]]  # [num_predictions, hidden_dim]
        
        # For each candidate repayment edge (dst -> src), we need to find the 
        # corresponding goods_service edge (src -> dst) to get its edge features.
        # Repayment edges are the REVERSE of goods_service edges.
        edge_emb_for_pred = torch.zeros(edge_label_index.size(1), self.hidden_dim, 
                                       device=x.device)
        
        # Create a mapping from (src, dst) to edge index in edge_index
        # This maps goods_service edges: (u, v) -> edge_index
        edge_dict = {}
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            edge_dict[(src, dst)] = i
        
        # Match candidate repayment edges with corresponding goods_service edges
        # Repayment edge (v, u) corresponds to goods_service edge (u, v)
        for i in range(edge_label_index.size(1)):
            # Candidate repayment edge: (src, dst) = (v, u)
            repayment_src, repayment_dst = edge_label_index[0, i].item(), edge_label_index[1, i].item()
            # Corresponding goods_service edge: (u, v) = (repayment_dst, repayment_src)
            goods_service_edge = (repayment_dst, repayment_src)
            if goods_service_edge in edge_dict:
                edge_emb_for_pred[i] = edge_emb[edge_dict[goods_service_edge]]
        
        # Concatenate embeddings for edge prediction
        edge_input = torch.cat([src_emb, dst_emb, edge_emb_for_pred], dim=1)
        
        # Predict edge existence
        edge_logits = self.edge_decoder(edge_input)
        
        # Predict edge features if enabled
        edge_features = None
        if self.predict_edge_features:
            edge_features = self.edge_feat_predictor(edge_input)
        
        return edge_logits, edge_features


class GraphDataLoader:
    """
    Utility class to load and convert NetworkX graphs to PyTorch Geometric format.
    """
    
    @staticmethod
    def load_graph(filepath: str) -> nx.DiGraph:
        """Load a NetworkX graph from pickle file."""
        with open(filepath, 'rb') as f:
            G = pickle.load(f)
        return G
    
    @staticmethod
    def nx_to_pyg(
        G: nx.DiGraph,
        include_repayment_edges: bool = False
    ) -> Tuple[Data, Dict]:
        """
        Convert NetworkX graph to PyTorch Geometric Data object.
        
        Args:
            G: NetworkX DiGraph with node and edge features
            include_repayment_edges: If True, includes repayment edges in the graph.
                                    If False, only includes goods_service edges (for prediction).
        
        Returns:
            data: PyTorch Geometric Data object
            metadata: Dictionary with additional information (edge mappings, etc.)
        """
        # Extract node features
        node_features = []
        node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
        
        for node in G.nodes():
            features = [
                G.nodes[node].get('company_size', 0.0),
                G.nodes[node].get('credit_rating', 0.0),
                G.nodes[node].get('industry_sector', 0),
                G.nodes[node].get('revenue', 0.0),
                G.nodes[node].get('num_employees', 0.0),
            ]
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Extract goods_service edges and their features
        edge_index_list = []
        edge_attr_list = []
        edge_to_transaction = {}  # Maps (src, dst) -> transaction_id
        
        for u, v, data in G.edges(data=True):
            if data.get('edge_type') == 'goods_service':
                src_idx = node_mapping[u]
                dst_idx = node_mapping[v]
                edge_index_list.append([src_idx, dst_idx])
                
                edge_features = [
                    data.get('transaction_amount', 0.0),
                    float(data.get('transaction_type', 0)),
                    data.get('days_until_due', 0.0),
                    data.get('interest_rate', 0.0),
                ]
                edge_attr_list.append(edge_features)
                
                transaction_id = data.get('transaction_id')
                edge_to_transaction[(src_idx, dst_idx)] = transaction_id
        
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
        
        # Extract repayment edges for labels (if they exist)
        repayment_edges = []
        repayment_edge_features = []
        repayment_edge_map = {}  # Maps transaction_id -> (src, dst) for repayment
        
        for u, v, data in G.edges(data=True):
            if data.get('edge_type') == 'repayment':
                src_idx = node_mapping[u]
                dst_idx = node_mapping[v]
                transaction_id = data.get('transaction_id')
                repayment_edges.append([src_idx, dst_idx])
                repayment_edge_map[transaction_id] = (src_idx, dst_idx)
                
                # Repayment edge features
                repayment_features = [
                    data.get('repayment_amount', 0.0),
                    float(data.get('payment_status', 0)),
                    data.get('days_overdue', 0.0),
                ]
                repayment_edge_features.append(repayment_features)
        
        repayment_edge_index = torch.tensor(repayment_edges, dtype=torch.long).t().contiguous() if repayment_edges else None
        repayment_edge_attr = torch.tensor(repayment_edge_features, dtype=torch.float) if repayment_edge_features else None
        
        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        metadata = {
            'node_mapping': node_mapping,
            'edge_to_transaction': edge_to_transaction,
            'repayment_edge_index': repayment_edge_index,
            'repayment_edge_attr': repayment_edge_attr,
            'repayment_edge_map': repayment_edge_map,
        }
        
        return data, metadata
    
    @staticmethod
    def create_edge_labels(
        data: Data,
        metadata: Dict,
        negative_ratio: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create positive and negative edge labels for training.
        
        Args:
            data: PyG Data object
            metadata: Metadata dictionary from nx_to_pyg
            negative_ratio: Ratio of negative samples to positive samples
        
        Returns:
            edge_label_index: Edge indices to predict [2, num_samples]
            edge_label: Binary labels (1 for repayment exists, 0 for doesn't) [num_samples]
            edge_label_attr: Edge features for positive edges [num_positive, 3]
        """
        repayment_edge_index = metadata['repayment_edge_index']
        repayment_edge_attr = metadata['repayment_edge_attr']
        repayment_edge_map = metadata['repayment_edge_map']
        
        if repayment_edge_index is None or repayment_edge_index.size(1) == 0:
            # No repayment edges in graph
            num_nodes = data.x.size(0)
            # Sample some negative edges
            num_negatives = 100
            neg_edge_index = negative_sampling(
                data.edge_index,
                num_nodes=num_nodes,
                num_neg_samples=num_negatives
            )
            edge_label_index = neg_edge_index
            edge_label = torch.zeros(neg_edge_index.size(1), dtype=torch.float)
            edge_label_attr = None
            return edge_label_index, edge_label, edge_label_attr
        
        # Positive edges (repayment exists)
        pos_edge_index = repayment_edge_index
        num_positives = pos_edge_index.size(1)
        
        # Negative edges (repayment doesn't exist)
        # Sample negative edges that don't have repayment
        num_negatives = int(num_positives * negative_ratio)
        num_nodes = data.x.size(0)
        
        # Get all existing edges (goods_service)
        existing_edges = set()
        for i in range(data.edge_index.size(1)):
            src = data.edge_index[0, i].item()
            dst = data.edge_index[1, i].item()
            existing_edges.add((src, dst))
        
        # Sample negative edges (reverse direction of goods_service edges that don't have repayment)
        neg_candidates = []
        for src, dst in existing_edges:
            # Check if reverse edge (dst -> src) doesn't have repayment
            reverse_edge = (dst, src)
            if reverse_edge not in repayment_edge_map.values():
                neg_candidates.append(reverse_edge)
        
        if len(neg_candidates) < num_negatives:
            # If not enough candidates, use random negative sampling
            neg_edge_index = negative_sampling(
                data.edge_index,
                num_nodes=num_nodes,
                num_neg_samples=num_negatives
            )
        else:
            # Sample from candidates
            import random
            sampled_negatives = random.sample(neg_candidates, min(num_negatives, len(neg_candidates)))
            neg_edge_index = torch.tensor(sampled_negatives, dtype=torch.long).t().contiguous()
        
        # Combine positive and negative edges
        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        edge_label = torch.cat([
            torch.ones(num_positives, dtype=torch.float),
            torch.zeros(neg_edge_index.size(1), dtype=torch.float)
        ])
        
        # Edge features for positive edges only
        edge_label_attr = repayment_edge_attr if repayment_edge_attr is not None else None
        
        return edge_label_index, edge_label, edge_label_attr
    
    @staticmethod
    def split_repayment_edges(
        metadata: Dict,
        val_ratio: float = 0.2,
        random_seed: Optional[int] = None
    ) -> Tuple[Dict, Dict]:
        """
        Split repayment edges into training and validation sets.
        
        Args:
            metadata: Metadata dictionary from nx_to_pyg
            val_ratio: Ratio of repayment edges to use for validation (0.0 to 1.0)
            random_seed: Random seed for reproducibility
        
        Returns:
            train_metadata: Metadata dictionary with training repayment edges
            val_metadata: Metadata dictionary with validation repayment edges
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        
        repayment_edge_index = metadata['repayment_edge_index']
        repayment_edge_attr = metadata['repayment_edge_attr']
        repayment_edge_map = metadata['repayment_edge_map']
        
        # Create copies of metadata
        train_metadata = metadata.copy()
        val_metadata = metadata.copy()
        
        if repayment_edge_index is None or repayment_edge_index.size(1) == 0:
            # No repayment edges to split
            train_metadata['repayment_edge_index'] = None
            train_metadata['repayment_edge_attr'] = None
            train_metadata['repayment_edge_map'] = {}
            val_metadata['repayment_edge_index'] = None
            val_metadata['repayment_edge_attr'] = None
            val_metadata['repayment_edge_map'] = {}
            return train_metadata, val_metadata
        
        num_repayment_edges = repayment_edge_index.size(1)
        num_val = int(num_repayment_edges * val_ratio)
        num_train = num_repayment_edges - num_val
        
        # Randomly shuffle indices
        indices = torch.randperm(num_repayment_edges)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]
        
        # Split repayment edges
        train_repayment_edge_index = repayment_edge_index[:, train_indices]
        val_repayment_edge_index = repayment_edge_index[:, val_indices]
        
        # Split repayment edge attributes
        if repayment_edge_attr is not None:
            train_repayment_edge_attr = repayment_edge_attr[train_indices]
            val_repayment_edge_attr = repayment_edge_attr[val_indices]
        else:
            train_repayment_edge_attr = None
            val_repayment_edge_attr = None
        
        # Split repayment edge map (map transaction_id to edge)
        train_repayment_edge_map = {}
        val_repayment_edge_map = {}
        
        # Create sets for faster lookup
        train_indices_set = set(train_indices.tolist())
        val_indices_set = set(val_indices.tolist())
        
        # Create mapping from edge to index for faster lookup
        edge_to_idx = {}
        for idx in range(repayment_edge_index.size(1)):
            src = repayment_edge_index[0, idx].item()
            dst = repayment_edge_index[1, idx].item()
            edge_to_idx[(src, dst)] = idx
        
        # Split the map based on transaction IDs
        for trans_id, edge in repayment_edge_map.items():
            if edge in edge_to_idx:
                idx = edge_to_idx[edge]
                if idx in train_indices_set:
                    train_repayment_edge_map[trans_id] = edge
                elif idx in val_indices_set:
                    val_repayment_edge_map[trans_id] = edge
        
        # Update metadata
        train_metadata['repayment_edge_index'] = train_repayment_edge_index
        train_metadata['repayment_edge_attr'] = train_repayment_edge_attr
        train_metadata['repayment_edge_map'] = train_repayment_edge_map
        
        val_metadata['repayment_edge_index'] = val_repayment_edge_index
        val_metadata['repayment_edge_attr'] = val_repayment_edge_attr
        val_metadata['repayment_edge_map'] = val_repayment_edge_map
        
        return train_metadata, val_metadata


def train_model(
    model: RepaymentEdgePredictor,
    data: Data,
    metadata: Dict,
    epochs: int = 100,
    lr: float = 0.001,
    weight_decay: float = 5e-4,
    device: str = 'cpu'
) -> List[float]:
    """
    Train the GNN model.
    
    Args:
        model: RepaymentEdgePredictor model
        data: PyG Data object
        metadata: Metadata dictionary
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        device: Device to train on ('cpu' or 'cuda')
    
    Returns:
        List of training losses
    """
    model = model.to(device)
    data = data.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion_edge = nn.BCEWithLogitsLoss()
    criterion_feat = nn.MSELoss()
    
    # Create edge labels
    edge_label_index, edge_label, edge_label_attr = GraphDataLoader.create_edge_labels(
        data, metadata, negative_ratio=1.0
    )
    edge_label_index = edge_label_index.to(device)
    edge_label = edge_label.to(device)
    if edge_label_attr is not None:
        edge_label_attr = edge_label_attr.to(device)
    
    losses = []
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        edge_logits, edge_features = model(
            data.x,
            data.edge_index,
            data.edge_attr,
            edge_label_index
        )
        
        # Edge existence loss
        edge_loss = criterion_edge(edge_logits.squeeze(), edge_label)
        
        # Edge feature loss (only for positive edges)
        feat_loss = torch.tensor(0.0, device=device)
        if model.predict_edge_features and edge_label_attr is not None:
            num_positives = edge_label.sum().int().item()
            if num_positives > 0:
                pos_mask = edge_label.bool()
                pred_features = edge_features[pos_mask]
                true_features = edge_label_attr
                feat_loss = criterion_feat(pred_features, true_features)
        
        # Total loss
        total_loss = edge_loss + feat_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        losses.append(total_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4f} "
                  f"(Edge: {edge_loss.item():.4f}, Feat: {feat_loss.item():.4f})")
    
    return losses


def evaluate_model(
    model: RepaymentEdgePredictor,
    data: Data,
    metadata: Dict,
    device: str = 'cpu',
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate the model on the given data.
    
    Args:
        model: Trained RepaymentEdgePredictor model
        data: PyG Data object
        metadata: Metadata dictionary
        device: Device to evaluate on
        threshold: Threshold for binary classification
    
    Returns:
        Dictionary with evaluation metrics
    """
    model = model.to(device)
    model.eval()
    data = data.to(device)
    
    # Create edge labels
    edge_label_index, edge_label, edge_label_attr = GraphDataLoader.create_edge_labels(
        data, metadata, negative_ratio=1.0
    )
    edge_label_index = edge_label_index.to(device)
    edge_label = edge_label.to(device)
    
    with torch.no_grad():
        edge_logits, edge_features = model(
            data.x,
            data.edge_index,
            data.edge_attr,
            edge_label_index
        )
        
        # Edge existence prediction
        edge_probs = torch.sigmoid(edge_logits.squeeze())
        edge_pred = (edge_probs >= threshold).float()
        
        # Calculate metrics
        correct = (edge_pred == edge_label).sum().item()
        total = edge_label.size(0)
        accuracy = correct / total
        
        # Precision, Recall, F1
        true_positives = ((edge_pred == 1) & (edge_label == 1)).sum().item()
        false_positives = ((edge_pred == 1) & (edge_label == 0)).sum().item()
        false_negatives = ((edge_pred == 0) & (edge_label == 1)).sum().item()
        
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_positives': edge_label.sum().item(),
            'num_predicted_positives': edge_pred.sum().item(),
        }
        
        # Edge feature prediction metrics (if applicable)
        if model.predict_edge_features and edge_label_attr is not None:
            num_positives = edge_label.sum().int().item()
            if num_positives > 0:
                pos_mask = edge_label.bool()
                pred_features = edge_features[pos_mask].cpu().numpy()
                true_features = edge_label_attr.cpu().numpy()
                
                mse = np.mean((pred_features - true_features) ** 2)
                mae = np.mean(np.abs(pred_features - true_features))
                
                metrics['edge_feat_mse'] = mse
                metrics['edge_feat_mae'] = mae
        
        return metrics


def predict_on_test_data(
    model: RepaymentEdgePredictor,
    data: Data,
    metadata: Dict,
    device: str = 'cpu',
    threshold: float = 0.5,
    candidate_edges: Optional[torch.Tensor] = None
) -> Dict:
    """
    Make predictions on test data (which has no repayment edges).
    
    Args:
        model: Trained RepaymentEdgePredictor model
        data: PyG Data object (test data with only goods_service edges)
        metadata: Metadata dictionary (should have no repayment edges)
        device: Device to run on
        threshold: Threshold for binary classification
        candidate_edges: Optional edge indices to predict [2, num_candidates].
                        If None, predicts for all reverse edges of goods_service edges.
    
    Returns:
        Dictionary with predictions:
            - edge_predictions: List of predicted repayment edges [(src, dst), ...]
            - edge_probs: Probabilities for each predicted edge
            - edge_features: Predicted edge features (if enabled)
    """
    model = model.to(device)
    model.eval()
    data = data.to(device)
    
    # Determine candidate edges to predict
    if candidate_edges is None:
        # Predict for all reverse edges of goods_service edges
        # Repayment edges are reverse of goods_service edges
        candidate_edges_list = []
        existing_edges = set()
        for i in range(data.edge_index.size(1)):
            src = data.edge_index[0, i].item()
            dst = data.edge_index[1, i].item()
            existing_edges.add((src, dst))
            # Add reverse edge as candidate (repayment direction)
            candidate_edges_list.append([dst, src])
        
        if len(candidate_edges_list) > 0:
            candidate_edges = torch.tensor(candidate_edges_list, dtype=torch.long).t().contiguous()
        else:
            # No edges to predict
            return {
                'edge_predictions': [],
                'edge_probs': [],
                'edge_features': None
            }
    
    candidate_edges = candidate_edges.to(device)
    
    with torch.no_grad():
        edge_logits, edge_features = model(
            data.x,
            data.edge_index,
            data.edge_attr,
            candidate_edges
        )
        
        # Get probabilities
        edge_probs = torch.sigmoid(edge_logits.squeeze()).cpu().numpy()
        edge_pred = (edge_probs >= threshold)
        
        # Get predicted edges
        predicted_edges = []
        predicted_probs = []
        for i in range(candidate_edges.size(1)):
            if edge_pred[i]:
                src = candidate_edges[0, i].item()
                dst = candidate_edges[1, i].item()
                predicted_edges.append((src, dst))
                predicted_probs.append(float(edge_probs[i]))
        
        result = {
            'edge_predictions': predicted_edges,
            'edge_probs': predicted_probs,
            'edge_features': None
        }
        
        # Add edge features if enabled
        if model.predict_edge_features and edge_features is not None:
            predicted_features = []
            for i in range(candidate_edges.size(1)):
                if edge_pred[i]:
                    predicted_features.append(edge_features[i].cpu().numpy())
            result['edge_features'] = predicted_features
        
        return result


def save_predictions_to_graph(
    original_graph: nx.DiGraph,
    predictions: Dict,
    output_path: str
) -> nx.DiGraph:
    """
    Save predictions as repayment edges in a new graph.
    
    Args:
        original_graph: Original NetworkX graph (test data with only goods_service edges)
        predictions: Dictionary from predict_on_test_data with edge_predictions, edge_probs, edge_features
        output_path: Path to save the graph with predictions
    
    Returns:
        Graph with predicted repayment edges added
    """
    import pickle
    
    # Create a copy of the original graph
    G_with_predictions = original_graph.copy()
    
    # Get node mapping (assuming nodes are integers)
    node_list = list(G_with_predictions.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    
    # Add predicted repayment edges
    edge_predictions = predictions.get('edge_predictions', [])
    edge_probs = predictions.get('edge_probs', [])
    edge_features = predictions.get('edge_features', None)
    
    # Find the maximum transaction_id to continue numbering
    max_transaction_id = -1
    for u, v, data in G_with_predictions.edges(data=True):
        trans_id = data.get('transaction_id', -1)
        if trans_id is not None and isinstance(trans_id, (int, np.integer)):
            max_transaction_id = max(max_transaction_id, int(trans_id))
    
    # Track which goods_service edges have predicted repayments
    goods_service_to_prediction = {}
    
    for i, (src_idx, dst_idx) in enumerate(edge_predictions):
        # Convert indices back to node IDs
        src_node = idx_to_node.get(src_idx, src_idx)
        dst_node = idx_to_node.get(dst_idx, dst_idx)
        
        # Find corresponding goods_service edge (reverse direction)
        # Predicted repayment edge is (src_node, dst_node) = (original_dst, original_src)
        # Corresponding goods_service edge should be (original_src, original_dst) = (dst_node, src_node)
        goods_service_edge = None
        for u, v, data in G_with_predictions.edges(data=True):
            if u == dst_node and v == src_node and data.get('edge_type') == 'goods_service':
                goods_service_edge = (u, v, data)
                break
        
        # Get transaction_id from corresponding goods_service edge
        if goods_service_edge is not None:
            transaction_id = goods_service_edge[2].get('transaction_id')
        else:
            # Create new transaction_id if no corresponding edge found
            max_transaction_id += 1
            transaction_id = max_transaction_id
        
        # Get probability
        prob = edge_probs[i] if i < len(edge_probs) else 0.5
        
        # Create repayment edge attributes
        repayment_attrs = {
            'edge_type': 'repayment',
            'transaction_id': transaction_id,
            'predicted': True,
            'prediction_probability': float(prob)
        }
        
        # Add predicted edge features if available
        if edge_features is not None and i < len(edge_features):
            feat = edge_features[i]
            if len(feat) >= 3:
                repayment_attrs['repayment_amount'] = float(feat[0])
                repayment_attrs['payment_status'] = int(feat[1]) if len(feat) > 1 else 0
                repayment_attrs['days_overdue'] = float(feat[2]) if len(feat) > 2 else 0.0
        
        # Add the predicted repayment edge
        G_with_predictions.add_edge(src_node, dst_node, **repayment_attrs)
    
    # Save the graph
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(G_with_predictions, f)
    
    print(f"Graph with predictions saved to {output_path}")
    
    return G_with_predictions


# Example usage and discussion
if __name__ == "__main__":
    print("=" * 80)
    print("GNN Model for Repayment Edge Prediction")
    print("=" * 80)
    print()
    print("This model predicts repayment edges given only goods_service edges.")
    print()
    print("SUITABILITY OF GNNS FOR THIS TASK:")
    print("-" * 80)
    print("✓ GNNs excel at this task because:")
    print("  1. Graph structure matters: Company relationships and transaction patterns")
    print("     form meaningful neighborhoods that GNNs can learn from")
    print("  2. Node features are informative: Credit ratings, company size, revenue")
    print("     provide rich signals for predicting repayment likelihood")
    print("  3. Edge features help: Transaction amounts, interest rates, payment terms")
    print("     directly relate to repayment probability")
    print("  4. Message passing captures dependencies: A company's repayment behavior")
    print("     may depend on its neighbors' behavior (contagion effects)")
    print()
    print("LIMITATIONS TO BE AWARE OF:")
    print("-" * 80)
    print("⚠ 1. TRANSACTION PAIRING CONSTRAINT:")
    print("     - Repayment edges are paired with goods_service edges via transaction_id")
    print("     - The model should ideally enforce this constraint (current implementation")
    print("       predicts independently, but you could add a constraint loss)")
    print()
    print("⚠ 2. CLASS IMBALANCE:")
    print("     - Only ~5% default rate means most transactions have repayment edges")
    print("     - Use appropriate sampling strategies (negative sampling, focal loss)")
    print()
    print("⚠ 3. TEMPORAL ASPECTS:")
    print("     - Current model doesn't explicitly model time (days_until_due, days_overdue)")
    print("     - Consider temporal GNNs (TGAT, TGN) if time ordering matters")
    print()
    print("⚠ 4. DIRECTED GRAPH HANDLING:")
    print("     - Repayment edges are reverse of goods_service edges")
    print("     - Model should explicitly learn this directional relationship")
    print("     - Consider using directed GNN layers (e.g., GAT with separate attention")
    print("       for in/out edges)")
    print()
    print("⚠ 5. GRAPH SIZE SCALABILITY:")
    print("     - For large graphs, predicting all possible edges is expensive")
    print("     - Use negative sampling and candidate edge generation strategies")
    print()
    print("⚠ 6. GENERALIZATION:")
    print("     - Model may overfit to training graph structure")
    print("     - Test on graphs with different topologies (different BA parameters)")
    print("     - Consider graph-level features or meta-learning approaches")
    print()
    print("⚠ 7. CAUSALITY VS CORRELATION:")
    print("     - GNNs learn correlations, not necessarily causal relationships")
    print("     - For financial risk, understanding causality is important")
    print("     - Consider incorporating domain knowledge or causal inference methods")
    print()
    print("=" * 80)
    print()
    print("Example usage:")
    print("-" * 80)
    print("""
    # Load data
    from gnn_model import GraphDataLoader, RepaymentEdgePredictor, train_model, evaluate_model
    
    # Load graph
    G = GraphDataLoader.load_graph('data/train.pkl')
    
    # Convert to PyG format (without repayment edges for prediction scenario)
    data, metadata = GraphDataLoader.nx_to_pyg(G, include_repayment_edges=False)
    
    # Initialize model
    model = RepaymentEdgePredictor(
        node_feat_dim=5,
        edge_feat_dim=4,
        hidden_dim=64,
        num_layers=2,
        gnn_type='GCN',
        predict_edge_features=True
    )
    
    # Train model
    losses = train_model(model, data, metadata, epochs=100, lr=0.001)
    
    # Evaluate
    metrics = evaluate_model(model, data, metadata)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    """)
    print("=" * 80)
