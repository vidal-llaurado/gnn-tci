import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

class SyntheticDataTrain:
    def __init__(self, n, m, default_prob=0.05):
        """
        Initialize SyntheticDataTrain with a Barabasi-Albert graph.
        
        Args:
            n: Number of nodes (companies)
            m: Number of edges to attach from a new node to existing nodes
            default_prob: Probability of default (repayment edge not appearing). Default is 0.05 (5%)
        """
        self.n = n
        self.m = m
        self.default_prob = default_prob
        self.G = self._ba_graph(n, m)
        self.G = self._edge_splitting(self.G)
        self.G = self._node_features(self.G)
        self.G = self._edge_features(self.G)
    
    def _ba_graph(self, n, m):
        """
        Generate a Barabasi-Albert graph and clean it.
        
        Returns:
            NetworkX DiGraph: Cleaned directed graph
        """
        # Generate BA graph (undirected)
        G = nx.barabasi_albert_graph(n, m)
        
        # Clear self loops
        G.remove_edges_from(nx.selfloop_edges(G))
        
        # Clear isolated nodes
        G.remove_nodes_from(list(nx.isolates(G)))
        
        # Clear nodes with degree 0 (shouldn't exist after removing isolates, but just in case)
        G.remove_nodes_from([node for node, degree in G.degree() if degree == 0])
        
        # Convert to directed graph
        G = G.to_directed()
        
        return G
    
    def _edge_splitting(self, G):
        """
        Split each edge into two: goods/service flow and repayment with contrary directions.
        Repayment edges appear with probability (1 - default_prob).
        Assign transaction IDs to keep track of edge pairs.
        
        Args:
            G: NetworkX DiGraph
            
        Returns:
            NetworkX DiGraph: Graph with split edges and transaction IDs
        """
        # Create a new directed graph
        G_new = nx.DiGraph()
        
        # Copy all nodes
        G_new.add_nodes_from(G.nodes())
        
        # Track transaction IDs
        transaction_id = 0
        
        # Get original undirected edges (to avoid duplicates)
        original_edges = set()
        for u, v in G.edges():
            # Store edges in canonical form (smaller node first)
            edge_tuple = tuple(sorted([u, v]))
            if edge_tuple not in original_edges:
                original_edges.add(edge_tuple)
        
        # Split each original edge into two directed edges
        for u, v in original_edges:
            # Determine if this transaction defaults
            is_default = np.random.random() < self.default_prob
            
            # Edge 1: Goods/service flow from u to v (always present)
            G_new.add_edge(u, v, transaction_id=transaction_id, 
                          edge_type='goods_service', is_default=is_default)
            
            # Edge 2: Repayment from v to u (only if no default)
            if not is_default:
                G_new.add_edge(v, u, transaction_id=transaction_id, edge_type='repayment')
            
            transaction_id += 1
        
        return G_new
    
    def _node_features(self, G):
        """
        Generate random features for nodes (companies).
        Features follow expected distributions for financial data.
        
        Args:
            G: NetworkX DiGraph
            
        Returns:
            NetworkX DiGraph: Graph with node features
        """
        for node in G.nodes():
            # Company size (log-normal distribution, typical for company sizes)
            company_size = np.random.lognormal(mean=10, sigma=2)
            
            # Credit rating (1-10 scale, normal distribution around 5-7)
            credit_rating = np.clip(np.random.normal(6, 1.5), 1, 10)
            
            # Industry sector (categorical: 0-9)
            industry_sector = np.random.randint(0, 10)
            
            # Revenue (log-normal, correlated with company size)
            revenue = np.random.lognormal(mean=12, sigma=1.5)
            
            # Number of employees (log-normal)
            num_employees = np.random.lognormal(mean=5, sigma=1.2)
            
            # Assign features to node
            G.nodes[node]['company_size'] = company_size
            G.nodes[node]['credit_rating'] = credit_rating
            G.nodes[node]['industry_sector'] = industry_sector
            G.nodes[node]['revenue'] = revenue
            G.nodes[node]['num_employees'] = num_employees
        
        return G
    
    def _edge_features(self, G):
        """
        Generate random features for edges (transactions).
        Features follow expected distributions for transaction data.
        
        Args:
            G: NetworkX DiGraph
            
        Returns:
            NetworkX DiGraph: Graph with edge features
        """
        # First pass: process goods_service edges
        for u, v, data in G.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            
            if edge_type == 'goods_service':
                # Transaction amount (log-normal distribution)
                transaction_amount = np.random.lognormal(mean=8, sigma=1.5)
                
                # Transaction type (0: goods, 1: service)
                transaction_type = np.random.randint(0, 2)
                
                # Days until payment due (exponential distribution, mean 30 days)
                days_until_due = np.random.exponential(30)
                
                # Interest rate (normal distribution around 0.05-0.15)
                interest_rate = np.clip(np.random.normal(0.10, 0.03), 0.0, 0.30)
                
                # Assign features
                G[u][v]['transaction_amount'] = transaction_amount
                G[u][v]['transaction_type'] = transaction_type
                G[u][v]['days_until_due'] = days_until_due
                G[u][v]['interest_rate'] = interest_rate
        
        # Second pass: process repayment edges (now we can reference goods_service features)
        for u, v, data in G.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            
            if edge_type == 'repayment':
                # Repayment amount (should match or be related to transaction amount)
                # Get the corresponding goods/service edge
                transaction_id = data.get('transaction_id')
                # Find the corresponding goods/service edge
                base_amount = None
                interest_rate = 0
                for u2, v2, data2 in G.edges(data=True):
                    if (data2.get('transaction_id') == transaction_id and 
                        data2.get('edge_type') == 'goods_service'):
                        base_amount = data2.get('transaction_amount', 0)
                        interest_rate = data2.get('interest_rate', 0)
                        break
                
                if base_amount is not None:
                    # Repayment amount includes interest
                    repayment_amount = base_amount * (1 + interest_rate)
                else:
                    # Fallback if corresponding edge not found
                    repayment_amount = np.random.lognormal(mean=8, sigma=1.5)
                
                # Payment status (0: pending, 1: completed, 2: overdue)
                payment_status = np.random.choice([0, 1, 2], p=[0.3, 0.6, 0.1])
                
                # Days overdue (if overdue)
                days_overdue = max(0, np.random.exponential(15)) if payment_status == 2 else 0
                
                # Assign features
                G[u][v]['repayment_amount'] = repayment_amount
                G[u][v]['payment_status'] = payment_status
                G[u][v]['days_overdue'] = days_overdue
        
        return G
    
    def visualize(self, figsize=(12, 8), node_size=500, font_size=8):
        """
        Visualize the network.
        
        Args:
            figsize: Figure size tuple
            node_size: Size of nodes in visualization
            font_size: Font size for labels
        """
        plt.figure(figsize=figsize)
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(self.G, k=1, iterations=50)
        
        # Color nodes by credit rating
        node_colors = [self.G.nodes[node].get('credit_rating', 5) for node in self.G.nodes()]
        
        # Draw nodes
        nx.draw_networkx_nodes(self.G, pos, 
                              node_color=node_colors,
                              node_size=node_size,
                              cmap=plt.cm.RdYlGn,
                              alpha=0.8)
        
        # Draw edges with different colors for different types
        goods_edges = [(u, v) for u, v, d in self.G.edges(data=True) 
                      if d.get('edge_type') == 'goods_service']
        repayment_edges = [(u, v) for u, v, d in self.G.edges(data=True) 
                          if d.get('edge_type') == 'repayment']
        
        nx.draw_networkx_edges(self.G, pos, 
                              edgelist=goods_edges,
                              edge_color='blue',
                              alpha=0.3,
                              arrows=True,
                              arrowsize=10,
                              label='Goods/Service Flow')
        
        nx.draw_networkx_edges(self.G, pos,
                              edgelist=repayment_edges,
                              edge_color='red',
                              alpha=0.3,
                              arrows=True,
                              arrowsize=10,
                              style='dashed',
                              label='Repayment')
        
        # Draw labels
        nx.draw_networkx_labels(self.G, pos, font_size=font_size)
        
        plt.title('Financial Transaction Network\n(Blue: Goods/Service Flow, Red: Repayment)')
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def get_summary(self):
        """
        Get a summary of the generated network.
        
        Returns:
            dict: Summary statistics
        """
        num_nodes = self.G.number_of_nodes()
        num_edges = self.G.number_of_edges()
        
        # Count transaction pairs
        transaction_ids = set()
        for u, v, d in self.G.edges(data=True):
            transaction_ids.add(d.get('transaction_id'))
        
        num_transactions = len(transaction_ids)
        
        # Get feature statistics
        credit_ratings = [self.G.nodes[node].get('credit_rating', 0) for node in self.G.nodes()]
        transaction_amounts = [d.get('transaction_amount', 0) 
                              for u, v, d in self.G.edges(data=True) 
                              if d.get('edge_type') == 'goods_service']
        
        # Count defaults
        defaulted_transactions = [d.get('is_default', False) 
                                 for u, v, d in self.G.edges(data=True) 
                                 if d.get('edge_type') == 'goods_service']
        num_defaults = sum(defaulted_transactions)
        default_rate = num_defaults / num_transactions if num_transactions > 0 else 0
        
        summary = {
            'num_companies': num_nodes,
            'num_edges': num_edges,
            'num_transactions': num_transactions,
            'num_defaults': num_defaults,
            'default_rate': default_rate,
            'avg_credit_rating': np.mean(credit_ratings) if credit_ratings else 0,
            'avg_transaction_amount': np.mean(transaction_amounts) if transaction_amounts else 0,
        }
        
        return summary
    
    def save(self, filename='train', data_dir='data'):
        """
        Save the generated network to the data folder.
        
        Args:
            filename: Base filename (without extension). Default is 'train'
            data_dir: Directory to save the file. Default is 'data'
        """
        # Create data directory if it doesn't exist
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for GraphML compatibility
        G_copy = self.G.copy()
        for node in G_copy.nodes():
            for key, value in G_copy.nodes[node].items():
                if isinstance(value, (np.integer, np.floating)):
                    G_copy.nodes[node][key] = value.item()
                elif isinstance(value, np.ndarray):
                    G_copy.nodes[node][key] = value.tolist()
        
        for u, v in G_copy.edges():
            for key, value in G_copy[u][v].items():
                if isinstance(value, (np.integer, np.floating)):
                    G_copy[u][v][key] = value.item()
                elif isinstance(value, np.ndarray):
                    G_copy[u][v][key] = value.tolist()
        
        # Save as GraphML (preserves all node and edge attributes)
        filepath = data_path / f"{filename}.graphml"
        nx.write_graphml(G_copy, filepath)
        
        print(f"Network saved to {filepath}")
        
        # Also save as pickle for easy Python loading (preserves numpy types)
        import pickle
        pickle_path = data_path / f"{filename}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.G, f)
        
        print(f"Network also saved as pickle to {pickle_path}")


class SyntheticDataTest:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        # Will be implemented later
        pass