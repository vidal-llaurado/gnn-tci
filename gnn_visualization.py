"""
Visualization tool for GNN predictions on financial transaction networks.

This module provides visualization of graphs with predicted repayment edges,
highlighting the model's predictions with different colors and styles.
"""

import gravis as gv
import networkx as nx
import numpy as np
import pickle
from pathlib import Path


def convert_numpy_types(obj):
    """
    Convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, (np.integer, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8, np.uint16,
                        np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def visualize_predictions(data_path, output_file="gnn_predictions_visualization.html"):
    """
    Visualize the financial transaction network with GNN predictions.
    
    The visualization highlights predicted repayment edges:
    - Node sizes represent revenue (normalized)
    - Node labels show revenue in millions of euros
    - Node colors represent degree centrality (black=highest, grey=lowest)
    - Goods/service edges: gray
    - Predicted repayment edges: green (high probability) to yellow (low probability)
    - Edge thickness represents prediction probability
    - Edge labels show prediction probability
    
    Args:
        data_path: Path to the pickle file containing the NetworkX graph with predictions
        output_file: Output HTML file name for the visualization
    """
    # Load the graph from pickle file
    print(f"Loading graph with predictions from {data_path}...")
    with open(data_path, 'rb') as f:
        G = pickle.load(f)
    
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Count edge types
    goods_service_edges = sum(1 for _, _, d in G.edges(data=True) 
                             if d.get('edge_type') == 'goods_service')
    repayment_edges = sum(1 for _, _, d in G.edges(data=True) 
                         if d.get('edge_type') == 'repayment')
    predicted_edges = sum(1 for _, _, d in G.edges(data=True) 
                         if d.get('predicted', False))
    print(f"  - Goods/service edges: {goods_service_edges}")
    print(f"  - Repayment edges: {repayment_edges}")
    print(f"  - Predicted repayment edges: {predicted_edges}")
    print()
    
    # Get all revenues for normalization
    revenues = []
    for node in G.nodes():
        revenue = float(convert_numpy_types(G.nodes[node].get('revenue', 0.0)))
        revenues.append(revenue)
    
    # Normalize revenues for node sizes
    if revenues:
        min_revenue = min(revenues)
        max_revenue = max(revenues)
        revenue_range = max_revenue - min_revenue if max_revenue > min_revenue else 1.0
    else:
        min_revenue = 0
        revenue_range = 1.0
    
    # Prepare node styling based on financial attributes
    for node in G.nodes():
        # Get node attributes and convert numpy types
        revenue = float(convert_numpy_types(G.nodes[node].get('revenue', 0.0)))
        
        # Normalize revenue for node size (0.1 to 1.0, then scale to reasonable size)
        if revenue_range > 0:
            normalized_revenue = (revenue - min_revenue) / revenue_range
        else:
            normalized_revenue = 0.5
        # Map normalized revenue to node size (between 10 and 50)
        node_size = 10 + 40 * normalized_revenue
        
        # Label: revenue in millions of euros
        revenue_millions = revenue / 1_000_000
        node_label = f"{revenue_millions:.2f}M€"
        
        # Create tooltip with company information
        credit_rating = G.nodes[node].get('credit_rating', 0)
        company_size = G.nodes[node].get('company_size', 0)
        tooltip = (
            f"Company {node}\n"
            f"Revenue: €{revenue:,.2f} ({revenue_millions:.2f}M€)\n"
            f"Credit Rating: {credit_rating:.2f}\n"
            f"Company Size: {company_size:,.2f}"
        )
        
        # Set node attributes for gravis
        G.nodes[node]['size'] = node_size
        G.nodes[node]['label'] = node_label
        G.nodes[node]['tooltip'] = tooltip
    
    # Group edges by transaction_id to unify goods_service and repayment
    transaction_edges = {}  # transaction_id -> {'goods_service': (u, v, data), 'repayment': (u, v, data) or None}
    
    for u, v, data in G.edges(data=True):
        edge_type = data.get('edge_type', 'unknown')
        transaction_id = convert_numpy_types(data.get('transaction_id', None))
        is_predicted = data.get('predicted', False)
        
        if transaction_id is not None:
            if transaction_id not in transaction_edges:
                transaction_edges[transaction_id] = {
                    'goods_service': None, 
                    'repayment': None,
                    'is_predicted': False
                }
            
            if edge_type == 'goods_service':
                transaction_edges[transaction_id]['goods_service'] = (u, v, data)
            elif edge_type == 'repayment':
                transaction_edges[transaction_id]['repayment'] = (u, v, data)
                if is_predicted:
                    transaction_edges[transaction_id]['is_predicted'] = True
                    transaction_edges[transaction_id]['prediction_prob'] = data.get('prediction_probability', 0.5)
    
    # Create a new undirected graph for unified edges
    G_unified = nx.Graph()
    
    # Copy all nodes
    G_unified.add_nodes_from(G.nodes())
    # Copy node attributes
    for node in G.nodes():
        for key, value in G.nodes[node].items():
            G_unified.nodes[node][key] = value
    
    # Process each transaction to create unified edges
    for transaction_id, edges_info in transaction_edges.items():
        goods_edge = edges_info['goods_service']
        repayment_edge = edges_info['repayment']
        is_predicted = edges_info.get('is_predicted', False)
        prediction_prob = edges_info.get('prediction_prob', 0.5)
        
        if goods_edge is None:
            continue
        
        u_goods, v_goods, goods_data = goods_edge
        
        # Determine the nodes for the unified edge (undirected)
        node1 = min(u_goods, v_goods)
        node2 = max(u_goods, v_goods)
        
        # Get transaction amount
        transaction_amount = float(convert_numpy_types(goods_data.get('transaction_amount', 0.0)))
        
        # Determine edge color and style based on whether repayment is predicted
        if repayment_edge is None:
            # No repayment (defaulted): red to orange based on amount
            all_amounts = [float(convert_numpy_types(data.get('transaction_amount', 0.0))) 
                          for u, v, data in G.edges(data=True) 
                          if data.get('edge_type') == 'goods_service']
            if all_amounts:
                min_amount = min(all_amounts)
                max_amount = max(all_amounts)
                amount_range = max_amount - min_amount if max_amount > min_amount else 1.0
                
                if amount_range > 0:
                    normalized_amount = (transaction_amount - min_amount) / amount_range
                else:
                    normalized_amount = 0.5
            else:
                normalized_amount = 0.5
            
            # Red (more money lost) to Orange (less money lost)
            red = 255
            green = int(165 * (1 - normalized_amount))
            blue = 0
            edge_color = f"#{red:02x}{green:02x}{blue:02x}"
            edge_label = f"€{transaction_amount:,.2f}"
            edge_width = 1 + 3 * np.log10(max(transaction_amount, 1))
            edge_style = 'solid'
            
        elif is_predicted:
            # Predicted repayment: green (high prob) to yellow (low prob)
            # Green: rgb(0, 255, 0), Yellow: rgb(255, 255, 0)
            # High probability = greener, low probability = yellower
            prob = float(prediction_prob)
            red = int(255 * (1 - prob))  # 0 (green) when prob=1, 255 (yellow) when prob=0.5
            green = 255
            blue = 0
            edge_color = f"#{red:02x}{green:02x}{blue:02x}"
            edge_label = f"P={prob:.2f}"
            # Edge width based on probability (thicker = higher probability)
            edge_width = 2 + 4 * prob
            edge_style = 'dashed'  # Dashed to indicate prediction
            tooltip = (
                f"Transaction {transaction_id}\n"
                f"Amount: €{transaction_amount:,.2f}\n"
                f"Predicted Repayment (Probability: {prob:.2%})"
            )
            
        else:
            # Has actual repayment (not predicted): gray
            edge_color = '#DCDCDC'  # Gray for completed transactions
            edge_label = ""
            edge_width = 1
            edge_style = 'solid'
            tooltip = f"Transaction {transaction_id}\nAmount: €{transaction_amount:,.2f}\nCompleted"
        
        # Add edge with attributes
        edge_attrs = {
            'transaction_id': transaction_id,
            'color': edge_color,
            'width': edge_width,
            'style': edge_style,
            'label': edge_label,
            'tooltip': tooltip if 'tooltip' in locals() else f"Transaction {transaction_id}\nAmount: €{transaction_amount:,.2f}"
        }
        
        # Add label color for predicted edges
        if is_predicted:
            edge_attrs['label_color'] = edge_color
        
        G_unified.add_edge(node1, node2, **edge_attrs)
    
    # Calculate degree centrality for node coloring (black=highest, grey=lowest)
    degree_centrality = nx.degree_centrality(G_unified)
    
    # Normalize degree centrality values for color mapping
    if degree_centrality:
        centrality_values = list(degree_centrality.values())
        min_centrality = min(centrality_values)
        max_centrality = max(centrality_values)
        centrality_range = max_centrality - min_centrality if max_centrality > min_centrality else 1.0
        
        for node in G_unified.nodes():
            centrality = degree_centrality.get(node, 0.0)
            
            # Normalize centrality (0 = lowest, 1 = highest)
            if centrality_range > 0:
                normalized_centrality = (centrality - min_centrality) / centrality_range
            else:
                normalized_centrality = 0.5
            
            # Map to color: black (highest) to grey (lowest)
            grey_value = int(220 * (1 - normalized_centrality))
            node_color = f"#{grey_value:02x}{grey_value:02x}{grey_value:02x}"
            
            # Set node color and shape
            G_unified.nodes[node]['color'] = node_color
            G_unified.nodes[node]['shape'] = 'rectangle'
    
    # Convert all numpy types to native Python types for JSON serialization
    print("Converting numpy types to native Python types...")
    G_copy = G_unified.copy()
    for node in G_copy.nodes():
        for key, value in G_copy.nodes[node].items():
            G_copy.nodes[node][key] = convert_numpy_types(value)
    
    for u, v in G_copy.edges():
        for key, value in G_copy[u][v].items():
            G_copy[u][v][key] = convert_numpy_types(value)
    
    # Create visualization with gravis
    print("Creating interactive visualization...")
    fig = gv.d3(G_copy,
                graph_height=705,
                node_hover_neighborhood=True, 
                use_collision_force=True,
                node_label_data_source="label",
                node_label_size_factor=0.45,
                edge_curvature=0,
                edge_label_size_factor=0.4, 
                show_edge_label=True,
                edge_label_data_source="label")
    
    # Export to HTML file
    print(f"Exporting visualization to {output_file}...")
    fig.export_html(output_file)
    print(f"Visualization saved to {output_file}")
    
    print("Visualization created! You can open the HTML file in your browser.")
    print("\nLegend:")
    print("  - Gray edges: Completed transactions (with repayment)")
    print("  - Green-Yellow dashed edges: Predicted repayments (green=high prob, yellow=low prob)")
    print("  - Red-Orange edges: Defaulted transactions (no repayment)")
    print("  - Edge thickness: Indicates prediction probability (thicker = higher probability)")
    print("  - Edge labels: Show prediction probability (P=0.XX) for predicted edges")
    
    return fig


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize GNN predictions on financial transaction network')
    parser.add_argument('--data', type=str, default='data/test_predictions.pkl',
                       help='Path to graph file with predictions (pickle file)')
    parser.add_argument('--output', type=str, default='gnn_predictions_visualization.html',
                       help='Output HTML file name')
    
    args = parser.parse_args()
    
    if not Path(args.data).exists():
        print(f"Error: {args.data} not found!")
        print("Please run train_gnn.py with --test_data to generate predictions first.")
    else:
        visualize_predictions(args.data, args.output)

