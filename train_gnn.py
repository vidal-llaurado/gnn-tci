"""
Training script for the GNN repayment edge prediction model.

Usage:
    python train_gnn.py --data data/train.pkl --epochs 100 --hidden_dim 64
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from gnn_model import (
    GraphDataLoader,
    RepaymentEdgePredictor,
    train_model,
    evaluate_model,
    predict_on_test_data,
    save_predictions_to_graph
)


def main():
    parser = argparse.ArgumentParser(description='Train GNN model for repayment edge prediction')
    parser.add_argument('--data', type=str, default='data/train.pkl',
                       help='Path to training data (pickle file)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension for GNN layers')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of GNN layers')
    parser.add_argument('--gnn_type', type=str, default='GCN',
                       choices=['GCN', 'GAT', 'SAGE'],
                       help='Type of GNN layer')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to train on')
    parser.add_argument('--predict_edge_features', action='store_true',
                       help='Whether to predict repayment edge features')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='Ratio of training data to use for validation (0.0 to 1.0). Default: 0.2')
    parser.add_argument('--test_data', type=str, default=None,
                       help='Path to test data (pickle file) for making predictions. If None, only evaluates on validation set.')
    parser.add_argument('--prediction_threshold', type=float, default=0.5,
                       help='Threshold for binary classification in predictions. Default: 0.5')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    print("=" * 80)
    print("Training GNN Model for Repayment Edge Prediction")
    print("=" * 80)
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Hidden dimension: {args.hidden_dim}")
    print(f"Number of layers: {args.num_layers}")
    print(f"GNN type: {args.gnn_type}")
    print(f"Device: {args.device}")
    print(f"Predict edge features: {args.predict_edge_features}")
    print(f"Validation ratio: {args.val_ratio}")
    if args.test_data:
        print(f"Test data: {args.test_data}")
    print("=" * 80)
    print()
    
    # Load graph
    print("Loading graph...")
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Data file not found: {args.data}")
    
    G = GraphDataLoader.load_graph(args.data)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Count edges by type
    goods_service_edges = sum(1 for _, _, d in G.edges(data=True) 
                             if d.get('edge_type') == 'goods_service')
    repayment_edges = sum(1 for _, _, d in G.edges(data=True) 
                          if d.get('edge_type') == 'repayment')
    print(f"  - Goods/service edges: {goods_service_edges}")
    print(f"  - Repayment edges: {repayment_edges}")
    print()
    
    # Convert to PyG format
    print("Converting to PyTorch Geometric format...")
    data, metadata = GraphDataLoader.nx_to_pyg(G, include_repayment_edges=False)
    print(f"PyG data: {data.x.size(0)} nodes, {data.edge_index.size(1)} edges")
    print()
    
    # Split into train and validation sets
    if args.val_ratio > 0:
        print(f"Splitting repayment edges into train/val sets (val_ratio={args.val_ratio})...")
        train_metadata, val_metadata = GraphDataLoader.split_repayment_edges(
            metadata, val_ratio=args.val_ratio, random_seed=42
        )
        
        # Count edges in each set
        train_repayment_count = train_metadata['repayment_edge_index'].size(1) if train_metadata['repayment_edge_index'] is not None else 0
        val_repayment_count = val_metadata['repayment_edge_index'].size(1) if val_metadata['repayment_edge_index'] is not None else 0
        print(f"  - Training repayment edges: {train_repayment_count}")
        print(f"  - Validation repayment edges: {val_repayment_count}")
        print()
        
        # Use training metadata for training
        train_metadata_to_use = train_metadata
    else:
        # No validation split, use all data for training
        print("No validation split (val_ratio=0), using all data for training")
        print()
        train_metadata_to_use = metadata
        val_metadata = None
    
    # Initialize model
    print("Initializing model...")
    model = RepaymentEdgePredictor(
        node_feat_dim=5,
        edge_feat_dim=4,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        gnn_type=args.gnn_type,
        predict_edge_features=args.predict_edge_features,
        edge_feat_output_dim=3
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {num_params:,} parameters")
    print()
    
    # Train model
    print("Training model...")
    print("-" * 80)
    losses = train_model(
        model,
        data,
        train_metadata_to_use,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device
    )
    print("-" * 80)
    print()
    
    # Evaluate on validation set if available
    if val_metadata is not None and val_metadata['repayment_edge_index'] is not None:
        print("Evaluating on validation set...")
        val_metrics = evaluate_model(model, data, val_metadata, device=args.device, threshold=args.prediction_threshold)
        print("-" * 80)
        print("Validation Results:")
        print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall:    {val_metrics['recall']:.4f}")
        print(f"  F1 Score: {val_metrics['f1']:.4f}")
        print(f"  True positives:  {val_metrics['num_positives']}")
        print(f"  Predicted positives: {val_metrics['num_predicted_positives']}")
        
        if 'edge_feat_mse' in val_metrics:
            print(f"  Edge feature MSE: {val_metrics['edge_feat_mse']:.4f}")
            print(f"  Edge feature MAE: {val_metrics['edge_feat_mae']:.4f}")
        print("-" * 80)
        print()
    
    # Make predictions on test data if provided
    if args.test_data:
        print(f"Making predictions on test data: {args.test_data}")
        if not Path(args.test_data).exists():
            print(f"Warning: Test data file not found: {args.test_data}")
        else:
            # Load test graph
            test_G = GraphDataLoader.load_graph(args.test_data)
            print(f"Test graph loaded: {test_G.number_of_nodes()} nodes, {test_G.number_of_edges()} edges")
            
            # Count edges by type
            test_goods_service_edges = sum(1 for _, _, d in test_G.edges(data=True) 
                                          if d.get('edge_type') == 'goods_service')
            test_repayment_edges = sum(1 for _, _, d in test_G.edges(data=True) 
                                       if d.get('edge_type') == 'repayment')
            print(f"  - Goods/service edges: {test_goods_service_edges}")
            print(f"  - Repayment edges: {test_repayment_edges}")
            
            if test_repayment_edges > 0:
                print("  Warning: Test data contains repayment edges. This should not happen in test mode.")
            print()
            
            # Convert test data to PyG format
            test_data, test_metadata = GraphDataLoader.nx_to_pyg(test_G, include_repayment_edges=False)
            
            # Make predictions
            predictions = predict_on_test_data(
                model,
                test_data,
                test_metadata,
                device=args.device,
                threshold=args.prediction_threshold
            )
            
            print("-" * 80)
            print("Test Data Predictions:")
            print(f"  Number of predicted repayment edges: {len(predictions['edge_predictions'])}")
            print(f"  Average probability: {np.mean(predictions['edge_probs']):.4f}" if predictions['edge_probs'] else "  No predictions")
            print(f"  Min probability: {np.min(predictions['edge_probs']):.4f}" if predictions['edge_probs'] else "")
            print(f"  Max probability: {np.max(predictions['edge_probs']):.4f}" if predictions['edge_probs'] else "")
            
            if predictions['edge_features'] is not None:
                print(f"  Edge features predicted: {len(predictions['edge_features'])}")
            
            # Show top predictions
            if predictions['edge_predictions']:
                print()
                print("  Top 10 predicted repayment edges (by probability):")
                sorted_predictions = sorted(
                    zip(predictions['edge_predictions'], predictions['edge_probs']),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                for (src, dst), prob in sorted_predictions:
                    print(f"    ({src}, {dst}): {prob:.4f}")
            
            print("-" * 80)
            print()
            
            # Save graph with predictions
            output_graph_path = Path(args.test_data).parent / f"{Path(args.test_data).stem}_predictions.pkl"
            print(f"Saving graph with predictions to {output_graph_path}...")
            save_predictions_to_graph(test_G, predictions, str(output_graph_path))
            print()
    
    # Save model
    model_path = Path('models')
    model_path.mkdir(exist_ok=True)
    model_file = model_path / 'gnn_model.pt'
    torch.save(model.state_dict(), model_file)
    print(f"Model saved to {model_file}")
    print()
    
    print("=" * 80)
    print("Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

