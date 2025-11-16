"""
Script to generate synthetic training data for the GNN-TCI project.

This script generates a financial transaction network using the Barabasi-Albert model,
splits edges into goods/service flow and repayment, adds node and edge features,
and saves the network to the data folder.
"""

from synthetic_data import SyntheticDataTrain
import argparse


def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate synthetic training data')
    parser.add_argument('--n', type=int, default=100, 
                       help='Number of nodes (companies). Default: 100')
    parser.add_argument('--m', type=int, default=3,
                       help='Number of edges to attach from a new node. Default: 3')
    parser.add_argument('--default_prob', type=float, default=0.05,
                       help='Probability of default. Default: 0.05 (5%%)')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize the generated network')
    parser.add_argument('--filename', type=str, default='train',
                       help='Filename to save the network. Default: train')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Generating Synthetic Training Data")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  - Number of nodes (companies): {args.n}")
    print(f"  - Edges per new node (m): {args.m}")
    print(f"  - Default probability: {args.default_prob} ({args.default_prob*100:.1f}%)")
    print(f"  - Output filename: {args.filename}")
    print("=" * 60)
    print()
    
    # Generate the synthetic data
    print("Generating network...")
    data = SyntheticDataTrain(n=args.n, m=args.m, default_prob=args.default_prob)
    print("Network generated successfully!")
    print()
    
    # Print summary
    print("Network Summary:")
    print("-" * 60)
    summary = data.get_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("-" * 60)
    print()
    
    # Visualize if requested
    if args.visualize:
        print("Visualizing network...")
        data.visualize()
        print()
    
    # Save the network
    print("Saving network...")
    data.save(filename=args.filename)
    print()
    
    print("=" * 60)
    print("Data generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

