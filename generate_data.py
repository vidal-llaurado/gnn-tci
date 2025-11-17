"""
Script to generate synthetic training and test data for the GNN-TCI project.

This script generates financial transaction networks using the Barabasi-Albert model:
- Training data: Includes both goods/service edges and repayment edges
- Test data: Includes only goods/service edges (repayment edges to be predicted by GNN)
"""

from synthetic_data import SyntheticDataTrain, SyntheticDataTest
import argparse


def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate synthetic training or test data for GNN-TCI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate training data (with repayment edges)
  python generate_data.py --mode train --n 100 --m 3 --default_prob 0.05
  
  # Generate test data (without repayment edges)
  python generate_data.py --mode test --n 100 --m 3
        """
    )
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test'],
                       help='Mode: train (with repayment edges) or test (without repayment edges). Default: train')
    parser.add_argument('--n', type=int, default=100, 
                       help='Number of nodes (companies). Default: 100')
    parser.add_argument('--m', type=int, default=3,
                       help='Number of edges to attach from a new node. Default: 3')
    parser.add_argument('--default_prob', type=float, default=0.05,
                       help='Probability of default (only for train mode). Default: 0.05 (5%%)')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize the generated network')
    parser.add_argument('--filename', type=str, default=None,
                       help='Filename to save the network. Default: "train" or "test" based on mode')
    
    args = parser.parse_args()
    
    # Set default filename based on mode if not provided
    if args.filename is None:
        args.filename = args.mode
    
    # Validate default_prob only needed for train mode
    if args.mode == 'test' and args.default_prob != 0.05:
        print("Warning: --default_prob is ignored in test mode (no repayment edges generated)")
    
    print("=" * 60)
    if args.mode == 'train':
        print("Generating Synthetic Training Data")
        print("=" * 60)
        print(f"Parameters:")
        print(f"  - Mode: {args.mode} (includes repayment edges)")
        print(f"  - Number of nodes (companies): {args.n}")
        print(f"  - Edges per new node (m): {args.m}")
        print(f"  - Default probability: {args.default_prob} ({args.default_prob*100:.1f}%)")
        print(f"  - Output filename: {args.filename}")
        print("=" * 60)
        print()
        
        # Generate the synthetic training data
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
        
    else:  # test mode
        print("Generating Synthetic Test Data")
        print("=" * 60)
        print(f"Parameters:")
        print(f"  - Mode: {args.mode} (NO repayment edges - to be predicted by GNN)")
        print(f"  - Number of nodes (companies): {args.n}")
        print(f"  - Edges per new node (m): {args.m}")
        print(f"  - Output filename: {args.filename}")
        print("=" * 60)
        print()
        
        # Generate the synthetic test data
        print("Generating network...")
        data = SyntheticDataTest(n=args.n, m=args.m)
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
        print("Note: This test data contains NO repayment edges.")
        print("      The GNN model will predict which repayment edges should exist.")
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

