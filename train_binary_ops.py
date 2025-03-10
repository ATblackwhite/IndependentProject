import torch
import torch.nn as nn
from Grokking.utils import (get_dataset, 
                  get_model,
                  stablemax_cross_entropy,
                  evaluate)
from grokadamw.grokadamw import GrokAdamW
import random
import time
import argparse
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Train binary operations with MLP and GrokAdamW")
    
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[200, 200],
                        help='List of hidden layer sizes')
    parser.add_argument('--num_epochs', type=int, default=1500,
                        help='Number of training epochs')
    parser.add_argument('--train_fraction', type=float, default=0.3,
                        help='Fraction of data for training')
    parser.add_argument('--modulo', type=int, default=113,
                        help='Modulo value for operations')
    parser.add_argument('--input_size', type=int, default=113,
                        help='Input size for the model')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--binary_operation', type=str, default="add_mod",
                        help='Binary operation: add_mod, product_mod, or subtract_mod')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help='Device to use for training')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='Weight decay for GrokAdamW')
    parser.add_argument('--alpha_init', type=float, default=0.98,
                        help='Initial alpha for GrokAdamW')
    parser.add_argument('--lamb', type=float, default=2.0,
                        help='Lambda parameter for GrokAdamW')
    parser.add_argument('--log_frequency', type=int, default=50,
                        help='Logging frequency in epochs')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Get dataset
    train_dataset, test_dataset = get_dataset(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    # Initialize model
    model = get_model(args).to(args.device)
    
    # Initialize optimizer
    optimizer = GrokAdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        alpha_init=args.alpha_init,
        lamb=args.lamb
    )
    
    # Training loop
    best_test_acc = 0
    start_time = time.time()
    
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = stablemax_cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        if epoch % args.log_frequency == 0:
            test_loss, test_accuracy = evaluate(model, test_loader, stablemax_cross_entropy)
            print(f'Epoch {epoch:4d} | Train Loss: {avg_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.2f}%')
            
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_accuracy': test_accuracy,
                }, 'best_model.pt')
            
            print(f"Time taken for the last {args.log_frequency} epochs: {(time.time() - start_time)/60:.2f} min")
            start_time = time.time()
    
    print(f"\nTraining completed. Best test accuracy: {best_test_acc:.2f}%")

if __name__ == "__main__":
    main() 