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
import wandb
import matplotlib.pyplot as plt
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Train binary operations with MLP and GrokAdamW")
    
    parser.add_argument('--dataset', type=str, default="algorithmic",
                        help='Dataset type: algorithmic, sparse_parity, or binary_algorithmic')
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
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Layer-wise momentum decay rate for GrokAdamW')
    parser.add_argument('--grokking_signal_decay_rate', type=float, default=0.1,
                        help='Decay rate for adjusting alpha based on the grokking signal')
    parser.add_argument('--gradient_clipping', type=float, default=1.0,
                        help='Maximum norm for gradient clipping (set to 0 to disable)')
    parser.add_argument('--log_frequency', type=int, default=50,
                        help='Logging frequency in epochs')
    parser.add_argument('--train_precision', type=int, default=32,
                        help='Floating point precision for the model and data: 16, 32, or 64. Default is 32.')
    parser.add_argument('--use_wandb', type=bool, default=True,
                        help='Whether to use Weights & Biases for logging')
    parser.add_argument('--project_name', type=str, default="binary_ops_training",
                        help='Weights & Biases project name')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Weights & Biases run name')
    
    return parser.parse_args()

def setup_matplotlib_plots():
    """创建用于记录训练过程的matplotlib图表"""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplot(1, 2, 2)
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    # 创建保存图表的目录
    os.makedirs('plots', exist_ok=True)
    
    return {
        'train_losses': [],
        'test_losses': [],
        'train_accuracies': [],
        'test_accuracies': [],
        'epochs': []
    }

def update_matplotlib_plots(tracking_data, epoch):
    """更新matplotlib图表并保存"""
    tracking_data['epochs'].append(epoch)
    
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(tracking_data['epochs'], tracking_data['train_losses'], label='Train Loss')
    plt.plot(tracking_data['epochs'], tracking_data['test_losses'], label='Test Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(tracking_data['epochs'], tracking_data['train_accuracies'], label='Train Accuracy')
    plt.plot(tracking_data['epochs'], tracking_data['test_accuracies'], label='Test Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots/training_curves_epoch_{epoch}.png')
    plt.savefig('plots/training_curves_latest.png')
    plt.close()

# Define your grokking signal function(s)
def grokking_signal_fn(training_loss: float, validation_loss: float) -> float:
    if training_loss == 0:
        return 0.0  # Avoid division by zero
    return (validation_loss - training_loss) / training_loss

# 增加另一种grokking信号函数，基于训练和测试准确率的差距
def grokking_signal_accuracy_fn(training_accuracy: float, validation_accuracy: float) -> float:
    # 训练准确率高但测试准确率低表示可能过拟合
    return max(0.0, (training_accuracy - validation_accuracy) / 100.0)  # 归一化到0-1范围

# 修改grokking信号函数的定义，使用闭包（closure）
def create_grokking_signal_fn():
    # 使用列表存储信号值，以便可以修改
    signal_data = {'train_loss': 0.0, 'test_loss': 0.0}
    
    def signal_fn():
        return grokking_signal_fn(signal_data['train_loss'], signal_data['test_loss'])
    
    # 添加更新方法
    signal_fn.update = lambda t_loss, v_loss: signal_data.update({'train_loss': t_loss, 'test_loss': v_loss})
    return signal_fn

def create_grokking_accuracy_signal_fn():
    signal_data = {'train_accuracy': 0.0, 'test_accuracy': 0.0}
    
    def signal_fn():
        return grokking_signal_accuracy_fn(signal_data['train_accuracy'], signal_data['test_accuracy'])
    
    signal_fn.update = lambda t_acc, v_acc: signal_data.update({'train_accuracy': t_acc, 'test_accuracy': v_acc})
    return signal_fn


def main():
    args = parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 初始化 Weights & Biases
    if args.use_wandb:
        run_name = args.run_name if args.run_name else f"{args.binary_operation}_mod{args.modulo}_lr{args.lr}_wd{args.weight_decay}"
        wandb.init(
            project=args.project_name,
            name=run_name,
            config={
                "binary_operation": args.binary_operation,
                "modulo": args.modulo,
                "hidden_sizes": args.hidden_sizes,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "alpha_init": args.alpha_init,
                "lamb": args.lamb,
                "gamma": args.gamma,
                "grokking_signal_decay_rate": args.grokking_signal_decay_rate,
                "gradient_clipping": args.gradient_clipping,
                "batch_size": args.batch_size,
                "train_fraction": args.train_fraction,
                "dataset": args.dataset
            }
        )
        
        # 定义自定义图表
        wandb.define_metric("train/loss")
        wandb.define_metric("test/loss")
        wandb.define_metric("train/accuracy")
        wandb.define_metric("test/accuracy")
        wandb.define_metric("epoch")
        
        # 不再使用 wandb.Plot.line_series，因为在某些版本中不可用
    
    # 创建matplotlib图表跟踪数据
    tracking_data = setup_matplotlib_plots()
    
    # 获取数据集
    train_dataset, test_dataset = get_dataset(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    # Initialize model
    model = get_model(args).to(args.device)
    
    # 创建grokking信号函数
    loss_signal_fn = create_grokking_signal_fn()
    accuracy_signal_fn = create_grokking_accuracy_signal_fn()

    # 初始化优化器，传递函数对象而不是直接传递函数
    optimizer = GrokAdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        alpha_init=args.alpha_init,
        lamb=args.lamb,
        gamma=args.gamma,
        grokking_signal_decay_rate=args.grokking_signal_decay_rate,
        gradient_clipping=args.gradient_clipping,
        grokking_signal_fns=[loss_signal_fn, accuracy_signal_fn]  # 传递函数对象
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
        
        # 计算测试集上的损失和准确率
        test_loss, test_accuracy = evaluate(model, test_loader, stablemax_cross_entropy)
            
        # 计算训练集上的损失和准确率
        train_loss, train_accuracy = evaluate(model, train_loader, stablemax_cross_entropy)
        
        # 更新信号函数中的值，而不是调用不存在的方法
        loss_signal_fn.update(train_loss, test_loss)
        accuracy_signal_fn.update(train_accuracy, test_accuracy)

        # 计算当前信号值用于记录
        signal_value1 = grokking_signal_fn(train_loss, test_loss)
        signal_value2 = grokking_signal_accuracy_fn(train_accuracy, test_accuracy)
        
        # 记录到wandb
        if args.use_wandb:
            # 直接记录指标，wandb会自动创建图表
            wandb.log({
                "epoch": epoch,
                "train/loss": avg_loss,
                "test/loss": test_loss,
                "train/accuracy": train_accuracy,
                "test/accuracy": test_accuracy,
                # 添加自定义图表数据
                "train_test_loss": {
                    "Train Loss": avg_loss,
                    "Test Loss": test_loss
                },
                "train_test_accuracy": {
                    "Train Accuracy": train_accuracy,
                    "Test Accuracy": test_accuracy
                },
                # 记录grokking信号值
                "grokking_signals": {
                    "Loss Signal": signal_value1,
                    "Accuracy Signal": signal_value2
                }
            })

        
        # 每log_frequency个epoch评估一次模型
        if epoch % args.log_frequency == 0:
            
            
            print(f'Epoch {epoch:4d} | Train Loss: {avg_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.2f}%')
            
            # 更新跟踪数据
            tracking_data['train_losses'].append(avg_loss)
            tracking_data['test_losses'].append(test_loss)
            tracking_data['train_accuracies'].append(train_accuracy)
            tracking_data['test_accuracies'].append(test_accuracy)
            
            # 更新matplotlib图表
            update_matplotlib_plots(tracking_data, epoch)
            
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_accuracy': test_accuracy,
                    'train_accuracy': train_accuracy,
                }, 'best_model.pt')
            
            print(f"Time taken for the last {args.log_frequency} epochs: {(time.time() - start_time)/60:.2f} min")
            start_time = time.time()
    
    print(f"\n训练完成。最佳测试准确率: {best_test_acc:.2f}%")
    
    # 关闭wandb
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main() 