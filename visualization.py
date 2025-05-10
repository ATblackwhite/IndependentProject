import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Grokking.datasets import AlgorithmicDataset
from Grokking.constants import DEFAULT_MODULO

def add(x, y, p=DEFAULT_MODULO):
    return (x + y) % p

def mul(x, y, p=DEFAULT_MODULO):
    return (x * y) % p

def visualize_dataset(dataset, operation_name, modulo):
    """Visualize algorithmic dataset"""
    # Get dataset size
    input_size = dataset.input_size
    
    # Create result matrix for heatmap
    result_matrix = np.zeros((input_size, input_size))
    
    # Fill result matrix
    idx = 0
    for x in range(input_size):
        for y in range(input_size):
            if 'div' in dataset.operation.__name__ and y == 0:
                result_matrix[x, y] = np.nan
                continue
            result_matrix[x, y] = dataset.targets[idx].item()
            idx += 1
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # 1. Heatmap visualization of operation results
    plt.subplot(2, 2, 1)
    sns.heatmap(result_matrix, annot=True, fmt=".0f", cmap="YlGnBu")
    plt.title(f"{operation_name} Operation Results (mod {modulo})")
    plt.xlabel("y")
    plt.ylabel("x")
    
    # 2. Visualize output label distribution
    plt.subplot(2, 2, 2)
    labels, counts = torch.unique(dataset.targets, return_counts=True)
    plt.bar(labels.numpy(), counts.numpy())
    plt.title("Label Distribution")
    plt.xlabel("Result Value")
    plt.ylabel("Frequency")
    
    # 3. Show one-hot encoding examples for selected inputs
    plt.subplot(2, 2, 3)
    example_idx = 0
    example_data = dataset.data[example_idx]
    x_one_hot = example_data[:input_size]
    y_one_hot = example_data[input_size:]
    
    plt.bar(range(input_size), x_one_hot.numpy(), alpha=0.5, label='x')
    plt.bar(range(input_size), y_one_hot.numpy(), alpha=0.5, label='y')
    plt.title(f"Input One-hot Encoding Example (x={x_one_hot.argmax().item()}, y={y_one_hot.argmax().item()})")
    plt.xlabel("Position")
    plt.ylabel("Value")
    plt.legend()
    
    # 4. 3D surface plot of operation results
    plt.subplot(2, 2, 4, projection='3d')
    x_indices = np.arange(input_size)
    y_indices = np.arange(input_size)
    x_mesh, y_mesh = np.meshgrid(x_indices, y_indices)
    
    ax = plt.gca()
    ax.plot_surface(x_mesh, y_mesh, result_matrix, cmap='viridis')
    ax.set_title(f"3D Surface of {operation_name} Operation Results")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Result')
    
    plt.tight_layout()
    plt.savefig(f"{operation_name}_dataset_visualization.png")
    plt.show()

if __name__ == "__main__":
    # Create addition and multiplication datasets
    add_dataset = AlgorithmicDataset(add, p=10)
    mul_dataset = AlgorithmicDataset(mul, p=10)
    
    # Visualize datasets
    visualize_dataset(add_dataset, "Addition", 10)
    visualize_dataset(mul_dataset, "Multiplication", 10)
    
    print("Visualization completed, images have been saved.") 