import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
import matplotlib.pyplot as plt

# First, install visualtorch if you don't have it
# !pip install visualtorch

import visualtorch
from visualtorch.layered import layered_view

# Define the MLP model with the requested architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = Sequential(
            Linear(194, 200),  # Input layer (194) -> First hidden layer (200)
            ReLU(),
            Linear(200, 200),  # First hidden layer (200) -> Second hidden layer (200)
            ReLU(),
            Linear(200, 97)    # Second hidden layer (200) -> Output layer (97)
        )
        
    def forward(self, x):
        return self.model(x)

# Create an instance of the model
mlp = MLP()

# Print the model architecture
print(mlp)

# Generate and display the visualization using visualtorch
# You can choose between different styles: 'layered', 'graph', or 'lenet'

# Layered style visualization
layered_view(mlp, input_shape=(512, 194), draw_volume=True, to_file="mlp_layered.png", legend=True)
print("Visualization saved as 'mlp_layered.png'")
# You can also try other visualization styles:
# 1. Graph style
# from visualtorch.graph import graph_view
# graph_view(mlp, input_shape=(512, 194), to_file="mlp_graph.png")

# 2. LeNet style
# from visualtorch import lenet_view
# lenet_view(mlp, input_shape=(512, 194), to_file="mlp_lenet.png")
