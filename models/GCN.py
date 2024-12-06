"""
GCN MODEL

GCNs take a graph data structure. A graph data structure
consists of nodes and edges that connect them. Nodes
and edges can have information associated with them,
which can help in predicting data.

For our purposes, nodes will represent the
stock price, sentiment analysis and any other information
at a specific moment in time. Edges simply represent
time progression(for example week 1 node --> week 2 node).
Optionally, edge information can include "strength" between
time periods.

GCN's input layer is a feature matrix
Then, they have hidden layers, which progressively aggregate
and transform node features. the hidden layers are made up of
Graph Convolutional Layers, which preform a convolution layer,
Activation functions, such as RelU, and Pooling layers, which
merge nodes to capture structure.
The output layer produces the final node embeddings or predictions.
"""
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import NormalizeFeatures

# Example Dataset Setup
# Replace with custom graph data for stock prediction
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

# Update the dataset to simulate stock prediction features and targets
data = dataset[0]  # Use the first graph
data.y = torch.randn(data.num_nodes)  # Continuous target values (e.g., future stock prices)
data.train_mask = torch.rand(data.num_nodes) < 0.8  # 80% for training
data.test_mask = ~data.train_mask  # Remaining for testing

print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of features: {data.num_features}')


# Define the GCN for Regression
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)  # Single output for regression

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x.view(-1)  # Flatten output to match target shape


def visualize_time_series(predictions, ground_truth, time_steps):
    # Convert tensors to numpy arrays for plotting
    predictions = predictions.detach().cpu().numpy()
    ground_truth = ground_truth.detach().cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, ground_truth, label="True Prices", color="blue", linewidth=2)
    plt.plot(time_steps, predictions, label="Predicted Prices", color="orange", linestyle="--", linewidth=2)
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Prices")
    plt.title("Stock Price Predictions Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


# Initialize Model, Optimizer, and Loss
model = GCN(hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.MSELoss()  # Mean Squared Error for regression


# Training Function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # MSE Loss
    loss.backward()
    optimizer.step()
    return loss


# Testing Function
def model_test():
    model.eval()
    out = model(data.x, data.edge_index)
    test_loss = criterion(out[data.test_mask], data.y[data.test_mask])
    return test_loss.item()


# Training Loop
for epoch in range(1, 101):
    loss = train()
    if epoch % 10 == 0:
        test_loss = model_test()
        print(f'Epoch: {epoch:03d}, Train Loss: {loss:.4f}, Test Loss: {test_loss:.4f}')

# Visualizing Node Embeddings
time_steps = range(len(data.y))
model.eval()
predictions = model(data.x, data.edge_index)
visualize_time_series(predictions[data.test_mask], data.y[data.test_mask],
                      time_steps=data.test_mask.nonzero(as_tuple=True)[0].cpu().numpy())
