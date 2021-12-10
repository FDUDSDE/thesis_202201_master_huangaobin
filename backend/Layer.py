import torch
from torch.nn import Linear, ReLU, BatchNorm1d, Sequential
from torch_geometric.nn import GINConv, BatchNorm, GCNConv


class GINLayer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GINLayer, self).__init__()

        self.linear1 = Linear(input_dim, hidden_dim)
        self.batch_norm1 = BatchNorm1d(hidden_dim)
        self.relu1 = ReLU()
        self.linear2 = Linear(hidden_dim, hidden_dim)

        self.linear = Sequential(self.linear1, self.batch_norm1, self.relu1, self.linear2)
        self.conv = GINConv(self.linear)
        self.batch_norm = BatchNorm1d(hidden_dim)
        self.relu = ReLU()

    def forward(self, x, edge_index_t):
        x = self.conv(x, edge_index_t)
        x = self.relu(self.batch_norm(x))
        return x


class GCNLayer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GCNLayer, self).__init__()

        self.conv = GCNConv(input_dim, hidden_dim, cached=True)
        self.batch_norm = BatchNorm(hidden_dim)
        self.relu = ReLU()

    def forward(self, x, edge_index_t):
        x = self.conv(x, edge_index_t)
        x = self.relu(self.batch_norm(x))
        return x
