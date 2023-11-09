import torch.nn as nn
import torch


class seMLP(nn.Module):
    def __init__(self, in_features, out_features, n_layers=1, n_nodes=None):
        super().__init__()

        if n_nodes is None:
            n_nodes = out_features

        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(in_features, n_nodes))
            in_features = n_nodes

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def prune_weights(self, threshold=0.3):
        weights = self.linear.weight
        weights = weights.flatten()

        percentile_ranks = torch.argsort(torch.abs(weights)) / len(weights)

        mask = percentile_ranks < threshold

        weights[mask] = 0

        weights = weights.reshape(self.linear.weight.shape)

        self.linear.weight = weights



