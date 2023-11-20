import torch.nn as nn
import torch
import numpy as np


class MLP(nn.Module):
    def __init__(self, in_features, out_features, layers, activation='relu', prune_percentile=0.0):
        super(MLP, self).__init__()

        # Create the linear layers
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.num_layer = len(layers)+1
        layer_nodes = layers
        layers_in_out = np.zeros((self.num_layer, 2))
        layers_in_out[0][0] = in_features
        layers_in_out[len(layers)][1] = out_features
        for i in range(len(layers)):
            layers_in_out[i][1] = layers_in_out[i+1][0] = layer_nodes[i]
        self.linears = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        for i in range(self.num_layer):
            self.linears.append(torch.nn.Linear(int(layers_in_out[i][0]), int(layers_in_out[i][1])))
            if i < self.num_layer-1:
                if activation == 'relu':
                    self.activations.append(torch.nn.ReLU())
                elif activation == 'sigmoid':
                    self.activations.append(torch.nn.Sigmoid())
                elif activation == 'tanh':
                    self.activations.append(torch.nn.Tanh())
                else:
                    raise ValueError(f'Unknown activation function: {activation}')
        # Choose the activation function
        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = torch.nn.Tanh()
        else:
            raise ValueError(f'Unknown activation function: {activation}')

        self.out_activation = torch.nn.Tanh()

        # Initialize the weights and biases
        # for linear in self.linears:
        #     torch.nn.init.xavier_normal_(linear.weight)
        #     torch.nn.init.zeros_(linear.bias)

        # Prune weights by percentile ranking
        self.prune_percentile = prune_percentile


    def prune_weights(self):
        # Calculate the absolute values of the weights
        abs_weights = torch.abs(self.linear1.weight)

        # Calculate the percentile threshold
        percentile_threshold = np.percentile(abs_weights.detach().numpy(), self.prune_percentile)

        # Zero out the weights that are below the percentile threshold
        self.linear1.weight.masked_fill_(abs_weights < percentile_threshold, 0.0)

    def forward(self, x):
        # Pass the input through the first linear layer and activation function
        # x = self.linear1(x)
        # x = self.activation(x)

        # Pass the input through the second linear layer
        # x = self.linear2(x)

        for i, linear in enumerate(self.linears):
            x = linear(x.to(self.device))
            if i < len(self.linears)-1:
                x = self.activations[i](x)
            # if i < len(self.linears)-1:
            #     x = self.activation(x)
            # else:
            #     x = self.out_activation(x)
        # x = self.activation(x)

        return x


