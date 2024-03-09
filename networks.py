import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        nb_neurons,
        depth,
        activation=nn.SiLU(),
    ):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, nb_neurons)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(nb_neurons, nb_neurons) for _ in range(depth - 1)]
        )
        self.output_layer = nn.Linear(nb_neurons, output_dim)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        return self.output_layer(x)
