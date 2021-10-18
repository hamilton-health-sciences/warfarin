from torch import nn


def build_mlp(input_dim, hidden_dim, output_dim, num_layers):
    layers = [
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU()
    ]
    for _ in range(num_layers - 2):
        layers += [
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        ]
    layers += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*layers)
