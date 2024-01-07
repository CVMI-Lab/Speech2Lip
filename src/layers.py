import torch
import torch.nn as nn

class DenseLayer(nn.Linear):
    def __init__(self, input_dim: int, out_dim: int, *args, activation=None, **kwargs):
        super().__init__(input_dim, out_dim, *args, **kwargs)
        if activation is None:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = activation

    def forward(self, x):
        out = super().forward(x)
        out = self.activation(out)
        return out