"""Model definitions"""

from typing import Any, Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ComplexNN(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, num_hidden_layers: int, hidden_dim: int
    ):
        super(ComplexNN, self).__init__()
        # Input layer
        self.input_layer = nn.Linear(in_dim, hidden_dim)

        # Hidden layers
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)]
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))

        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))

        output = self.output_layer(x)
        return output


class SimpleNN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc1(x)


class ClassNet(nn.Module):
    def __init__(self, in_dim, out_dim, to_emp_softmax=True):
        super(ClassNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, out_dim)

        self.to_emp_softmax = to_emp_softmax

    def forward(self, x_in, label_proportions):
        x_in = torch.relu(self.fc1(x_in))
        x_in = torch.relu(self.fc2(x_in))
        x_in = torch.relu(self.fc3(x_in))
        x_in = self.fc4(x_in)
        if label_proportions is not None and self.to_emp_softmax:
            x_in = torch.log(own_softmax(x_in, label_proportions) + 1e-6)
        else:
            x_in = torch.log(F.softmax(x_in, dim=1) + 1e-6)
        return x_in
