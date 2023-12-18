"""General Utils"""

import os
import pickle
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import partial
from math import isclose, log2
from typing import Any, Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader, TensorDataset

from src.model import ComplexNN


def get_device():
    # return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")
    # return torch.device("cuda")


@dataclass
class TrainResult:
    target_entropy: float
    train_class_balance: float
    train_dataset: TensorDataset = field(repr=False)
    test_dataset: TensorDataset = field(repr=False)
    cel_model: nn.Module = field(repr=False)
    mil_model: nn.Module = field(repr=False)
    cel_hist: Dict
    mil_hist: Dict

    def get_file_path(self, dir: str) -> str:
        num_classes = len(self.train_class_balance)
        data_dim = self.train_dataset.tensors[0].shape[1]
        return f"{dir}/{data_dim}feat-{num_classes}class-{self.target_entropy}entropy"

    def save(self, filepath):
        os.makedirs(filepath, exist_ok=True)

        # Save PyTorch model state_dicts
        torch.save(self.cel_model.state_dict(), f"{filepath}/cel_model.pth")
        torch.save(self.mil_model.state_dict(), f"{filepath}/mil_model.pth")

        # Save PyTorch TensorDatasets
        torch.save(self.train_dataset, f"{filepath}/train_dataset.pth")
        torch.save(self.test_dataset, f"{filepath}/test_dataset.pth")

        # Save histories
        with open(f"{filepath}/cel_hist.pkl", "wb") as f:
            pickle.dump(self.cel_hist, f)
        with open(f"{filepath}/mil_hist.pkl", "wb") as f:
            pickle.dump(self.mil_hist, f)

        # Save the rest of the dataclass fields
        dataclass_fields = {
            k: v
            for k, v in asdict(self).items()
            if k
            not in [
                "cel_model",
                "mil_model",
                "train_dataset",
                "test_dataset",
            ]
        }
        with open(f"{filepath}/dataclass_fields.pkl", "wb") as f:
            pickle.dump(dataclass_fields, f)

    @staticmethod
    def load(filepath):
        # Load the model architectures
        # cel_model = cel_model_class()
        # mil_model = mil_model_class()

        # Load the model state_dicts
        ComplexNN.load_state_dict(torch.load(f"{filepath}/cel_model.pth"))
        ComplexNN.load_state_dict(torch.load(f"{filepath}/mil_model.pth"))

        # Load the TensorDatasets
        train_dataset = torch.load(f"{filepath}/train_dataset.pth")
        test_dataset = torch.load(f"{filepath}/test_dataset.pth")

        # Load histories
        with open(f"{filepath}/cel_hist.pkl", "rb") as f:
            cel_hist = pickle.load(f)
        with open(f"{filepath}/mil_hist.pkl", "rb") as f:
            mil_hist = pickle.load(f)

        # Load the rest of the dataclass fields
        with open(f"{filepath}/dataclass_fields.pkl", "rb") as f:
            dataclass_fields = pickle.load(f)

        return SweepResult(
            cel_model=cel_model,
            mil_model=mil_model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            cel_hist=cel_hist,
            mil_hist=mil_hist,
            **dataclass_fields,
        )


# Example usage:
# sweep_result = SweepResult(target_entropy=0.5, train_class_balance=0.6, train_dataset=train_dataset, test_dataset=test_dataset, cel_model=cel_model, mil_model=mil_model)
# sweep_result.save('path/to/save/sweep_result')

# loaded_sweep_result = SweepResult.load('path/to/save/sweep_result', cel_model_class, mil_model_class)


# ---- Loss Functions ---- #
def milLoss(outputs, targets, lambda_reg=0.1):
    """
    Mutual Information Learning Loss (milLoss) implementation.

    Parameters:
    outputs (torch.Tensor): The logits output from the model. Shape (batch_size, num_classes).
    targets (torch.Tensor): The target labels. Shape (batch_size).
    lambda_reg (float): Regularization hyperparameter.

    Returns:
    torch.Tensor: Calculated milLoss.
    """
    epsilon = 1e-8
    # Cross-entropy loss (conditional entropy learning loss)
    cel_loss = F.cross_entropy(outputs, targets)

    # Regularization term (label entropy loss)
    softmax_outputs = F.softmax(outputs, dim=1)

    # Calculate label entropy
    label_entropy = -torch.sum(
        softmax_outputs * torch.log(softmax_outputs + epsilon), dim=1
    ).mean()

    # Combine losses
    loss = cel_loss + lambda_reg * label_entropy
    return loss


def own_softmax(x, label_proportions):
    if not isinstance(label_proportions, torch.Tensor):
        label_proportions = torch.tensor(label_proportions).to("cuda")
    x_exp = torch.exp(x)
    weighted_x_exp = x_exp * label_proportions
    # weighted_x_exp = x_exp
    x_exp_sum = torch.sum(weighted_x_exp, 1, keepdim=True)

    return x_exp / x_exp_sum


# ---- Training Functions ---- #


def train_epoch_callback(model, hist, epoch_loss, epoch, train_loader, test_loader):
    test_acc = eval_model(model, test_loader)
    train_acc = eval_model(model, train_loader)
    hist["train_acc"].append(train_acc)
    hist["test_acc"].append(test_acc)

    # Print loss every 10 epochs
    if epoch % 10 == 0:
        print(
            f"Epoch {epoch}, Loss:{epoch_loss:.2f}, Train_Acc:{train_acc*100:.2f} %, Test_Acc:{test_acc*100:.2f} %"
        )


def train(
    model,
    train_dataloader,
    loss_fn,
    optimizer,
    epochs=100,
    test_loader=None,
    device_name: str = None,
) -> Dict[str, List]:
    if device_name is None:
        device = get_device()

    print(f"Training on device: {device}")

    model = model.to(device)

    hist = defaultdict(list)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            try:
                outputs = model(inputs)
            except Exception as err:
                import pdb

                pdb.set_trace()
                pass
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            loss_item = loss.item()
            total_loss += loss_item

            # compute accuracy
            soft_out = F.softmax(outputs, dim=1)
            _, predicted = torch.max(soft_out, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)

        train_acc = correct_predictions / total_samples
        hist["epoch"].append(epoch)
        hist["loss"].append(total_loss)
        hist["train_acc"].append(train_acc)

        if test_loader:
            test_acc = eval_model(model, test_loader)
            hist["test_acc"].append(test_acc)
        else:
            test_acc = np.nan

        # Print loss every 10 epochs
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch},   Loss:{total_loss:.2f},   Train_Acc: {train_acc*100:.2f}%,   Test_Acc: {test_acc*100:.2f} %"
            )
    return dict(hist)


def eval_per_class_accuracy(model, data_loader, num_classes):
    device = get_device()
    model = model.to(device)

    model.eval()
    correct, total = 0, 0
    correct_per_class = [0] * num_classes
    total_per_class = [0] * num_classes

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)

            for label in range(num_classes):
                correct_labels = (predicted == targets) & (targets == label)
                correct_per_class[label] += correct_labels.sum().item()
                total_labels = targets == label
                total_per_class[label] += total_labels.sum().item()

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    overall_accuracy = correct / total
    per_class_accuracy = [
        correct_per_class[i] / total_per_class[i] if total_per_class[i] > 0 else 0
        for i in range(num_classes)
    ]

    return overall_accuracy, per_class_accuracy


def eval_model(model, data_loader):
    device = get_device()
    model = model.to(device)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = correct / total
    return accuracy
