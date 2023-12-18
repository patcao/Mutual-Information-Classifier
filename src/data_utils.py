"""Synthetic Data Generation Utils"""

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
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader, TensorDataset

# def generate_random_positive_semidefinite_matrix(data_dim):
#     # Generate a random matrix
#     random_matrix = torch.randn(data_dim, data_dim)

#     # Perform eigenvalue decomposition
#     eigenvalues, eigenvectors = torch.linalg.eigh(random_matrix, UPLO='L')

#     # Replace negative eigenvalues with zero
#     positive_eigenvalues = torch.clamp(eigenvalues, min=0)

#     # Reconstruct the matrix
#     positive_semidefinite_matrix = torch.mm(eigenvectors, torch.mm(torch.diag(positive_eigenvalues), eigenvectors.t()))

#     return positive_semidefinite_matrix


def generate_random_positive_semidefinite_matrix(data_dim):
    while True:
        # Generate a random lower triangular matrix
        random_matrix = torch.randn(data_dim, data_dim)
        random_matrix = torch.tril(random_matrix)

        # Make the matrix positive semidefinite using Cholesky decomposition
        try:
            chol = torch.linalg.cholesky(random_matrix, upper=False)
            positive_semidefinite_matrix = torch.mm(chol, chol.t())
            return positive_semidefinite_matrix
        except torch.linalg.LinAlgError as err:
            print(err)
            pass


def freq_entropy(frequencies):
    total = sum(frequencies)
    probabilities = [freq / total for freq in frequencies]

    entropy = 0
    for prob in probabilities:
        if prob > 0:
            entropy -= prob * np.log2(prob)

    return entropy


def prob_entropy(probabilities):
    return -sum(p * log2(p) for p in probabilities if p > 0)


def max_entropy(num_classes):
    probs = [1.0 / num_classes] * num_classes
    return prob_entropy(probs)


def generate_even_frequency(num_classes, total_count=1000):
    probs = [1.0 / num_classes] * num_classes
    return [int(p * total_count) for p in probs]


def generate_frequency_list(length, target_entropy, iterations=100, total_count=1000):
    best_freqs = None
    best_entropy_diff = float("inf")

    for _ in range(iterations):
        # Random initial probabilities
        initial_probabilities = np.random.dirichlet(np.ones(length), size=1)[0]

        # Objective function to minimize the difference from the target entropy
        def objective(probabilities):
            return (prob_entropy(probabilities) - target_entropy) ** 2

        # Constraints: sum of probabilities must be 1, and each probability >= 0
        constraints = ({"type": "eq", "fun": lambda p: sum(p) - 1},)
        bounds = [(0, 1) for _ in range(length)]

        # Minimize the objective function
        result = minimize(
            objective, initial_probabilities, bounds=bounds, constraints=constraints
        )

        # Calculate the current entropy difference
        current_entropy_diff = abs(prob_entropy(result.x) - target_entropy)

        # Update the best frequencies if this result is closer to the target entropy
        if current_entropy_diff < best_entropy_diff:
            best_entropy_diff = current_entropy_diff
            best_freqs = [int(p * total_count) for p in result.x]

        # Break early if we are close enough
        if isclose(best_entropy_diff, 0, abs_tol=1e-3):
            break

    return best_freqs


def generate_multivariate_gaussian_data(
    class_balance, num_classes, input_dim, distributions=None
):
    assert len(class_balance) == num_classes

    # Initialize lists to store distributions and data
    data = []
    labels = []

    if distributions is None:
        distributions = []
        for class_idx in range(num_classes):
            # Generate random mean and covariance matrix for each class
            mean = torch.randn(input_dim)
            covariance_matrix = torch.randn(input_dim, input_dim)
            covariance_matrix = torch.mm(
                covariance_matrix, covariance_matrix.t()
            )  # Ensure covariance matrix is positive semidefinite

            # Create a MultivariateNormal distribution with the random mean and covariance matrix
            mvn = MultivariateNormal(mean, torch.Tensor(covariance_matrix))
            distributions.append(mvn)

    for class_idx in range(num_classes):
        num_samples = class_balance[class_idx]
        mvn = distributions[class_idx]
        # Generate data samples from the distribution
        samples = mvn.sample((num_samples,))

        # Append the data, and labels to the respective lists
        data.append(samples)
        labels.extend([class_idx] * num_samples)

    # Stack the data samples and labels to create the final dataset
    data = torch.cat(data, dim=0)
    labels = torch.tensor(labels)
    dataset = TensorDataset(data, labels)
    return dataset, distributions


def generate_seperable_data(
    class_balance: List[int], num_classes, data_dim, distributions=None
):
    assert len(class_balance) == num_classes

    if distributions is None:
        # Create multivariate normal distributions for each class
        mean_unit = 2
        distributions = [
            torch.distributions.MultivariateNormal(
                torch.zeros(data_dim), torch.eye(data_dim)
            )
        ]

        for i in range(1, num_classes // 2 + 1):
            mean = torch.ones(data_dim) * i * mean_unit
            distributions.append(
                torch.distributions.MultivariateNormal(mean, torch.eye(data_dim))
            )
            distributions.append(
                torch.distributions.MultivariateNormal(-mean, torch.eye(data_dim))
            )

        distributions = distributions[:num_classes]

    # Generate training data
    data = []
    labels = []
    for class_idx in range(num_classes):
        for _ in range(class_balance[class_idx]):
            sample = distributions[class_idx].sample()
            label = torch.LongTensor([class_idx])
            data.append(sample)
            labels.append(label)

    # Create datasets and dataloaders
    data = torch.stack(data)
    labels = torch.stack(labels).squeeze()

    return TensorDataset(data, labels), distributions
