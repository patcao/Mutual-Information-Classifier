"""Plot Utils"""

import os
import random
from typing import Any, Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset


def make_pca_plot(X, Y, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Perform PCA to reduce dimensionality
    n_components = 2  # Number of principal components to retain
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Assuming you have class labels stored in 'Y'
    unique_labels = np.unique(Y)

    # Assign different colors/markers to each class
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = Y == label
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            label=f"Class {label}",
            color=colors[i],
            marker="o",
            s=30,
        )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA Plot of Data")
    ax.legend()
    ax.grid(True)


def plot_compare_test_error(cel_hist, mil_hist, axs):
    # Test Error vs Epoch
    axs.plot(
        cel_hist["epoch"],
        1 - np.array(cel_hist["test_acc"]),
        label="CE-Test Loss",
        color="C0",
        linestyle="-",
    )
    axs.plot(
        mil_hist["epoch"],
        1 - np.array(mil_hist["test_acc"]),
        label="MI-Test Loss",
        color="C1",
        linestyle="-",
    )
    axs.plot(
        cel_hist["epoch"],
        1 - np.array(cel_hist["train_acc"]),
        label="CE-Train Loss",
        color="C0",
        linestyle="--",
    )
    axs.plot(
        mil_hist["epoch"],
        1 - np.array(mil_hist["train_acc"]),
        label="MI-Train Loss",
        color="C1",
        linestyle="--",
    )
    axs.set_ylabel("Test Error %")
    axs.set_xlabel("Epochs")
    axs.set_title("Test Error vs. Epochs")
    axs.legend()


def make_data_test_err_plot(num_classes, train_dataset, cel_hist, mil_hist):
    rows, cols = 1, 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))

    # Data PCA
    make_pca_plot(*train_dataset.tensors, axes[0])

    # Label Distribution
    label_tensor = train_dataset.tensors[1]
    labels = list(range(num_classes))
    total_labels = label_tensor.shape[0]

    label_pct = [
        100 * count / total_labels
        for count in label_tensor.bincount(minlength=num_classes)
    ]

    axes[1].set_xticks(labels)
    axes[1].bar(labels, label_pct)

    axes[1].set_ylabel("Percent %")
    axes[1].set_xlabel("Class #")
    axes[1].set_title("Class Label Distribution")

    # Test Error vs Epoch
    plot_compare_test_error(cel_hist, mil_hist, axes[2])

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()
