import os
import numpy as np
from matplotlib import pyplot as plt


def plot_examples(dataloader, N=10):
    fig, ax = plt.subplots(ncols=N, nrows=2, figsize=(N*1.5,3), tight_layout=True)
    x, y, mask = next(iter(dataloader))
    for i in range(N):
        ax[0,i].imshow(x[i][0], cmap="Greys"); ax[0,i].axis("off")
        ax[0,i].text(s=f"{y[i].numpy()}", x=10, y=10, verticalalignment="top", fontsize=14)
        ax[1,i].imshow(mask[i][0], cmap="Greys_r"); ax[1,i].axis("off")


def plot_examples_histograms(dataloader, N=4):
    fig, ax = plt.subplots(nrows=1, ncols=N, figsize=(N*2.5,1.6), tight_layout=True)
    x, y, mask = next(iter(dataloader))
    for i in range(N):
        ax[i].hist(x[i][0].reshape(-1)[:], bins=15)


