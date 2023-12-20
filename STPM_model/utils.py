import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc



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



def plot_multi_hist(ax, data, labels, title, **kwargs):
    limits = [np.min(data), np.max(data)]
    ax.hist(data[labels==0], color="tab:blue",
            range=limits, density=True, **kwargs)
    ax.hist(data[labels==1], color="tab:orange",
            range=limits, density=True, **kwargs)
    ax.set_title(title)


    
def plot_roc_curve(ax, data, labels, title, **kwargs):
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, data)
    # Compute AUC
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, **kwargs)
    ax.set_xlabel("FP rate")
    ax.set_ylabel("TP rate")
    ax.set_title(title)
    ax.text(s = f"AUROC = {roc_auc:.5f}", x=0.95, y=0.03,
            horizontalalignment='right', verticalalignment='bottom',
            transform=ax.transAxes, fontsize=10)
    


def plot_pr_curve(ax, data, labels, title, **kwargs):
    # Compute ROC curve
    prec, rec, thresholds = precision_recall_curve(labels, data, drop_intermediate=True)

    # Compute AUC
    pr_auc = auc(rec, prec)

    ax.plot(rec, prec, **kwargs)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.text(s = f"AUPR = {pr_auc:.5f}", x=0.95, y=0.03,
            horizontalalignment='right', verticalalignment='bottom',
            transform=ax.transAxes, fontsize=10)
    


def plot_results_examples(results, N=10):

    idxs_0 =  np.nonzero(results['labels']==0)[0]
    idxs_1 = np.nonzero(results['labels']==1)[0]

    vmin = np.quantile(results["anomaly_maps"].reshape(-1)[:], 0.001)
    vmax = np.quantile(results["anomaly_maps"].reshape(-1)[:], 0.999)


    # normal examples
    fig, ax = plt.subplots(ncols=N, nrows=3, figsize=(N*1.3, 3.9), tight_layout=True)
    for i in range(N):
        idx = idxs_0[i]
        ax[0,i].imshow(results["inputs"][idx][0], cmap="Greys")
        ax[0,i].text(s=f"{results['labels'][idx]}", x=10, y=10, verticalalignment="top", fontsize=14)
        ax[1,i].imshow(results["masks"][idx], cmap="Greys_r")
        ax[2,i].imshow(results["anomaly_maps"][idx], cmap="jet", vmin=vmin, vmax=vmax);
        ax[0,i].axis("off")
        ax[1,i].axis("off")
        ax[2,i].axis("off")

    # anomalous examples
    fig, ax = plt.subplots(ncols=N, nrows=3, figsize=(N*1.3, 3.9), tight_layout=True)
    for i in range(N):
        idx = idxs_1[i]
        ax[0,i].imshow(results["inputs"][idx][0], cmap="Greys")
        ax[0,i].text(s=f"{results['labels'][idx]}", x=10, y=10, verticalalignment="top", fontsize=14)
        ax[1,i].imshow(results["masks"][idx], cmap="Greys_r")
        ax[2,i].imshow(results["anomaly_maps"][idx], cmap="jet", vmin=vmin, vmax=vmax);
        ax[0,i].axis("off")
        ax[1,i].axis("off")
        ax[2,i].axis("off")