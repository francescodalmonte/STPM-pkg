import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import json


def plot_examples(dataloader, N=10, save_to=None):
    x, y, mask = next(iter(dataloader))
    N=np.min([len(x), N])
    fig, ax = plt.subplots(ncols=N, nrows=2, figsize=(N*1.5,3), tight_layout=True)
    for i in range(N):
        ax[0,i].imshow(x[i][0], cmap="Greys_r", vmin=0., vmax=1.); ax[0,i].axis("off")
        ax[0,i].text(s=f"{y[i].numpy()}", x=10, y=10, verticalalignment="top", fontsize=14)
        ax[1,i].imshow(mask[i][0], cmap="Greys_r"); ax[1,i].axis("off")
    if save_to is not None:
        fig.savefig(save_to)



def plot_examples_histograms(dataloader, N=4, save_to=None):
    x, y, mask = next(iter(dataloader))
    N=np.min([len(x), N])
    fig, ax = plt.subplots(nrows=1, ncols=N, figsize=(N*2.5,1.6), tight_layout=True)
    for i in range(N):
        ax[i].hist(x[i][0].reshape(-1)[:], bins=15)
        ax[i].set_yticks([])
    if save_to is not None:
        fig.savefig(save_to)


def plot_training_loss(train_dict, save_to=None):
    fig, ax = plt.subplots(figsize=(5,4))
    ax.set_title("train (blue) and val (orange) losses during training")
    ax.plot(train_dict["losses_train"], c="tab:blue")
    ax.plot(train_dict["losses_val"], c="tab:orange")
    ax.set_yscale("log")
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    if save_to is not None:
        fig.savefig(save_to)


def plot_hist_on_ax(ax, data, labels, title, **kwargs):
    limits = [np.min(data), np.max(data)]
    ax.hist(data[labels==0], color="tab:blue",
            range=limits, density=True, **kwargs)
    ax.hist(data[labels==1], color="tab:orange",
            range=limits, density=True, **kwargs)
    ax.set_title(title)

def plot_multi_hist(results, save_to=None):
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8,3.5), tight_layout=True)
    plot_hist_on_ax(ax[0], results["avg_anomaly"], results["labels"], "anomaly map AVG value", bins=20, alpha=0.5)
    plot_hist_on_ax(ax[1], results["anomaly_peak"], results["labels"], "anomaly map PEAK value", bins=20, alpha=0.5)
    if save_to is not None:
        fig.savefig(save_to)


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
    

def plot_multi_curves(results, save_to=None):
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(8,7), tight_layout=True)
    plot_roc_curve(ax[0,0], results["avg_anomaly"], results["labels"], "anomaly map AVG value")
    plot_roc_curve(ax[0,1], results["anomaly_peak"], results["labels"], "anomaly map PEAK value")
    plot_pr_curve(ax[1,0], results["avg_anomaly"], results["labels"], "anomaly map AVG value")
    plot_pr_curve(ax[1,1], results["anomaly_peak"], results["labels"], "anomaly map PEAK value")
    if save_to is not None:
        fig.savefig(save_to)



def plot_results_examples(results, N=10, save_to=None):

    idxs_0 =  np.nonzero(results['labels']==0)[0]
    idxs_1 = np.nonzero(results['labels']==1)[0]

    vmin = np.quantile(results["anomaly_maps"].reshape(-1)[:], 0.001)
    vmax = np.quantile(results["anomaly_maps"].reshape(-1)[:], 0.999)


    # normal examples
    N_0 = np.min([N, len(idxs_0)])
    fig1, ax1 = plt.subplots(ncols=N_0, nrows=3, figsize=(N_0*1.3, 3.9), tight_layout=True)
    for i in range(N_0):
        idx = idxs_0[i]
        ax1[0,i].imshow(results["inputs"][idx][0], cmap="Greys_r", vmin=0., vmax=1.)
        ax1[0,i].text(s=f"{results['labels'][idx]}", x=10, y=10, verticalalignment="top", fontsize=14)
        ax1[1,i].imshow(results["masks"][idx], cmap="Greys_r")
        ax1[2,i].imshow(results["anomaly_maps"][idx], cmap="jet", vmin=vmin, vmax=vmax)
        ax1[0,i].axis("off")
        ax1[1,i].axis("off")
        ax1[2,i].axis("off")

    # anomalous examples
    N_1 = np.min([N, len(idxs_1)])
    fig2, ax2 = plt.subplots(ncols=N_1, nrows=3, figsize=(N_1*1.3, 3.9), tight_layout=True)
    for i in range(N_1):
        idx = idxs_1[i]
        ax2[0,i].imshow(results["inputs"][idx][0], cmap="Greys_r", vmin=0., vmax=1.)
        ax2[0,i].text(s=f"{results['labels'][idx]}", x=10, y=10, verticalalignment="top", fontsize=14)
        ax2[1,i].imshow(results["masks"][idx], cmap="Greys_r")
        ax2[2,i].imshow(results["anomaly_maps"][idx], cmap="jet", vmin=vmin, vmax=vmax)
        ax2[0,i].axis("off")
        ax2[1,i].axis("off")
        ax2[2,i].axis("off")

    if save_to is not None:
        splitted = os.path.splitext(save_to)
        fig1.savefig(splitted[0]+"_normal"+splitted[1])
        fig2.savefig(splitted[0]+"_anomalous"+splitted[1])
        


def save_config_info(path, config_dict):
    print("---------- TRAIN INFO ----------")
    with open(path, 'w') as file:
        for k in config_dict.keys():
            file.writelines([k, " >>> ", config_dict[k], '\n'])
            print(f"{k}  :  {config_dict[k]}")
    print("--------------------------------")



def plot_worst_results_examples(results, N=10, save_to=None):

    vmin = np.quantile(results["anomaly_maps"].reshape(-1)[:], 0.001)
    vmax = np.quantile(results["anomaly_maps"].reshape(-1)[:], 0.999)

    splitted = os.path.splitext(save_to)

    for label in [0,1]:
        inputs = results["inputs"][results["labels"]==label]
        anomaly_maps = results["anomaly_maps"][results["labels"]==label]
        peaks = np.max(anomaly_maps, axis=(1,2))
        masks = results["masks"][results["labels"]==label]

        sorted_idxs = np.argsort(peaks) if label==1 else np.argsort(peaks)[::-1]

        N = np.min([N, len(sorted_idxs)])
        fig, ax = plt.subplots(ncols=N, nrows=4, figsize=(N*1.3, 5.2), tight_layout=True)
        for i in range(N):
            idx = sorted_idxs[i]
            ax[0,i].imshow(inputs[idx][0], cmap="Greys_r", vmin=0., vmax=1.)
            ax[0,i].text(s=str(label), x=10, y=10, verticalalignment="top", fontsize=14)
            ax[1,i].imshow(masks[idx], cmap="Greys_r")
            ax[2,i].imshow(anomaly_maps[idx], cmap="jet", vmin=vmin, vmax=vmax)
            ax[3,i].hist(inputs[idx][0].reshape(-1)[:], 15)
            ax[0,i].axis("off")
            ax[1,i].axis("off")
            ax[2,i].axis("off")
            ax[2,i].set_title(f"peak={peaks[idx]:.4f}")
            ax[3,i].set_yticks([])
        if save_to is not None:
            path = (splitted[0]+"_normal"+splitted[1]) if label==0 else (splitted[0]+"_anomalous"+splitted[1])
            fig.savefig(path)


def plot_best_results_examples(results, N=10, save_to=None):

    vmin = np.quantile(results["anomaly_maps"].reshape(-1)[:], 0.001)
    vmax = np.quantile(results["anomaly_maps"].reshape(-1)[:], 0.999)

    splitted = os.path.splitext(save_to)

    for label in [0,1]:
        inputs = results["inputs"][results["labels"]==label]
        anomaly_maps = results["anomaly_maps"][results["labels"]==label]
        peaks = np.max(anomaly_maps, axis=(1,2))
        masks = results["masks"][results["labels"]==label]

        sorted_idxs = np.argsort(peaks) if label==0 else np.argsort(peaks)[::-1]

        N = np.min([N, len(sorted_idxs)])
        fig, ax = plt.subplots(ncols=N, nrows=4, figsize=(N*1.3, 5.2), tight_layout=True)
        for i in range(N):
            idx = sorted_idxs[i]
            ax[0,i].imshow(inputs[idx][0], cmap="Greys_r", vmin=0., vmax=1.)
            ax[0,i].text(s=str(label), x=10, y=10, verticalalignment="top", fontsize=14)
            ax[1,i].imshow(masks[idx], cmap="Greys_r")
            ax[2,i].imshow(anomaly_maps[idx], cmap="jet", vmin=vmin, vmax=vmax)
            ax[3,i].hist(inputs[idx][0].reshape(-1)[:], 15)
            ax[0,i].axis("off")
            ax[1,i].axis("off")
            ax[2,i].axis("off")
            ax[2,i].set_title(f"peak={peaks[idx]:.4f}")
            ax[3,i].set_yticks([])
        if save_to is not None:
            path = (splitted[0]+"_normal"+splitted[1]) if label==0 else (splitted[0]+"_anomalous"+splitted[1])
            fig.savefig(path)

