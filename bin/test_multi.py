import numpy as np
from matplotlib import pyplot as plt
import os
import time
import configparser
import argparse

import torch

from STPM_model import utils
from STPM_model.dataset import FilterClothsDataset
from STPM_model.model import modified_resnet18
from STPM_model.training import test_student_model


def setupArgs():
    parser = argparse.ArgumentParser()
    config = configparser.ConfigParser()

    parser.add_argument("--config",
                        type=str,
                        help="Absolute filepath of config (.INI) file (default: ./config.INI)",
                        default=os.path.join(os.path.dirname(__file__), "config.INI")
                        )

    config_path = parser.parse_args().config

    if os.path.isfile(config_path):
        config.read(config_path)
    else:
        raise ValueError(f"can't find configuration file {config_path}")

    return config


def test_multi_model():
    # setup input arguments
    params = setupArgs()["TEST_MULTI"]
    
    data_paths = [p.strip() for p in params["DATA_PATHS"].split(",")]
    checkpoint_paths = [p.strip() for p in params["CHECKPOINT_PATHS"].split(",")]

    batch_size = int(params["BATCH_SIZE"])
    crop_size = int(params["CROP_SIZE"])

    save_path = params["SAVE_PATH"]
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    name_train = params["NAME_TRAIN"]
    out_features = [f.strip() for f in params["OUT_FEATURES"].split(",")]

    device = params["DEVICE"] if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # save configuration file
    utils.save_config_info(os.path.join(save_path, "TEST_CONFIG.txt"), dict(params))

    # instantiate test datasets and dataloaders
    print("Creating dataset and dataloader...")
    test_dss = []
    test_dls = []


    for i, data_path in enumerate(data_paths):
        if not os.path.isdir(data_path):
            raise ValueError(f"Data path {data_path} not found")
        test_ds = FilterClothsDataset(data_path,
                                       is_train=False,
                                       resize=crop_size,
                                       cropsize=crop_size)
        test_dss.append(test_ds)
        print(f"Test ds {i} size: {len(test_ds)}")
        test_dl = torch.utils.data.DataLoader(test_ds,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=0,
                                              pin_memory=False)
        test_dls.append(test_dl)
 

    print("Instantiating models...")
    student_nets = []
    teacher_net = modified_resnet18(pretrained=True, out_features=out_features).to(device)

    for i, checkpoint_path in enumerate(checkpoint_paths):
        if not os.path.isfile(checkpoint_path):
            raise ValueError(f"Checkpoint file {checkpoint_path} not found")
        # instantiate models
        student_net = modified_resnet18(pretrained=False, out_features=out_features).to(device)
        student_net.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded checkpoint {checkpoint_path} for student model {i}")
        student_nets.append(student_net)



    for param in teacher_net.parameters():
        param.requires_grad = False
    _ = teacher_net.eval() # teacher model will always remain in eval mode

    # test
    print("Testing models...")
    for i, (test_dl, student_net) in enumerate(zip(test_dls, student_nets)):
        results = test_student_model(teacher_net,
                                     student_net,
                                     test_dl,
                                     device)
    
        # save results to file
        print("Saving results to file...")

        # save avg anomaly and peak anomaly histograms to file
        #utils.plot_multi_hist(results, save_to=os.path.join(save_path, f"anomaly_histograms_{i}.png"))
        #utils.plot_multi_curves(results, save_to=os.path.join(save_path, f"curves_{i}.png"))

        # save examples of heatmaps
        #utils.plot_results_examples(results, N=20, save_to=os.path.join(save_path, f"results_example_{i}.png"))
        #utils.plot_best_results_examples(results, N=25, save_to=os.path.join(save_path, f"results_examples_best_{i}.png"))
        #utils.plot_worst_results_examples(results, N=25, save_to=os.path.join(save_path, f"results_examples_worst_{i}.png"))

        # save anomaly maps to file
        np.save(os.path.join(save_path, f"anomaly_maps_{i}.npy"), results["anomaly_maps"])
        if i==0:
            np.save(os.path.join(save_path, f"inputs_{i}.npy"), results["inputs"])
            np.save(os.path.join(save_path, f"labels_{i}.npy"), results["labels"])


def combine_results():
    # setup input arguments
    params = setupArgs()["TEST_MULTI"]
    save_path = params["SAVE_PATH"]

    checkpoint_paths = [p.strip() for p in params["CHECKPOINT_PATHS"].split(",")]
    N = len(checkpoint_paths)

    # Load data
    anomaly_maps = []
    
    for i in range(N):
        anomaly_maps.append(np.load(os.path.join(save_path, f"anomaly_maps_{i}.npy")))
        if i==0:
            inputs = np.load(os.path.join(save_path, f"inputs_{i}.npy"))
            labels = np.load(os.path.join(save_path, f"labels_{i}.npy"))
    anomaly_maps = np.array(anomaly_maps)

    print(f"Loaded anomaly maps shape: {anomaly_maps.shape}")


    # Plot anomaly maps
    mean_maps = np.mean(anomaly_maps, axis=0)
    std_maps = np.std(anomaly_maps, axis=0)

    suppression_factor = 20
    agreement_mean_maps = mean_maps.copy()/(1 + suppression_factor*std_maps)

    fig, ax = plt.subplots(nrows=N+3, ncols=12, figsize=(20, 2*(N+3)), tight_layout=True)
    for i in range(12):
        ax[0,i].imshow(inputs[i][0])
        ax[0,i].axis("off")
        ax[0,i].set_title(f"input {i}")
        for j in range(N):
            ax[j+1,i].imshow(anomaly_maps[j][i], cmap="jet")
            ax[j+1,i].axis("off")
            ax[j+1,i].set_title(f"anom. map {j}-{i}")
        ax[-2,i].imshow(mean_maps[i], cmap="jet")
        ax[-2,i].axis("off")
        ax[-2,i].set_title(f"mean map {i}")
        ax[-1,i].imshow(agreement_mean_maps[i], cmap="jet")
        ax[-1,i].axis("off")
        ax[-1,i].set_title(f"agreem. map {i}")

    plt.savefig(os.path.join(save_path, "anomaly_maps_combined.png"))

    # Plot Histograms
    fig, ax = plt.subplots(nrows=N+2, ncols=1, figsize=(6, 3*(N+2)), tight_layout=True)
    for j in range(N):
        utils.plot_hist_on_ax(ax[j], np.max(anomaly_maps[j], axis=(1,2)), labels,
                              f"Peak {j}", vlines = True, bins=25, alpha=0.5)
        ax[j].set_xlim((0, 0.42))
    utils.plot_hist_on_ax(ax[-2], np.max(mean_maps, axis=(1,2)), labels, 
                          "mean maps", vlines = True, bins=25, alpha=0.5)
    ax[-2].set_xlim((0, 0.42))
    utils.plot_hist_on_ax(ax[-1], np.max(agreement_mean_maps, axis=(1,2)), labels,
                          "agreem. maps", vlines = True, bins=25, alpha=0.5)
    ax[-1].set_xlim((0, 0.42))

    plt.savefig(os.path.join(save_path, "histograms_combined.png"))


    # Plot ROC curves
    fig, ax = plt.subplots(nrows=2, ncols=N+2, figsize=(3*(N+2), 6), tight_layout=True)
    for j in range(N):
        utils.plot_roc_curve(ax[0, j], np.max(anomaly_maps[j], axis=(1,2)), labels, f"ROC Peak {j}")
        utils.plot_roc_curve(ax[1, j], np.max(anomaly_maps[j], axis=(1,2)), labels, f"ROClog Peak {j}", log=True)
    utils.plot_roc_curve(ax[0, -2], np.max(mean_maps, axis=(1,2)), labels, "ROC mean maps")
    utils.plot_roc_curve(ax[1, -2], np.max(mean_maps, axis=(1,2)), labels, "ROClog mean maps", log=True)
    utils.plot_roc_curve(ax[0, -1], np.max(agreement_mean_maps, axis=(1,2)), labels, "ROC agreem. maps")
    utils.plot_roc_curve(ax[1, -1], np.max(agreement_mean_maps, axis=(1,2)), labels, "ROClog agreem. maps", log=True)

    plt.savefig(os.path.join(save_path, "curves_combined.png"))

    # Plot ROC (area) curves
    fig, ax = plt.subplots(nrows=2, ncols=N+2, figsize=(3*(N+2), 6), tight_layout=True)
    for j in range(N):
        utils.plot_roc_curve_area(ax[0, j], np.max(anomaly_maps[j], axis=(1,2)), labels, f"ROC(area) Peak {j}")
        utils.plot_roc_curve_area(ax[1, j], np.max(anomaly_maps[j], axis=(1,2)), labels, f"ROC(area)log Peak {j}", log=True)
    utils.plot_roc_curve_area(ax[0, -2], np.max(mean_maps, axis=(1,2)), labels, "ROC(area) mean maps")
    utils.plot_roc_curve_area(ax[1, -2], np.max(mean_maps, axis=(1,2)), labels, "ROC(area)log mean maps", log=True)
    utils.plot_roc_curve_area(ax[0, -1], np.max(agreement_mean_maps, axis=(1,2)), labels, "ROC(area) agreem. maps")
    utils.plot_roc_curve_area(ax[1, -1], np.max(agreement_mean_maps, axis=(1,2)), labels, "ROC(area)log agreem. maps", log=True)

    plt.savefig(os.path.join(save_path, "curves_combined(area).png"))


if __name__ == "__main__":
    start = time.time()

    test_multi_model()
    combine_results()

    print(f"Elapsed time: {(time.time()-start):2f} s")