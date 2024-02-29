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
    
    data_path1 = params["DATA_PATH1"]
    data_path2 = params["DATA_PATH2"]
    batch_size = int(params["BATCH_SIZE"])
    crop_size = int(params["CROP_SIZE"])
    checkpoint1 = params["CHECKPOINT_PATH1"]
    checkpoint2 = params["CHECKPOINT_PATH2"]

    save_path = params["SAVE_PATH"]
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    name_train = params["NAME_TRAIN"]
    out_features = [f.strip() for f in params["OUT_FEATURES"].split(",")]

    device = params["DEVICE"] if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # save configuration file
    utils.save_config_info(os.path.join(save_path, "TEST_CONFIG.txt"), dict(params))

    # instantiate test datasets
    print("Creating dataset and dataloader...")
    test_ds1 = FilterClothsDataset(data_path1,
                                  is_train=False,
                                  resize=crop_size,
                                  cropsize=crop_size)
    test_ds2 = FilterClothsDataset(data_path2,
                                    is_train=False,
                                    resize=crop_size,
                                    cropsize=crop_size)

    print(f"Test ds sizes: {len(test_ds1)}=={len(test_ds2)}")

    # dataloaders
    test_dl1 = torch.utils.data.DataLoader(test_ds1,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=0,
                                          pin_memory=False)
    test_dl2 = torch.utils.data.DataLoader(test_ds2,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=0,
                                          pin_memory=False)
    
    # instantiate models
    print("Instantiating models...")
    teacher_net = modified_resnet18(pretrained=True, out_features=out_features).to(device)
    student_net1 = modified_resnet18(pretrained=False, out_features=out_features).to(device)
    student_net2 = modified_resnet18(pretrained=False, out_features=out_features).to(device)

    if os.path.isfile(checkpoint1):
        student_net1.load_state_dict(torch.load(checkpoint1))
    else:
        print(f"No checkpoint file ound at {checkpoint1}")
    if os.path.isfile(checkpoint2):
        student_net2.load_state_dict(torch.load(checkpoint2))
    else:
        print(f"No checkpoint file ound at {checkpoint2}")

    for param in teacher_net.parameters():
        param.requires_grad = False
    _ = teacher_net.eval() # teacher model will always remain in eval mode

    # test
    print("Testing models...")
    results1 = test_student_model(teacher_net,
                                 student_net1,
                                 test_dl1,
                                 device)
    results2 = test_student_model(teacher_net,
                                 student_net2,
                                 test_dl2,
                                 device)
    
    # save results to file
    print("Saving results to file...")
    # save avg anomaly and peak anomaly histograms to file
    utils.plot_multi_hist(results1, save_to=os.path.join(save_path, "anomaly_histograms1.png"))
    utils.plot_multi_hist(results2, save_to=os.path.join(save_path, "anomaly_histograms2.png"))
    utils.plot_multi_curves(results1, save_to=os.path.join(save_path, "curves1.png"))
    utils.plot_multi_curves(results2, save_to=os.path.join(save_path, "curves2.png"))
    # save examples of heatmaps
    utils.plot_results_examples(results1, N=20, save_to=os.path.join(save_path, "results1_examples.png"))
    utils.plot_results_examples(results2, N=20, save_to=os.path.join(save_path, "results2_examples.png"))
    utils.plot_best_results_examples(results1, N=25, save_to=os.path.join(save_path, "results1_examples_best.png"))
    utils.plot_best_results_examples(results2, N=25, save_to=os.path.join(save_path, "results2_examples_best.png"))
    utils.plot_worst_results_examples(results1, N=25, save_to=os.path.join(save_path, "results1_examples_worst.png"))
    utils.plot_worst_results_examples(results2, N=25, save_to=os.path.join(save_path, "results2_examples_worst.png"))


    # save anomaly maps to file
    np.save(os.path.join(save_path, "anomaly_maps1.npy"), results1["anomaly_maps"])
    np.save(os.path.join(save_path, "anomaly_maps2.npy"), results2["anomaly_maps"])
    np.save(os.path.join(save_path, "inputs1.npy"), results1["inputs"])
    np.save(os.path.join(save_path, "inputs2.npy"), results2["inputs"])
    np.save(os.path.join(save_path, "labels1.npy"), results1["labels"])
    np.save(os.path.join(save_path, "labels2.npy"), results2["labels"])


def combine_results():
    # setup input arguments
    params = setupArgs()["TEST_MULTI"]
    save_path = params["SAVE_PATH"]

    # Load data
    anomaly_maps1 = np.load(os.path.join(save_path, "anomaly_maps1.npy"))
    anomaly_maps2 = np.load(os.path.join(save_path, "anomaly_maps2.npy"))
    labels1 = np.load(os.path.join(save_path, "labels1.npy"))
    labels2 = np.load(os.path.join(save_path, "labels2.npy"))
    inputs1 = np.load(os.path.join(save_path, "inputs1.npy"))
    inputs2 = np.load(os.path.join(save_path, "inputs2.npy"))

    assert (labels1==labels2).all()

    # Plot anomaly maps
    sum = anomaly_maps1 + anomaly_maps2
    sumsq = np.sqrt(anomaly_maps1**2 + anomaly_maps2**2)


    fig, ax = plt.subplots(nrows=5, ncols=8, figsize=(16, 8), tight_layout=True)
    for i in range(8):
        ax[0,i].imshow(inputs1[i][0]); ax[0,i].axis("off"); ax[0,i].set_title(f"input {i}")
        ax[1,i].imshow(anomaly_maps1[i], cmap="jet"); ax[1,i].axis("off"); ax[1,i].set_title(f"anomaly1 {i}")
        ax[2,i].imshow(anomaly_maps2[i], cmap="jet"); ax[2,i].axis("off"); ax[2,i].set_title(f"anomaly2 {i}")
        ax[3,i].imshow(sum[i], cmap="jet"); ax[3,i].axis("off"); ax[3,i].set_title(f"sum {i}")
        ax[4,i].imshow(sumsq[i], cmap="jet"); ax[4,i].axis("off"); ax[4,i].set_title(f"sumsq {i}")

    plt.savefig(os.path.join(save_path, "anomaly_maps_combined.png"))

    # Plot Histograms
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 4), tight_layout=True)
    utils.plot_hist_on_ax(ax[0], np.max(anomaly_maps1, axis=(1,2)), labels1, "Peak 1", bins=20, alpha=0.5)
    utils.plot_hist_on_ax(ax[1], np.max(anomaly_maps2, axis=(1,2)), labels1, "Peak 2", bins=20, alpha=0.5)
    utils.plot_hist_on_ax(ax[2], np.max(sum, axis=(1,2)), labels1, "Sum", bins=20, alpha=0.5)
    utils.plot_hist_on_ax(ax[3], np.max(sumsq, axis=(1,2)), labels1, "sumsq", bins=20, alpha=0.5)

    plt.savefig(os.path.join(save_path, "histograms_combined.png"))


    # Plot ROC curves
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(16, 4), tight_layout=True)
    utils.plot_roc_curve(ax[0,0], np.max(anomaly_maps1, axis=(1,2)), labels1, "ROC Peak 1")
    utils.plot_roc_curve(ax[1,0], np.max(anomaly_maps1, axis=(1,2)), labels1, "ROClog Peak 1", log=True)
    utils.plot_roc_curve(ax[0,1], np.max(anomaly_maps2, axis=(1,2)), labels1, "ROC Peak 2")
    utils.plot_roc_curve(ax[1,1], np.max(anomaly_maps2, axis=(1,2)), labels1, "ROClog Peak 2", log=True)
    utils.plot_roc_curve(ax[0,2], np.max(sum, axis=(1,2)), labels1, "ROC Sum")
    utils.plot_roc_curve(ax[1,2], np.max(sum, axis=(1,2)), labels1, "ROClog Sum", log=True)
    utils.plot_roc_curve(ax[0,3], np.max(sumsq, axis=(1,2)), labels1, "ROC sumsq")
    utils.plot_roc_curve(ax[1,3], np.max(sumsq, axis=(1,2)), labels1, "ROClog sumsq", log=True)

    plt.savefig(os.path.join(save_path, "curves_combined.png"))


if __name__ == "__main__":
    start = time.time()

    test_multi_model()
    combine_results()

    print(f"Elapsed time: {(time.time()-start):2f} s")