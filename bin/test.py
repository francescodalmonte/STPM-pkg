import numpy as np
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


def test_model():
    # setup input arguments
    params = setupArgs()["TEST"]
    
    data_path = params["DATA_PATH"]
    batch_size = int(params["BATCH_SIZE"])
    crop_size = int(params["CROP_SIZE"])
    checkpoint = params["CHECKPOINT_PATH"]
    num_workers = int(params["N_WORKERS"])
    pin_memory = params["PIN_MEMORY"] is True
    save_path = params["SAVE_PATH"]
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    name_train = params["NAME_TRAIN"]
    out_features = [f.strip() for f in params["OUT_FEATURES"].split(",")]

    device = params["DEVICE"] if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # save configuration file
    utils.save_config_info(os.path.join(save_path, "TEST_CONFIG.txt"), dict(params))

    # instantiate test dataset
    print("Creating dataset and dataloader...")
    test_ds = FilterClothsDataset(data_path,
                                  is_train=False,
                                  resize=crop_size,
                                  cropsize=crop_size)

    print(f"Test ds size: {len(test_ds)}")

    # dataloaders
    test_dl = torch.utils.data.DataLoader(test_ds,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=0,
                                          pin_memory=False)
    
    # instantiate models
    print("Instantiating models...")
    teacher_net = modified_resnet18(pretrained=True, out_features=out_features).to(device)
    student_net = modified_resnet18(pretrained=False, out_features=out_features).to(device)

    if os.path.isfile(checkpoint):
        student_net.load_state_dict(torch.load(checkpoint))
    else:
        print(f"No checkpoint file ound at {checkpoint}")

    for param in teacher_net.parameters():
        param.requires_grad = False
    _ = teacher_net.eval() # teacher model will always remain in eval mode

    # test
    print("Testing...")
    results = test_student_model(teacher_net,
                                 student_net,
                                 test_dl,
                                 device)
    # save avg anomaly and peak anomaly histograms to file
    utils.plot_multi_hist(results, save_to=os.path.join(save_path, "anomaly_histograms.png"))
    utils.plot_multi_curves(results, save_to=os.path.join(save_path, "curves.png"))
    utils.plot_multi_curves_area(results, save_to=os.path.join(save_path, "curves.png"))


    # save examples of heatmaps
    utils.plot_results_examples(results, N=20, save_to=os.path.join(save_path, "results_examples.png"))
    utils.plot_best_results_examples(results, N=25, save_to=os.path.join(save_path, "results_examples_best.png"))
    utils.plot_worst_results_examples(results, N=25, save_to=os.path.join(save_path, "results_examples_worst.png"))


if __name__ == "__main__":
    start = time.time()

    test_model()

    print(f"Elapsed time: {(time.time()-start):2f} s")