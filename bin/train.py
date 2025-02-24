import numpy as np
import os
import time
import configparser
import argparse

import torch

from STPM_model import utils
from STPM_model.dataset import FilterClothsDataset, custom_collate_function, ImagenetteDataset, InfiniteDataLoader
from STPM_model.model import model_selector
from STPM_model.training import train_loop, test_student_model


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


def train_model():
    # setup input arguments
    params = setupArgs()["TRAINING"]
    
    data_path = params["DATA_PATH"].strip()
    auxilary_data_path = params["AUXILARY_DATA_PATH"].strip()
    batch_size_train = int(params["BATCH_SIZE_TRAIN"])
    batch_size_val = int(params["BATCH_SIZE_VAL"])
    batch_size_aux = int(params["BATCH_SIZE_AUX"])
    crop_size = int(params["CROP_SIZE"])
    checkpoint = params["CHECKPOINT_PATH"]
    num_workers = int(params["N_WORKERS"])
    pin_memory = params["PIN_MEMORY"] is True
    save_path = params["SAVE_PATH"]
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    name_train = params["NAME_TRAIN"]
    model_name = params["MODEL_NAME"]
    num_epochs = int(params["N_EPOCHS"])
    limit_train_batches = int(params["LIMIT_TRAIN_BATCHES"])
    out_features_train = [int(f.strip()) for f in params["OUT_FEATURES_TRAIN"].split(",")]
    out_features_test = [int(f.strip()) for f in params["OUT_FEATURES_TEST"].split(",")]
    lr = float(params["LR"])
    lr_scheduler_steps = [int(s) for s in params["LR_SCHEDULER_STEPS"].split(",")]
    gamma = float(params["LR_SCHEDULER_GAMMA"])

    device = params["DEVICE"] if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # save configuration file
    utils.save_config_info(os.path.join(save_path, "TRAIN_CONFIG.txt"), dict(params))

    print("Creating datasets and dataloaders...")
    train_ds_tot = FilterClothsDataset(data_path,
                                       is_train=True,
                                       resize=crop_size,
                                       cropsize=crop_size)
    test_ds = FilterClothsDataset(data_path,
                                  is_train=False,
                                  resize=crop_size,
                                  cropsize=crop_size)

    # validation split
    # (in an AD task usually there is no risk of overtraining. However, a validation set
    # can used to assess the model capability to generalize its behaviour to new normal samples)
    img_nums = len(train_ds_tot)
    valid_num = int(img_nums * 0.15)
    train_num = img_nums - valid_num
    train_ds, val_ds = torch.utils.data.random_split(train_ds_tot, [train_num, valid_num])

    if len(auxilary_data_path)>0:
        auxilary_ds = ImagenetteDataset(
            data_path=auxilary_data_path, resize=crop_size,
            seed=42, max_samples=9999, prefetch=False
        )
        print(f"Auxilary ds size: {len(auxilary_ds)}")


    print(f"Train/val ds sizes: {len(train_ds)}/{len(val_ds)}")
    print(f"Test ds size: {len(test_ds)}")

    # dataloaders
    persistent_workers = True if num_workers>0 else False
    train_dl = torch.utils.data.DataLoader(train_ds,
                                           batch_size=batch_size_train,
                                           shuffle=True,
                                           num_workers=num_workers,
                                           persistent_workers=persistent_workers,
                                           collate_fn=custom_collate_function,
                                           pin_memory=pin_memory)
    val_dl = torch.utils.data.DataLoader(val_ds,
                                         batch_size=batch_size_val,
                                         shuffle=False,
                                         num_workers=num_workers,
                                         persistent_workers=persistent_workers,
                                         collate_fn=custom_collate_function,
                                         pin_memory=pin_memory)
    test_dl = torch.utils.data.DataLoader(test_ds,
                                          batch_size=batch_size_val,
                                          shuffle=False,
                                          num_workers=num_workers,
                                          persistent_workers=persistent_workers,
                                          collate_fn=custom_collate_function,
                                          pin_memory=pin_memory)
    if len(auxilary_data_path)>0:
        auxilary_dl = InfiniteDataLoader(
            torch.utils.data.DataLoader(auxilary_ds,
                                        batch_size=batch_size_aux,
                                        shuffle=True,
                                        num_workers=num_workers,
                                        persistent_workers=persistent_workers,
                                        pin_memory=pin_memory))
    else:
        auxilary_dl = None
        print("No auxilary data provided")

    # plot some examples and histograms
    utils.plot_examples(train_dl, N=8, save_to=os.path.join(save_path, "examples_train_dl.png"))
    utils.plot_examples(test_dl, N=8, save_to=os.path.join(save_path, "examples_test_dl.png"))
    utils.plot_examples_histograms(train_dl, N=4, save_to=os.path.join(save_path, "examplesHist_train_dl.png"))
    utils.plot_examples_histograms(test_dl, N=4, save_to=os.path.join(save_path, "examplesHist_test_dl.png"))
    if len(auxilary_data_path)>0:
        utils.plot_examples(auxilary_dl, N=8, save_to=os.path.join(save_path, "examples_auxilary_dl.png"), mean=0.446, std=0.224)
        utils.plot_examples_histograms(auxilary_dl, N=4, save_to=os.path.join(save_path, "examplesHist_auxilary_dl.png"))


    # instantiate models
    print("Instantiating models...")
    teacher_net = model_selector(
        model_name, pretrained=True).to(device)
    student_net = model_selector(
        model_name, pretrained=False).to(device)
    print(f"N. parameters student model: {sum(p.numel() for p in student_net.parameters())/1e6:.2f}M")

    if len(checkpoint)>0:
        if os.path.isfile(checkpoint):
            student_net.load_state_dict(torch.load(checkpoint))
            if not os.path.isdir(os.path.join(save_path, f"checkpoints")):
                os.mkdir(os.path.join(save_path, f"checkpoints"))
            torch.save(student_net.state_dict(), os.path.join(save_path, f"checkpoints/{name_train}.ckpt"))
        else:
            print(f"No checkpoint file ound at {checkpoint}")

    for param in teacher_net.parameters():
        param.requires_grad = False
    _ = teacher_net.eval() # teacher model will always remain in eval mode
    

    # define optimization and learning rate scheduling strategies
    optimizer = torch.optim.Adam(student_net.parameters(),
                                 lr=lr,
                                 weight_decay=0.0001)
    n_effective_updates = min(len(train_dl), limit_train_batches) * num_epochs
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                                    total_steps=n_effective_updates,
                                                    pct_start=0.2)
    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                    lr_scheduler_steps,
    #                                                    gamma=gamma)


    # start training
    train_dict = train_loop(teacher_net,
                            student_net,
                            out_features_train,
                            train_dl,
                            val_dl,
                            device,
                            num_epochs,
                            optimizer,
                            name_train,
                            save_path,
                            test_loader=test_dl,
                            log_interval=1,
                            lr_scheduler=lr_scheduler,
                            limit_train_batches=limit_train_batches,
                            verbose=True,
                            auxilary_dl=auxilary_dl)
    utils.plot_training_losses(train_dict, save_to=os.path.join(save_path, "training_loss.png"))

    student_net.load_state_dict(torch.load(os.path.join(save_path, f"checkpoints/{name_train}.ckpt")))

    # test
    results = test_student_model(teacher_net,
                                 student_net,
                                 out_features_test,
                                 test_dl,
                                 device)
    # save avg anomaly and peak anomaly histograms to file
    utils.plot_multi_hist(results, save_to=os.path.join(save_path, "anomaly_histograms.png"))
    utils.plot_multi_curves(results, save_to=os.path.join(save_path, "curves.png"))
    # save examples of heatmaps
    utils.plot_results_examples(results, N=12, save_to=os.path.join(save_path, "results_examples.png"))
    utils.plot_best_results_examples(results, N=20, save_to=os.path.join(save_path, "results_examples_best.png"))
    utils.plot_worst_results_examples(results, N=20, save_to=os.path.join(save_path, "results_examples_worst.png"))


if __name__ == "__main__":
    start = time.time()

    train_model()

    print(f"Elapsed time: {(time.time()-start):2f} s")