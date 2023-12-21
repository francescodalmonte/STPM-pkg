import numpy as np
import os
import time
import configparser

import torch

from STPM_model import utils
from STPM_model.dataset import FilterClothsDataset
from STPM_model.model import modified_resnet18
from STPM_model.training import train_loop, test_student_model


def setupArgs():
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), "config.INI")
    if os.path.isfile(config_path):
        config.read(config_path)
    else:
        raise ValueError(f"can't find configuration file {config_path}")
    
    return config


def train_model():
    # setup input arguments
    params = setupArgs()["TRAINING"]
    
    data_path = params["DATA_PATH"]
    batch_size = int(params["BATCH_SIZE"])
    num_workers = int(params["N_WORKERS"])
    pin_memory = params["PIN_MEMORY"] is True
    save_path = params["SAVE_PATH"]
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    name_train = params["NAME_TRAIN"]
    num_epochs = int(params["N_EPOCHS"])
    lr = float(params["LR"])
    lr_scheduler_steps = [int(s) for s in params["LR_SCHEDULER_STEPS"].split(",")]
    gamma = float(params["LR_SCHEDULER_GAMMA"])


    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # instantiate train and test datasets
    train_ds_tot = FilterClothsDataset(data_path,
                                       is_train=True,
                                       resize=224,
                                       cropsize=224)
    test_ds = FilterClothsDataset(data_path,
                                  is_train=False,
                                  resize=224,
                                  cropsize=224)

    # validation split
    # (in an AD task usually there is no risk of overtraining. However, a validation set
    # can used to assess the model capability to generalize its behaviour to new normal samples)
    img_nums = len(train_ds_tot)
    valid_num = int(img_nums * 0.1)
    train_num = img_nums - valid_num
    train_ds, val_ds = torch.utils.data.random_split(train_ds_tot, [train_num, valid_num])

    # dataloaders
    train_dl = torch.utils.data.DataLoader(train_ds,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers,
                                           pin_memory=pin_memory)
    val_dl = torch.utils.data.DataLoader(val_ds,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=num_workers,
                                         pin_memory=pin_memory)
    test_dl = torch.utils.data.DataLoader(test_ds,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=num_workers,
                                          pin_memory=pin_memory)

    # plot some examples and histograms
    utils.plot_examples(train_dl, N=8, save_to=os.path.join(save_path, "examples_train_dl.png"))
    utils.plot_examples(test_dl, N=8, save_to=os.path.join(save_path, "examples_test_dl.png"))


    # instantiate models
    teacher_net = modified_resnet18(pretrained=True).to(device)
    student_net = modified_resnet18(pretrained=False).to(device)

    for param in teacher_net.parameters():
        param.requires_grad = False
    _ = teacher_net.eval() # teacher model will always remain in eval mode
    

    # define optimization and learning rate scheduling strategies
    optimizer = torch.optim.Adam(student_net.parameters(),
                                 lr=lr,
                                 weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        lr_scheduler_steps,
                                                        gamma=gamma)


    # start training
    train_dict = train_loop(teacher_net,
                            student_net,
                            train_dl,
                            val_dl,
                            device,
                            num_epochs,
                            optimizer,
                            name_train,
                            save_path,
                            log_interval=-1,
                            lr_scheduler=lr_scheduler,
                            verbose=True)
    utils.plot_training_loss(train_dict, save_to=os.path.join(save_path, "training_loss.png"))


    # load model checkpoint
    teacher_net = modified_resnet18(pretrained=True).to(device)
    student_net = modified_resnet18(pretrained=False).to(device)

    for param in teacher_net.parameters():
        param.requires_grad = False
    _ = teacher_net.eval() # teacher model will always remain in eval mode

    student_net.load_state_dict(torch.load(os.path.join(save_path, f"checkpoints/{name_train}.ckpt")))

    # test
    results = test_student_model(teacher_net,
                                 student_net,
                                 test_dl,
                                 device)
    # save avg anomaly and peak anomaly histograms to file
    utils.plot_multi_hist(results, save_to=os.path.join(save_path, "anomaly_histograms.png"))
    utils.plot_multi_curves(results, save_to=os.path.join(save_path, "curves.png"))
    # save examples of heatmaps
    utils.plot_results_examples(results, N=8, save_to=os.path.join(save_path, "results_examples.png"))


if __name__ == "__main__":
    start = time.time()

    train_model()

    print(f"Elapsed time: {(time.time()-start):2f} s")