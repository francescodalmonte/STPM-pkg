import numpy as np
import os
import time
import configparser, argparse
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
import torch

from preprocessing_tele.image_processing import saveCrops
from preprocessing_tele.dataset import conditionalMkDir

from STPM_model.model import modified_resnet18
from STPM_model.training import compute_anomaly_maps
from STPM_model import inference



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


def run_inference(dataloader, teacher_net, student_net, device, verbose=True, gauss=0.5, m=2.):
    """Run inference on the input data and compute anomaly maps.
    """
    start = time.time()
    # inference
    anomaly_maps = []
    with torch.no_grad():
        for i, (x,) in enumerate(dataloader):
            if (i % (len(dataloader)//10)) == 0 and verbose:
                print(f"Processing crop {i}/{len(dataloader)}")
            # forward pass
            x = x.to(device)
            features_t = teacher_net(x)
            features_s = student_net(x)

            a_map = compute_anomaly_maps(features_s, features_t, out_size=crop_size)
            a_map = gaussian_filter(a_map, sigma=gauss)
            a_map = np.clip((a_map*m), 0., 1.)

            anomaly_maps.append(a_map)

    anomaly_maps = np.concatenate(anomaly_maps)
    anomaly_peaks = np.max(anomaly_maps, axis=(1,2))

    print(f"run_inference() elapsed time: {(time.time()-start):2f} s")
    return anomaly_maps, anomaly_peaks



if __name__ == "__main__":

    start = time.time()
    # setup input arguments
    params = setupArgs()["INFERENCE"]

    ckpt_path = params["CKPT_PATH"]
    input_path = params["INPUT_PATH"]
    input_name = params["INPUT_NAME"]
    save_path = params["SAVE_PATH"]
    batch_size = int(params["BATCH_SIZE"])
    n_workers = int(params["N_WORKERS"])
    crop_size = int(params["CROP_SIZE"])
    overlap = int(params["OVERLAP"])
    mode = params["MODE"]
    term1 = int(params["TERM1"])
    term2 = int(params["TERM2"])
    eval_mode = bool(int(params["EVAL_MODE"]))
    save_eval_crops = bool(int(params["SAVE_EVAL_CROPS"]))
    contrast_correction = bool(int(params["CONTRAST_CORRECTION"]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")


    # TILE INPUT IMAGE, SAVE CROPS TO FILE
    tiles, coords, image = inference.tile_input_image(name = input_name,
                                                      root_path = input_path,
                                                      size = crop_size,
                                                      overlap = overlap,
                                                      scale = 1.,
                                                      mode = mode,
                                                      term1 = term1,
                                                      term2 = term2,
                                                      wmask = eval_mode,
                                                      contrast_correction = contrast_correction)
    

    if eval_mode:
        tiles_masks = tiles[:,:,:,3]
        tiles = tiles[:,:,:,:3]

    conditionalMkDir(save_path)
    conditionalMkDir(os.path.join(save_path, "crops"))
    conditionalMkDir(os.path.join(save_path, "crops_masks"))

    saveCrops(save_to = os.path.join(save_path, "crops"),
              crops_set = (255*tiles[:,:,:,0]).astype(int),
              centers_set = coords,
              prefix = input_name,
              suffix = "")
    if eval_mode:
        saveCrops(save_to = os.path.join(save_path, "crops_masks"),
                  crops_set = (255*tiles_masks).astype(int),
                  centers_set = coords,
                  prefix = input_name,
                suffix = "_M")



    # LOAD TRAINED MODEL

    # load model checkpoint
    teacher_net = modified_resnet18(pretrained=True).to(device)
    student_net = modified_resnet18(pretrained=False).to(device)

    for param in list(teacher_net.parameters())+list(student_net.parameters()):
        param.requires_grad = False
    _ = teacher_net.eval()
    _ = student_net.eval()

    student_net.load_state_dict(torch.load(ckpt_path))

    # INFERENCE

    # create dataloader
    inputs = TensorDataset(torch.Tensor(np.transpose(tiles, (0,3,1,2))))
    inputs_dl = DataLoader(
        inputs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=False
    )

    anomaly_maps, anomaly_peaks = run_inference(
        inputs_dl,
        teacher_net,
        student_net,
        device,
        verbose=True,
        gauss=0.5,
        m=2.
    )
    
    print(f"Tot elapsed time: {(time.time()-start):2f} s")

    # run evaluation if needed
    if eval_mode:
        print("Evaluating model performance...")
        threshold_set = [qt for qt in np.linspace(0.15, .50, 50)]

        for qt in threshold_set:
            conditionalMkDir(os.path.join(save_path, f"eval_{qt:.5f}"))
            TP, TN, FP, FN, TPclasses, FNclasses = inference.evaluate_detection(anomaly_maps,
                                     tiles_masks,
                                     tiles,
                                     threshold = qt,
                                     save_eval_crops = save_eval_crops,
                                     save_path = os.path.join(save_path, f"eval_{qt:.5f}")
                                     )
            s = f"Threshold: {qt:.5f}\tTP: {TP} ({TPclasses})\tTN: {TN}\tFP: {FP}\tFN: {FN} ({FNclasses})"
            print(s)
            with open(os.path.join(save_path, "results.dat"), "a") as file:
                file.write(s)
                file.write("\n")
            if FP == 0:
                break

    # SAVE RESULTS

    saveCrops(save_to = os.path.join(save_path, "crops"),
              crops_set = (255*anomaly_maps).astype(np.uint8),
              centers_set = coords,
              prefix = input_name,
              suffix = "_AMAP",
              mode = "color_map") 

    # sorted arrays
    #sorted_input_images = tiles[coords[:,0].argsort()] 
    sorted_anomaly_maps = anomaly_maps[coords[:,0].argsort()] 
    sorted_anomaly_peaks = anomaly_peaks[coords[:,0].argsort()]  
    sorted_coords = coords[coords[:,0].argsort()] 

    #sorted_input_images = sorted_input_images[sorted_coords[:,1].argsort(kind='mergesort')]
    sorted_anomaly_maps = sorted_anomaly_maps[sorted_coords[:,1].argsort(kind='mergesort')]
    sorted_anomaly_peaks = sorted_anomaly_peaks[sorted_coords[:,1].argsort(kind='mergesort')]
    sorted_coords = sorted_coords[sorted_coords[:,1].argsort(kind='mergesort')]

    # save to file
    #np.save(os.path.join(save_path, "inputs_set.npy"), sorted_input_images)
    #np.save(os.path.join(save_path, "coords_set.npy"), sorted_coords)
    #np.save(os.path.join(save_path, "anomaly_maps_set.npy"), sorted_anomaly_maps)
    #np.save(os.path.join(save_path, "anomaly_scores_set.npy"), sorted_anomaly_peaks)

    inference.save_anomaly_hist(anomaly_peaks,
                                os.path.join(save_path, "anomaly_hist_log.png"))
    inference.save_anomaly_hist(anomaly_peaks,
                                os.path.join(save_path, "anomaly_hist.png"))
    inference.save_annotated_image(image[:,:,:3],
                                   crop_size,
                                   sorted_coords,
                                   sorted_anomaly_peaks,
                                   os.path.join(save_path, "annotated_image.png"),
                                   threshold = 0.75)
    inference.compose_anomalyImage(sorted_anomaly_maps,
                                   sorted_coords,
                                   save_path=os.path.join(save_path, "anomaly_heatmap_tot_C.png"),
                                   image_shape=[12500,12500],
                                   crop_size=crop_size,
                                   false_color=True)


