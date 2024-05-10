import numpy as np
import os
import time
import configparser, argparse
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt

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



if __name__ == "__main__":


    # setup input arguments
    params = setupArgs()["INFERENCE"]

    ckpt_path = params["CKPT_PATH"]
    input_path = params["INPUT_PATH"]
    input_name = params["INPUT_NAME"]
    save_path = params["SAVE_PATH"]
    crop_size = int(params["CROP_SIZE"])
    overlap = int(params["OVERLAP"])
    mode = params["MODE"]
    term1 = int(params["TERM1"])
    term2 = int(params["TERM2"])
    contrast_correction = bool(int(params["CONTRAST_CORRECTION"]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")



    # TILE INPUT IMAGE, SAVE CROPS TO FILE
    start = time.time()
    tiles, coords, image = inference.tile_input_image(name = input_name,
                                                      root_path = input_path,
                                                      size = crop_size,
                                                      overlap = overlap,
                                                      scale = 1.,
                                                      mode = mode,
                                                      term1 = term1,
                                                      term2 = term2,
                                                      contrast_correction = contrast_correction)
    
    conditionalMkDir(save_path)
    conditionalMkDir(os.path.join(save_path, "crops"))

    saveCrops(save_to = os.path.join(save_path, "crops"),
              crops_set = (255*tiles[:,:,:,0]).astype(int),
              centers_set = coords,
              prefix = input_name,
              suffix = "")



    # LOAD TRAINED MODEL, INFERENCE

    # load model checkpoint
    teacher_net = modified_resnet18(pretrained=True).to(device)
    student_net = modified_resnet18(pretrained=False).to(device)

    for param in teacher_net.parameters():
        param.requires_grad = False
    _ = teacher_net.eval()
    _ = student_net.eval()

    student_net.load_state_dict(torch.load(ckpt_path))

    # inference
    anomaly_maps = []
    with torch.no_grad():
        inputs = np.expand_dims(np.transpose(tiles, (0,3,1,2)), 1)

        for i, x in enumerate(torch.Tensor(inputs)):
            # forward pass
            features_t = teacher_net(x.to(device))
            features_s = student_net(x.to(device))

            a_map = compute_anomaly_maps(features_s, features_t, out_size=crop_size)
            a_map = gaussian_filter(a_map, sigma=1)
            a_map = np.clip((a_map*4), 0., 1.)

            anomaly_maps.append(a_map)

    anomaly_maps = np.array(anomaly_maps)
    anomaly_peaks = np.max(anomaly_maps, axis=(1,2))

    print(f"Elapsed time: {(time.time()-start):2f} s")

    # SAVE RESULTS

    saveCrops(save_to = os.path.join(save_path, "crops"),
              crops_set = (255*anomaly_maps).astype(np.uint8),
              centers_set = coords,
              prefix = input_name,
              suffix = "_AMAP",
              mode = "color_map") 

    # sorted arrays
    sorted_input_images = tiles[coords[:,0].argsort()] 
    sorted_anomaly_maps = anomaly_maps[coords[:,0].argsort()] 
    sorted_anomaly_peaks = anomaly_peaks[coords[:,0].argsort()]  
    sorted_coords = coords[coords[:,0].argsort()] 

    sorted_input_images = sorted_input_images[sorted_coords[:,1].argsort(kind='mergesort')]
    sorted_anomaly_maps = sorted_anomaly_maps[sorted_coords[:,1].argsort(kind='mergesort')]
    sorted_anomaly_peaks = sorted_anomaly_peaks[sorted_coords[:,1].argsort(kind='mergesort')]
    sorted_coords = sorted_coords[sorted_coords[:,1].argsort(kind='mergesort')]

    # save to file
    np.save(os.path.join(save_path, "inputs_set.npy"), sorted_input_images)
    np.save(os.path.join(save_path, "coords_set.npy"), sorted_coords)
    np.save(os.path.join(save_path, "anomaly_maps_set.npy"), sorted_anomaly_maps)
    np.save(os.path.join(save_path, "anomaly_scores_set.npy"), sorted_anomaly_peaks)

    inference.save_anomaly_hist(anomaly_peaks,
                                os.path.join(save_path, "anomaly_hist_log.png"))
    inference.save_anomaly_hist(anomaly_peaks,
                                os.path.join(save_path, "anomaly_hist.png"))
        
    inference.save_anomaly_hist_pixelwise(anomaly_maps.reshape(-1)[:],
                                          os.path.join(save_path, "anomaly_hist_pixelwise.png"))

    inference.save_anomaly_heatmap(sorted_coords,
                                   sorted_anomaly_peaks,
                                   os.path.join(save_path, "anomaly_heatmap.png"))
        
    inference.save_annotated_image(image,
                                   crop_size,
                                   sorted_coords,
                                   sorted_anomaly_peaks,
                                   os.path.join(save_path, "annotated_image.png"),
                                   threshold = 0.2)

    inference.compose_anomalyImage(sorted_anomaly_maps,
                                   sorted_coords,
                                   save_path=os.path.join(save_path, "anomaly_heatmap_tot_C.png"),
                                   image_shape=[7750,2048],
                                   crop_size=crop_size,
                                   false_color=True)
    inference.compose_anomalyImage(sorted_anomaly_maps,
                                   sorted_coords,
                                   save_path=os.path.join(save_path, "anomaly_heatmap_tot.png"),
                                   image_shape=[7750,2048],
                                   crop_size=crop_size,
                                   false_color=False)


