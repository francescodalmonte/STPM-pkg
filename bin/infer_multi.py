import numpy as np
import os
import time
import configparser
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt

import torch

from preprocessing_tele.image_processing import saveCrops
from preprocessing_tele.dataset import conditionalMkDir
from preprocessing_tele.multiChannelImage import multiChannelImage

from STPM_model.model import modified_resnet18
from STPM_model.training import compute_anomaly_maps
from STPM_model import inference



def setupArgs():
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), "config.INI")
    if os.path.isfile(config_path):
        config.read(config_path)
    else:
        raise ValueError(f"can't find configuration file {config_path}")
    
    return config



if __name__ == "__main__":


    # setup input arguments
    params = setupArgs()["INFERENCE_MULTI"]

    ckpt_path_R1 = params["CKPT_PATH_R1"]
    ckpt_path_R3 = params["CKPT_PATH_R3"]
    mask_path_R1 = params["MASK_PATH_R1"]
    mask_path_R3 = params["MASK_PATH_R3"]
    input_path = params["INPUT_PATH"]
    input_name = params["INPUT_NAME"]
    save_path = params["SAVE_PATH"]
    overlap = int(params["OVERLAP"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    mcImg = multiChannelImage(input_name, input_path)
    mask_R1 = mcImg.__get_allignedRegionMask__(mask_path_R1, scale=1.)
    mask_R3 = mcImg.__get_allignedRegionMask__(mask_path_R3, scale=1.)


    # TILE INPUT IMAGE, SAVE CROPS TO FILE
    start = time.time()
    tiles, coords, image = inference.tile_input_image(name = input_name,
                                                      root_path = input_path,
                                                      size = 224,
                                                      overlap = overlap,
                                                      scale = 1.)
    
    conditionalMkDir(save_path)
    conditionalMkDir(os.path.join(save_path, "crops"))

    saveCrops(save_to = os.path.join(save_path, "crops"),
              crops_set = (256*tiles[:,:,:,0]).astype(int),
              centers_set = coords,
              prefix = input_name,
              suffix = "")



    # LOAD TRAINED MODEL, INFERENCE

    # load model checkpoint
    teacher_net = modified_resnet18(pretrained=True).to(device)
    student_net_R1 = modified_resnet18(pretrained=False).to(device)
    student_net_R3 = modified_resnet18(pretrained=False).to(device)

    for param in teacher_net.parameters():
        param.requires_grad = False
    _ = teacher_net.eval()
    _ = student_net_R1.eval()
    _ = student_net_R3.eval()

    student_net_R1.load_state_dict(torch.load(ckpt_path_R1))
    student_net_R3.load_state_dict(torch.load(ckpt_path_R3))


    # inference
    anomaly_maps = []
    with torch.no_grad():
        inputs = np.expand_dims(np.transpose(tiles, (0,3,1,2)), 1)

        for i, x in enumerate(torch.Tensor(inputs)):
            # forward pass
            features_t = teacher_net(x.to(device))
            if mask_R1[coords[i][1], coords[i][0]]:
                features_s = student_net_R1(x.to(device))
            elif mask_R3[coords[i][1], coords[i][0]]:
                features_s = student_net_R3(x.to(device))
            else:
                features_s = student_net_R3(x.to(device))

            a_map = compute_anomaly_maps(features_s, features_t)
            a_map = gaussian_filter(a_map, sigma=3)
            a_map = ((15*a_map)**3)/15

            anomaly_maps.append(a_map)

    anomaly_maps = np.array(anomaly_maps)
    anomaly_peaks =  np.max(anomaly_maps, axis=(1,2))

    print(f"Elapsed time: {(time.time()-start):2f} s")

    # SAVE RESULTS

    saveCrops(save_to = os.path.join(save_path, "crops"),
              crops_set = (256*anomaly_maps).astype(np.uint8),
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
                                os.path.join(save_path, "anomaly_hist.png"))
        
    inference.save_anomaly_hist_pixelwise(anomaly_maps.reshape(-1)[:],
                                          os.path.join(save_path, "anomaly_hist_pixelwise.png"))

    inference.save_anomaly_heatmap(sorted_coords,
                                   sorted_anomaly_peaks,
                                   os.path.join(save_path, "anomaly_heatmap.png"))
        
    inference.save_annotated_image(image,
                                   224,
                                   sorted_coords,
                                   sorted_anomaly_peaks,
                                   os.path.join(save_path, "annotated_image.png"),
                                   threshold = 0.2)

    inference.compose_anomalyImage(sorted_anomaly_maps,
                                   sorted_coords,
                                   save_path=os.path.join(save_path, "anomaly_heatmap_tot.png"),
                                   image_shape=[7750,2048],
                                   crop_size=224)


