import numpy as np
import os
import time
import configparser

import torch

from preprocessing_tele.image_processing import saveCrops
from preprocessing_tele.dataset import conditionalMkDir


from STPM_model import utils
from STPM_model.dataset import FilterClothsDataset
from STPM_model.model import modified_resnet18
from STPM_model.training import test_student_model
from STPM_model.inference import tile_input_image

def setupArgs():
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), "config.INI")
    if os.path.isfile(config_path):
        config.read(config_path)
    else:
        raise ValueError(f"can't find configuration file {config_path}")
    
    return config



if __name__ == "__main__":
    start = time.time()

    # setup input arguments
    params = setupArgs()["INFERENCE"]

    ckpt_path = params["CKPT_PATH"]
    input_path = params["INPUT_PATH"]
    input_name = params["INPUT_NAME"]
    save_path = params["SAVE_PATH"]
    overlap = int(params["OVERLAP"])


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")



    # TILE INPUT IMAGE, SAVE CROPS TO FILE

    tiles, coords, image = tile_input_image(name = input_name,
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



    # LOAD TRAINED MODEL, TEST STEP

    # load model checkpoint
    teacher_net = modified_resnet18(pretrained=True).to(device)
    student_net = modified_resnet18(pretrained=False).to(device)

    for param in teacher_net.parameters():
        param.requires_grad = False
    _ = teacher_net.eval() # teacher model will always remain in eval mode

    student_net.load_state_dict(torch.load(ckpt_path))

######################################## code is ok up to this line

    # test
    # loop on all the images in the "crops" directory, 1 at the time.
    # (write an ad hoc infer function to replace test_studet_model)
    results_dict = test_student_model(teacher_net,
                                      student_net,

                                      device)


    # RESULTS

########### dont know what's going on here

    # read coords from filename (original sorting of the array was messed up during save/load operations)
    unsorted_coords = []
    for item in results_dict:
        n = os.path.basename(item["image_path"][0]).split("(")[1].split(")")[0].split("-")
        unsorted_coords.append([int(n[0]),int(n[1])])
    unsorted_coords = np.array(unsorted_coords)

    input_images = np.array([item["image"][0][0].numpy() for item in predictions_dict])
    anomaly_maps = np.array([item["anomaly_maps"][0][0].numpy() for item in predictions_dict])
    anomaly_scores = np.array([item["pred_scores"][0].numpy() for item in predictions_dict])
    

    # sorted arrays
    # (they must be sorted together with the coords array we've read from file)
    sorted_input_images = input_images[unsorted_coords[:,0].argsort()] 
    sorted_anomaly_maps = anomaly_maps[unsorted_coords[:,0].argsort()] 
    sorted_anomaly_scores = anomaly_scores[unsorted_coords[:,0].argsort()]  
    sorted_coords = unsorted_coords[unsorted_coords[:,0].argsort()] 

    sorted_input_images = sorted_input_images[sorted_coords[:,1].argsort(kind='mergesort')]
    sorted_anomaly_maps = sorted_anomaly_maps[sorted_coords[:,1].argsort(kind='mergesort')]
    sorted_anomaly_scores = sorted_anomaly_scores[sorted_coords[:,1].argsort(kind='mergesort')]
    sorted_coords = sorted_coords[sorted_coords[:,1].argsort(kind='mergesort')]



    print(f"Elapsed time: {(time.time()-start):2f} s")