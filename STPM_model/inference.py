import os
import numpy as np 
from matplotlib import pyplot as plt
import cv2 as cv

from preprocessing_tele.multiChannelImage import multiChannelImage
from preprocessing_tele.image_processing import tileImage


def tile_input_image(name: str,
                     root_path: str,
                     size: int = 224, 
                     overlap: int = 0,
                     scale: float = 1.
                     ):
    
    print("Tiling image")
    object = multiChannelImage(name, root_path)
    image = object.__get_diffImage__(scale = scale, minuend = 3, subtrahend = 0)

    tiles, coords = tileImage(image, size, overlap, gauss_blur = .0)
    tiles = np.stack((tiles, tiles, tiles)).transpose(1,2,3,0)

    return tiles/255, coords, image




def save_anomaly_hist(anomaly_scores_set: np.ndarray,
                      save_path: str
                      ):
    d = np.clip(anomaly_scores_set, 0, 1)
    plt.hist(anomaly_scores_set, bins = 50, range = (-0.01, 1.01))
    plt.yscale("log")
    plt.title("Anomaly scores (clipped [0.0:1.0])")
    plt.savefig(save_path)




def save_anomaly_hist_pixelwise(anomaly_maps_set: np.ndarray,
                                save_path: str
                                ):
    plt.hist(anomaly_maps_set, bins = 50)
    plt.yscale("log")
    plt.title("Pixels anomaly scores")
    plt.savefig(save_path)




def save_anomaly_heatmap(coords_set: np.ndarray,
                         anomaly_scores_set: np.ndarray,
                         save_path: str
                        ):
    
    h = anomaly_scores_set.reshape(len(np.unique(coords_set[:,1])), len(np.unique(coords_set[:,0])))
        
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,12), dpi=600)
    ax[0].imshow(h, vmin = 0., vmax = 0.5); ax[0].set_title("Anomaly scores")
    ax[1].imshow(np.sqrt(h), vmin = 0., vmax = 0.5); ax[1].set_title("Anomaly scores (sqrt)")
    
    plt.savefig(save_path)




def save_annotated_image(image: np.ndarray,
                         size: int,
                         coords_set: np.ndarray,
                         anomaly_scores_set: np.ndarray,
                         save_path: str,
                         threshold: float = 0.1
                         ):
    # image
    CVimage = np.array(np.stack((image, image, image)).transpose(1,2,0)).copy()

    # threshold mask
    m = anomaly_scores_set > threshold

    for c, s in zip(coords_set[m], anomaly_scores_set[m]):
        # draw rectangle
        topleft = (int(c[0])-size//2, int(c[1])-size//2)
        bottomright = (int(c[0])+size//2, int(c[1])+size//2)

        # map color according to anomaly score
        r = int(220*np.clip(s, 0, 1))
        g = 0
        b = int(r*0.1)
        w = int(8*np.clip(s,0,1))
        tw = int(3*np.clip(s,0,1))
        color = (b,g,r)

        cv.rectangle(CVimage, topleft, bottomright, color, w)
        cv.putText(CVimage, f"{s:.5f}", (topleft[0]+5, topleft[1]-5), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), tw)

    # save image to file
    cv.imwrite(save_path, CVimage)




def save_info(config: dict,
              save_path: str):

    with open(save_path,"w") as file:
        for k in config.keys():
            file.write(k + " >>> " + str(config[k]) + "\n")




def read_infotxt(path: str,
                 filename: str = "INFO.txt"):
    """
    Fetch metadata from the INFO.txt file.
    """
    with open(os.path.join(path, filename), "r") as file:
        info_dict = {}
        for line in file:
            key, value = line.split(" >>> ")
            info_dict[key] = value.strip()

    return info_dict




def compose_anomalyImage(anomaly_maps: np.ndarray,
                         coords: np.ndarray,
                         image_shape: list,
                         crop_size: int):
    """
    Aseemble a set of crop-sized anomaly maps from FCDD results
    to create a full scale anomaly heatmap.
    """
    anomaly_image = np.zeros(image_shape)
    count = np.zeros(image_shape)

    for i, c in enumerate(coords):
        top, bottom = c[0]-crop_size//2, c[0]+crop_size//2
        left, right = c[1]-crop_size//2, c[1]+crop_size//2

        anomaly_image[left:right, top:bottom]+=anomaly_maps[i]
        count[left:right, top:bottom]+=1

    anomaly_image = (np.divide(anomaly_image, count, where=count!=0))

    return anomaly_image




def jaccard_similarity(array1, array2):
    """
    Compute Jaccard similarity index between two binary arrays.
    https://en.wikipedia.org/wiki/Jaccard_index
    """

    array1, array2 = (array1 > 0).astype(np.int_), (array2 > 0).astype(np.int_)

    intersection = array1*array2
    union = array1+array2-intersection
    return np.sum(intersection)/np.sum(union)