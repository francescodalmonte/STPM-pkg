import os
import numpy as np 
from matplotlib import pyplot as plt
import cv2 as cv
from scipy.ndimage import gaussian_filter, median_filter

from preprocessing_tele.multiChannelImage import multiChannelImage
from preprocessing_tele.image_processing import tileImage
from preprocessing_tele.dataset import conditionalMkDir


def tile_input_image(name: str,
                     root_path: str,
                     size: int = 224, 
                     overlap: int = 0,
                     scale: float = 1.,
                     mode: str = "diff",
                     term1: int = 2,
                     term2: int = 1,
                     wmask: bool = False,
                     contrast_correction: bool = False,
                     contrast_correction_sigma: float = 50
                     ):
    
    print("Tiling image")
    object = multiChannelImage(name, root_path)
    # images
    image = object.image_mode_selector(mode=mode,
                                       scale=scale,
                                       term1=term1,
                                       term2=term2,
                                       contrast_correction=contrast_correction,
                                       contrast_correction_sigma=contrast_correction_sigma
                                       )  
    if wmask:
        mask = object.__get_anomalousMask__(scale=scale)
        print(f"Mask shape: {mask.shape}")
        image = np.concatenate((image, np.expand_dims(mask, axis=2)), axis = 2)
        #image = image[10:, :1490] #for comparison w specialvideo results

    tiles, coords = tileImage(image, size, overlap, normalize = False, gauss_blur = 0., saturate_mask = False)
    print(tiles.shape)

    return tiles/255, coords, image




def save_anomaly_hist(anomaly_scores_set: np.ndarray,
                      save_path: str,
                      log: bool = True
                      ):
    d = np.clip(anomaly_scores_set, 0, 1)
    plt.hist(anomaly_scores_set, bins = 50, range = (-0.01, 1.01))
    if log:
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
    ax[0].imshow(h, vmin = 0., vmax = 1.); ax[0].set_title("Anomaly scores")
    ax[1].imshow(h**2, vmin = 0., vmax = 1.); ax[1].set_title("Anomaly scores (pow2)")
    
    plt.savefig(save_path)




def save_annotated_image(image: np.ndarray,
                         size: int,
                         coords_set: np.ndarray,
                         anomaly_scores_set: np.ndarray,
                         save_path: str,
                         threshold: float = 0.1
                         ):
    # image
    print(image.shape)
    CVimage = np.array(image).copy()#.transpose(1,2,0)

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
                         save_path: str,
                         image_shape: list,
                         crop_size: int,
                         false_color: bool = True):
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
    anomaly_image = (255*(anomaly_image**2)).astype(np.uint8)
    anomaly_image = gaussian_filter(anomaly_image, sigma=1)
    # save image to file
    if false_color:
        anomaly_image = cv.applyColorMap(anomaly_image, cv.COLORMAP_JET)
    cv.imwrite(save_path, anomaly_image)




def jaccard_similarity(array1, array2):
    """
    Compute Jaccard similarity index between two binary arrays.
    https://en.wikipedia.org/wiki/Jaccard_index
    """

    array1, array2 = (array1 > 0).astype(np.int_), (array2 > 0).astype(np.int_)

    intersection = array1*array2
    union = array1+array2-intersection
    return np.sum(intersection)/np.sum(union)



def defects_class(array):
    m = np.max(array)
    if m<80: return "1"
    elif m<200: return "2"
    else: return "3"


def evaluate_detection(anomaly_maps: np.ndarray,
                       tiles_masks: np.ndarray,
                       tiles: np.ndarray,
                       threshold: float,
                       save_eval_crops: bool,
                       save_path: str
                       ):
    
    TP, TN, FP, FN = 0, 0, 0, 0
    TP_defects_classes = {"1": 0, "2": 0, "3": 0}
    FN_defects_classes = {"1": 0, "2": 0, "3": 0}

    conditionalMkDir(os.path.join(save_path, "TPs"))
    conditionalMkDir(os.path.join(save_path, "FPs"))
    conditionalMkDir(os.path.join(save_path, "FNs"))
    for i, (amap, mask, tile) in enumerate(zip(anomaly_maps, tiles_masks, tiles)):
        amap_th = amap > threshold
        mask_l = mask > 0
        if np.any(mask_l):
            if np.any(np.logical_and(amap_th, mask_l)):
                path = os.path.join(save_path, "TPs")
                if save_eval_crops:
                    cv.imwrite(os.path.join(path, f"crop_{i}.png"), (tile*255).astype(np.uint8))
                    cv.imwrite(os.path.join(path, f"crop_{i}_Ma.png"), amap_th.astype(np.uint8)*255)
                    cv.imwrite(os.path.join(path, f"crop_{i}_M.png"), (mask*255).astype(np.uint8))
                TP_defects_classes[defects_class(mask*255)]+=1
                TP+=1
            else:
                path = os.path.join(save_path, "FNs")
                if save_eval_crops:
                    cv.imwrite(os.path.join(path, f"crop_{i}.png"), (tile*255).astype(np.uint8))
                    cv.imwrite(os.path.join(path, f"crop_{i}_Ma.png"), amap_th.astype(np.uint8)*255)
                    cv.imwrite(os.path.join(path, f"crop_{i}_M.png"), (mask*255).astype(np.uint8))
                FN_defects_classes[defects_class(mask*255)]+=1
                FN+=1
        else:
            if np.any(amap_th):
                path = os.path.join(save_path, "FPs")
                if save_eval_crops:
                    cv.imwrite(os.path.join(path, f"crop_{i}.png"), (tile*255).astype(np.uint8))
                    cv.imwrite(os.path.join(path, f"crop_{i}_Ma.png"), amap_th.astype(np.uint8)*255)
                    cv.imwrite(os.path.join(path, f"crop_{i}_M.png"), (mask*255).astype(np.uint8))
                FP+=1
            else:
                TN+=1
    
    return TP, TN, FP, FN, TP_defects_classes, FN_defects_classes