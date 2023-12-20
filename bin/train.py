import numpy as np
import os
import time
import configparser

from STPM_model.dataset import FilterClothsDataset

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
    config = setupArgs()["DEFAULT"]



    print(f"Elapsed time: {(time.time()-start):2f} s")