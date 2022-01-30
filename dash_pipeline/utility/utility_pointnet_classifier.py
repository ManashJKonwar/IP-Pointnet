__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import os
import sys
import glob
import tqdm
import trimesh
import numpy as np
import pandas as pd
from tensorflow import keras
from modelling_pipeline.modelling.train_pointnet_classifier import generate_pointnet_model

def load_classifier_dataset(dataset_directory=None, num_points=None):
    """
    Read all meshed off files per 10 classes and return the numpy arrays for the same
    Parameters: 
        dataset_directory (str): http link for downloading point cloud dataset
        num_points (int): number of points by which each mesh file needs to be generated for
                        converting it into point mesh data
    Returns: 
        tuple of numpy arrays representating all files under test folders for each classes
    """
    test_points = []
    test_labels = []
    class_map = {}
    folders = [os.path.join(dataset_directory, o) for o in os.listdir(dataset_directory) if os.path.isdir(os.path.join(dataset_directory,o))]

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("\\")[-1] if sys.platform == 'win32' else folder.split("/")[-1]
        # gather all files
        # train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)
    
    return (
        np.array(test_points), 
        np.array(test_labels),
        class_map)

def load_classifier_model(model_wts_filename=None, **kwargs):
    num_points=kwargs.get('num_points')
    num_classes=kwargs.get('num_classes')

    trained_classifier_model = None
    if os.path.exists(model_wts_filename):
        trained_classifier_model = generate_pointnet_model(num_points=num_points,
                                                            num_classes=num_classes)
        trained_classifier_model.load_weights(model_wts_filename)
    
    return trained_classifier_model

def load_classifier_training_history(model_history_filename=None):
    trained_classifier_history = None
    if os.path.exists(model_history_filename):
        trained_classifier_history = pd.read_csv(model_history_filename)

    return trained_classifier_history