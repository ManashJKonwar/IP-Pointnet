__author__ = "konwar.m"
__copyright__ = "Copyright 2021, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import os, sys, glob
import pickle
import trimesh
import numpy as np
from numpy import save, load

#region Pointnet Classifier
NUM_POINTS = 2048
NUM_CLASSES = 10
CLASSIFIER_DATA_DIR = r'modelling_pipeline\datasets\ModelNet10'

def load_classifier_dataset(dataset_directory=None, num_points=None):
    """
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

test_points, test_labels, class_map = None, None, None

if not os.path.exists(r'dash_pipeline\datasets\pointnet_classifier\test_points.npy') \
or not os.path.exists(r'dash_pipeline\datasets\pointnet_classifier\test_labels.npy') \
or not os.path.exists(r'dash_pipeline\datasets\pointnet_classifier\class_map.pkl'):
    test_points, test_labels, class_map = load_classifier_dataset(dataset_directory=CLASSIFIER_DATA_DIR, num_points=NUM_POINTS)
    
    save(r'dash_pipeline\datasets\pointnet_classifier\test_points.npy', test_points)
    save(r'dash_pipeline\datasets\pointnet_classifier\test_labels.npy', test_labels)
    with open(r'dash_pipeline\datasets\pointnet_classifier\class_map.pkl', 'wb') as f:
        pickle.dump(class_map, f)
else:
    test_points = load(r'dash_pipeline\datasets\pointnet_classifier\test_points.npy')
    test_labels = load(r'dash_pipeline\datasets\pointnet_classifier\test_labels.npy')
    with open(r'dash_pipeline\datasets\pointnet_classifier\class_map.pkl', 'rb') as f:
        class_map = pickle.load(f)
#endregion