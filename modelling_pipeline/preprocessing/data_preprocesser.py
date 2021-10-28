__author__ = "konwar.m"
__copyright__ = "Copyright 2021, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import os, sys, glob
import trimesh
import numpy as np
import tensorflow as tf

def parse_dataset(dataset_directory=None, num_points=2048):
    """
    Read all meshed off files per 10 classes and return the numpy arrays for the same
    Parameters: 
        dataset_directory (str): http link for downloading point cloud dataset
        num_points (int): number of points by which each mesh file needs to be generated for
                        converting it into point mesh data
    Returns: 
        tuple of numpy arrays representating all files under train and test folders for each classes
    """
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = [os.path.join(dataset_directory, o) for o in os.listdir(dataset_directory) if os.path.isdir(os.path.join(dataset_directory,o))]

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("\\")[-1] if sys.platform == 'win32' else folder.split("/")[-1]
        # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )

def augment_dataset(train_points=None, test_points=None, train_labels=None, test_labels=None, batch_size=32):
    """
    Augment the train and test dataset
    Parameters: 
        dataset_directory (str): http link for downloading point cloud dataset
        num_points (int): number of points by which each mesh file needs to be generated for
                        converting it into point mesh data
    Returns: 
        tuple of numpy arrays representating all files under train and test folders for each classes
    """
    def augment(points, label):
        # jitter points
        points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
        # shuffle points
        points = tf.random.shuffle(points)
        return points, label


    train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

    train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(batch_size)
    test_dataset = test_dataset.shuffle(len(test_points)).batch(batch_size)

    return (
        train_dataset,
        test_dataset
    )