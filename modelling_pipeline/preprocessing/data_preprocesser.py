__author__ = "konwar.m"
__copyright__ = "Copyright 2021, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import os, sys, glob, tqdm
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras

#region Pointnet Classifier
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
#endregion

#region Pointnet Part Segmenter
def parse_segmentation_dataset(dataset_directory=None, **kwargs):
    """
    Parse the dataset metadata in order to easily map model categories to their respective directories 
    and segmentation classes to colors for the purpose of visualization.
    
    We generate the following in-memory data structures from the Airplane point clouds and their labels:
    1. point_clouds is a list of np.array objects that represent the point cloud data in the form of x, y and z coordinates. 
        Axis 0 represents the number of points in the point cloud, while axis 1 represents the coordinates. 
        all_labels is the list that represents the label of each coordinate as a string (needed mainly for visualization purposes).
    2. test_point_clouds is in the same format as point_clouds, but doesn't have corresponding the labels of the point clouds.
    3. all_labels is a list of np.array objects that represent the point cloud labels for each coordinate, 
        corresponding to the point_clouds list.
    4. point_cloud_labels is a list of np.array objects that represent the point cloud labels for each coordinate in one-hot encoded form, 
        corresponding to the point_clouds list.
    Parameters: 
        dataset_directory (str): http link for downloading point cloud dataset
    Returns: 
        tuple of numpy arrays representating all files under train and test folders for each classes
    """
    point_clouds, test_point_clouds = [], []
    point_cloud_labels, all_labels = [], []

    points_dir = kwargs.get('points_dir')
    labels_dir = kwargs.get('labels_dir')
    VAL_SPLIT = kwargs.get('VAL_SPLIT')
    NUM_SAMPLE_POINTS = kwargs.get('NUM_SAMPLE_POINTS')
    BATCH_SIZE = kwargs.get('BATCH_SIZE')
    EPOCHS = kwargs.get('EPOCHS')
    INITIAL_LR = kwargs.get('INITIAL_LR')
    LABELS = kwargs.get('LABELS')
    COLORS = kwargs.get('COLORS')

    points_files = glob.glob(os.path.join(points_dir, "*.pts"))
    for point_file in tqdm.tqdm(points_files):
        point_cloud = np.loadtxt(point_file)
        if point_cloud.shape[0] < NUM_SAMPLE_POINTS:
            continue

        # Get the file-id of the current point cloud for parsing its
        # labels.
        file_id = os.path.basename(point_file).split('.')[0]
        label_data, num_labels = {}, 0
        for label in LABELS:
            label_file = os.path.join(labels_dir, label, file_id + ".seg")
            if os.path.exists(label_file):
                label_data[label] = np.loadtxt(label_file).astype("float32")
                num_labels = len(label_data[label])

        # Point clouds having labels will be our training samples.
        try:
            label_map = ["none"] * num_labels
            for label in LABELS:
                for i, data in enumerate(label_data[label]):
                    label_map[i] = label if data == 1 else label_map[i]
            label_data = [
                LABELS.index(label) if label != "none" else len(LABELS)
                for label in label_map
            ]
            # Apply one-hot encoding to the dense label representation.
            label_data = keras.utils.to_categorical(label_data, num_classes=len(LABELS) + 1)

            point_clouds.append(point_cloud)
            point_cloud_labels.append(label_data)
            all_labels.append(label_map)
        except KeyError:
            test_point_clouds.append(point_cloud)
    
    return point_cloud, point_cloud_labels, test_point_clouds, all_labels
#endregion