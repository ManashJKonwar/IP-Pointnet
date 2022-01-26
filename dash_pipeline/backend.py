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
import json
import tqdm
import pickle
import trimesh
import numpy as np
import pandas as pd
from numpy import save, load
from tensorflow import keras
from modelling_pipeline.modelling.train_pointnet_classifier import generate_pointnet_model

#region Pointnet Classifier
NUM_POINTS = 2048
NUM_CLASSES = 10
CLASSIFIER_DATA_DIR = r'modelling_pipeline\datasets\ModelNet10'

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

def load_classifier_model(model_wts_filename=None):
    trained_classifier_model = None
    if os.path.exists(model_wts_filename):
        trained_classifier_model = generate_pointnet_model(num_points=NUM_POINTS,
                                                        num_classes=NUM_CLASSES)
        trained_classifier_model.load_weights(model_wts_filename)
    
    return trained_classifier_model

trained_classifier_model = None
trained_classifier_model = load_classifier_model(model_wts_filename=r'modelling_pipeline\models\pointnet_classifier_10cls.h5')

def load_training_history(model_history_filename=None):
    trained_classifier_history = None
    if os.path.exists(model_history_filename):
        trained_classifier_history = pd.read_csv(model_history_filename)

    return trained_classifier_history

trained_classifier_history = None
trained_classifier_history = load_training_history(model_history_filename=r'modelling_pipeline\models\pointnet_classifier_10cls_history.csv')
#endregion

#region Pointnet Part Segmenter
VAL_SPLIT = 0.2
NUM_SAMPLE_POINTS = 1024
BATCH_SIZE = 32
EPOCHS = 60
INITIAL_LR = 1e-3
PART_SEGMENTER_DATA_DIR = r'modelling_pipeline\datasets\PartAnnotation'
object_name='Airplane'

with open(os.path.join(PART_SEGMENTER_DATA_DIR,'metadata.json')) as json_file:
    metadata = json.load(json_file)
# Data Reading which focus to specific object
points_dir = os.path.join(PART_SEGMENTER_DATA_DIR, metadata[object_name]["directory"], 'points')
labels_dir = os.path.join(PART_SEGMENTER_DATA_DIR, metadata[object_name]["directory"], 'points_label')
LABELS = metadata[object_name]["lables"]
COLORS = metadata[object_name]["colors"]

def load_segmentation_dataset(dataset_directory=None, **kwargs):
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
    
    return point_clouds, point_cloud_labels, test_point_clouds, all_labels

if not os.path.exists(r'dash_pipeline\datasets\pointnet_part_segmenter\%s\point_clouds.npy' %(object_name.lower())) \
or not os.path.exists(r'dash_pipeline\datasets\pointnet_part_segmenter\%s\point_cloud_labels.npy' %(object_name.lower())) \
or not os.path.exists(r'dash_pipeline\datasets\pointnet_part_segmenter\%s\test_point_clouds.npy' %(object_name.lower())) \
or not os.path.exists(r'dash_pipeline\datasets\pointnet_part_segmenter\%s\all_labels.npy' %(object_name.lower())):
    if not os.path.exists(os.path.join('dash_pipeline\datasets\pointnet_part_segmenter',object_name.lower())):
        os.makedirs(os.path.join('dash_pipeline\datasets\pointnet_part_segmenter',object_name.lower()))

    point_clouds, point_cloud_labels, test_point_clouds, all_labels = load_segmentation_dataset(dataset_directory=PART_SEGMENTER_DATA_DIR,
                                                                                                points_dir=points_dir,
                                                                                                labels_dir=labels_dir,
                                                                                                VAL_SPLIT=VAL_SPLIT,
                                                                                                NUM_SAMPLE_POINTS=NUM_SAMPLE_POINTS,
                                                                                                BATCH_SIZE=BATCH_SIZE,
                                                                                                EPOCHS=EPOCHS,
                                                                                                INITIAL_LR=INITIAL_LR,
                                                                                                LABELS=LABELS,
                                                                                                COLORS=COLORS)

    save(r'dash_pipeline\datasets\pointnet_part_segmenter\%s\point_clouds.npy' %(object_name.lower()) , point_clouds)
    save(r'dash_pipeline\datasets\pointnet_part_segmenter\%s\point_cloud_labels.npy' %(object_name.lower()), point_cloud_labels)
    save(r'dash_pipeline\datasets\pointnet_part_segmenter\%s\test_point_clouds.npy' %(object_name.lower()), test_point_clouds)
    save(r'dash_pipeline\datasets\pointnet_part_segmenter\%s\all_labels.npy' %(object_name.lower()), all_labels)
else:
    point_clouds = load(r'dash_pipeline\datasets\pointnet_part_segmenter\%s\point_clouds.npy' %(object_name.lower()), allow_pickle=True)
    point_cloud_labels = load(r'dash_pipeline\datasets\pointnet_part_segmenter\%s\point_cloud_labels.npy' %(object_name.lower()), allow_pickle=True)
    test_point_clouds = load(r'dash_pipeline\datasets\pointnet_part_segmenter\%s\test_point_clouds.npy' %(object_name.lower()), allow_pickle=True)
    all_labels = load(r'dash_pipeline\datasets\pointnet_part_segmenter\%s\all_labels.npy' %(object_name.lower()), allow_pickle=True)
#endregion