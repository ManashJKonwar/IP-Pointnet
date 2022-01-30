__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import os
import glob
import tqdm
import random
import numpy as np
import pandas as pd
from tensorflow import keras
from dash_pipeline.utility.data_preprocesser import generate_dataset
from modelling_pipeline.modelling.train_pointnet_part_segmenter import generate_pointnet_segmentation_model

def load_segmentation_dataset(dataset_directory=None, **kwargs):
    points_dir=kwargs.get('points_dir')
    labels_dir=kwargs.get('labels_dir')
    VAL_SPLIT=kwargs.get('VAL_SPLIT')
    NUM_SAMPLE_POINTS=kwargs.get('NUM_SAMPLE_POINTS')
    BATCH_SIZE=kwargs.get('BATCH_SIZE')
    EPOCHS=kwargs.get('EPOCHS')
    INITIAL_LR=kwargs.get('INITIAL_LR')
    LABELS=kwargs.get('LABELS')
    COLORS=kwargs.get('COLORS')

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
        
        return point_clouds, point_cloud_labels, test_point_clouds, all_labels
    
    def preprocess_segmentation_dataset(**kwargs):
        """
        Point clouds loaded have inconistent number of points and it becomes difficult to batch them together,
        However, this can be overcomed by randomly sampling a fixed number of points from each point cloud. 
        Normalizing the point cloud is also done to remove data invariance issue.
        Parameters: 
            kwargs (multiple arguments): point_clouds, point_cloud_labels, all_labels, NUM_SAMPLE_POINTS 
        Returns: 
            tuple of numpy arrays representating all files under train and test folders for each classes
        """
        point_clouds = kwargs.get('point_clouds')
        point_cloud_labels = kwargs.get('point_cloud_labels')
        all_labels = kwargs.get('all_labels')
        NUM_SAMPLE_POINTS = kwargs.get('NUM_SAMPLE_POINTS')

        for index in tqdm.tqdm(range(len(point_clouds))):
            current_point_cloud = point_clouds[index]
            current_label_cloud = point_cloud_labels[index]
            current_labels = all_labels[index]
            num_points = len(current_point_cloud)
            # Randomly sampling respective indices.
            sampled_indices = random.sample(list(range(num_points)), NUM_SAMPLE_POINTS)
            # Sampling points corresponding to sampled indices.
            sampled_point_cloud = np.array([current_point_cloud[i] for i in sampled_indices])
            # Sampling corresponding one-hot encoded labels.
            sampled_label_cloud = np.array([current_label_cloud[i] for i in sampled_indices])
            # Sampling corresponding labels for visualization.
            sampled_labels = np.array([current_labels[i] for i in sampled_indices])
            # Normalizing sampled point cloud.
            norm_point_cloud = sampled_point_cloud - np.mean(sampled_point_cloud, axis=0)
            norm_point_cloud /= np.max(np.linalg.norm(norm_point_cloud, axis=1))
            point_clouds[index] = norm_point_cloud
            point_cloud_labels[index] = sampled_label_cloud
            all_labels[index] = sampled_labels

        return point_clouds, point_cloud_labels, all_labels

    def generate_segmentation_dataset(**kwargs):
        """
        Convert numpy arrays into tensorflow datasets
        Parameters: 
            kwargs (multiple arguments): point_clouds, point_cloud_labels, all_labels, NUM_SAMPLE_POINTS 
        Returns: 
            tuple of numpy arrays representating all files under train and test folders for each classes
        """
        point_clouds = kwargs.get('point_clouds')
        point_cloud_labels = kwargs.get('point_cloud_labels')
        VAL_SPLIT = kwargs.get('VAL_SPLIT')
        BATCH_SIZE = kwargs.get('BATCH_SIZE')

        split_index = int(len(point_clouds) * (1 - VAL_SPLIT))
        train_point_clouds = point_clouds[:split_index]
        train_label_cloud = point_cloud_labels[:split_index]
        total_training_examples = len(train_point_clouds)

        val_point_clouds = point_clouds[split_index:]
        val_label_cloud = point_cloud_labels[split_index:]

        print("Num train point clouds:", len(train_point_clouds))
        print("Num train point cloud labels:", len(train_label_cloud))
        print("Num val point clouds:", len(val_point_clouds))
        print("Num val point cloud labels:", len(val_label_cloud))

        # train_dataset = generate_dataset(point_clouds=train_point_clouds, label_clouds=train_label_cloud)
        val_dataset = generate_dataset(point_clouds=val_point_clouds, label_clouds=val_label_cloud, is_training=False, BATCH_SIZE=BATCH_SIZE)

        # print("Train Dataset:", train_dataset)
        print("Validation Dataset:", val_dataset)

        return val_dataset

    point_clouds, point_cloud_labels, test_point_clouds, all_labels = parse_segmentation_dataset(dataset_directory=dataset_directory,
                                                                                                points_dir=points_dir,
                                                                                                labels_dir=labels_dir,
                                                                                                VAL_SPLIT=VAL_SPLIT,
                                                                                                NUM_SAMPLE_POINTS=NUM_SAMPLE_POINTS,
                                                                                                BATCH_SIZE=BATCH_SIZE,
                                                                                                EPOCHS=EPOCHS,
                                                                                                INITIAL_LR=INITIAL_LR,
                                                                                                LABELS=LABELS,
                                                                                                COLORS=COLORS)
    
    # Preprocess the point cloud to perform random batching and also performing normalization on top of that
    point_clouds, point_cloud_labels, all_labels = preprocess_segmentation_dataset(point_clouds=point_clouds,
                                                                                point_cloud_labels=point_cloud_labels,
                                                                                all_labels=all_labels,
                                                                                NUM_SAMPLE_POINTS=NUM_SAMPLE_POINTS)
    
    # Data Augmentation to convert the numpy arrays of train and test dataset into tensorflow based dataset format
    validation_dataset = generate_segmentation_dataset(point_clouds=point_clouds,
                                                    point_cloud_labels=point_cloud_labels,
                                                    VAL_SPLIT=VAL_SPLIT,
                                                    BATCH_SIZE=BATCH_SIZE)
    
    return validation_dataset
    
def load_part_segmenter_model(model_wts_filename=None):
    trained_part_segmenter_model = None
    if os.path.exists(model_wts_filename):
        trained_part_segmenter_model = generate_pointnet_segmentation_model(num_points=1024,
                                                                            num_classes=5)
        trained_part_segmenter_model.load_weights(model_wts_filename)
    
    return trained_part_segmenter_model

def load_part_segmenter_training_history(model_history_filename=None):
    trained_part_segmenter_history = None
    if os.path.exists(model_history_filename):
        trained_part_segmenter_history = pd.read_csv(model_history_filename)

    return trained_part_segmenter_history