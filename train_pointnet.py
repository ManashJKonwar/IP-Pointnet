__author__ = "konwar.m"
__copyright__ = "Copyright 2021, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import os
import pickle
from numpy import load, save
from modelling_pipeline.preprocessing.data_preprocesser import augment_dataset, parse_dataset
from modelling_pipeline.modelling.train_pointnet_classifier import generate_pointnet_model, train_pointnet_classifier
from modelling_pipeline.utility.utility_datatransformation import download_dataset, save_model_weights

if __name__ == '__main__':
    TRAIN_POINTNET_CLASSIFIER = True
    TRAIN_POINTNET_PART_SEGMENTATOR = False
    TRAIN_POINTNET_SEMANTIC_SEGMANTATOR = False
    DATASET_URL = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
    DATASET_DIR = r"modelling_pipeline"

    # Downloading Point Cloud Dataset
    dataset_directory = download_dataset(dataset_url=DATASET_URL, dataset_directory=DATASET_DIR)
    
    # Training Point Cloud Classifier
    if TRAIN_POINTNET_CLASSIFIER:
        NUM_POINTS = 2048
        NUM_CLASSES = 10
        BATCH_SIZE = 32

        train_points, test_points, train_labels, test_labels, CLASS_MAP = None, None, None, None, None

        # Creating data points for this pointnet dataset by reading each mesh file using trimesh module and generating 2048
        # points on each of these files.
        if not os.path.exists(r'dash_pipeline\datasets\pointnet_classifier\train_points.npy') \
        or not os.path.exists(r'dash_pipeline\datasets\pointnet_classifier\train_labels.npy') \
        or not os.path.exists(r'dash_pipeline\datasets\pointnet_classifier\test_points.npy') \
        or not os.path.exists(r'dash_pipeline\datasets\pointnet_classifier\test_labels.npy') \
        or not os.path.exists(r'dash_pipeline\datasets\pointnet_classifier\class_map.pkl'):
            train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(dataset_directory=dataset_directory,
                                                                                            num_points=NUM_POINTS)
            
            save(r'dash_pipeline\datasets\pointnet_classifier\train_points.npy', train_points)
            save(r'dash_pipeline\datasets\pointnet_classifier\test_points.npy', test_points)
            save(r'dash_pipeline\datasets\pointnet_classifier\train_labels.npy', train_labels)
            save(r'dash_pipeline\datasets\pointnet_classifier\test_labels.npy', test_labels)
            with open(r'dash_pipeline\datasets\pointnet_classifier\class_map.pkl', 'wb') as f:
                pickle.dump(CLASS_MAP, f)
        else:
            train_points = load(r'dash_pipeline\datasets\pointnet_classifier\train_points.npy')
            test_points = load(r'dash_pipeline\datasets\pointnet_classifier\test_points.npy')
            train_labels = load(r'dash_pipeline\datasets\pointnet_classifier\train_labels.npy')
            test_labels = load(r'dash_pipeline\datasets\pointnet_classifier\test_labels.npy')
            with open(r'dash_pipeline\datasets\pointnet_classifier\class_map.pkl', 'rb') as f:
                CLASS_MAP = pickle.load(f)

        # Data Augmentation to convert the numpy arrays of train and test dataset into tensorfloe based dataset format
        train_dataset, test_dataset = augment_dataset(train_points=train_points, 
                                                    test_points=test_points,
                                                    train_labels=train_labels,
                                                    test_labels=test_labels,
                                                    batch_size=BATCH_SIZE)

        # Generate the Point Net Model Architecture
        pointnet_model = generate_pointnet_model(num_points=NUM_POINTS,
                                                num_classes=NUM_CLASSES)
        
        # Train the Point Net Model 
        trained_pointnet_model = train_pointnet_classifier(model=pointnet_model, train_dataset=train_dataset, test_dataset=test_dataset)

        # Save Trained Point Net Model Weights
        save_model_weights(model=trained_pointnet_model, model_name = 'pointnet_classifier_10cls.h5', path_to_save=r'modelling_pipeline\models')