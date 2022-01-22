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
from modelling_pipeline.utility.utility_datatransformation import download_dataset, generate_history_callback, save_model_weights 

if __name__ == '__main__':
    TRAIN_POINTNET_CLASSIFIER = False
    TRAIN_POINTNET_PART_SEGMENTATOR = True
    TRAIN_POINTNET_SEMANTIC_SEGMANTATOR = False
    POINTNET_CLASSIFIER_DATASET_URL = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
    POINTNET_CLASSIFIER_DATASET_DIR = r"modelling_pipeline"
    POINTNET_CLASSIFIER_DATASET_NAME = "ModelNet10"
    POINTNET_PART_SEGMENTER_DATASET_URL = "https://git.io/JiY4i"
    POINTNET_PART_SEGMENTER_DATASET_DIR = r"modelling_pipeline"
    POINTNET_PART_SEGMENTER_DATASET_NAME = "PartAnnotation"

    # Downloading Point Cloud Dataset
    classifier_dataset_directory = download_dataset(dataset_url=POINTNET_CLASSIFIER_DATASET_URL, 
                                                    dataset_directory=POINTNET_CLASSIFIER_DATASET_DIR, 
                                                    zipped_name="modelnet.zip",
                                                    extracted_folder_name=POINTNET_CLASSIFIER_DATASET_NAME)
    part_segmenter_dataset_directory = download_dataset(dataset_url=POINTNET_PART_SEGMENTER_DATASET_URL, 
                                                        dataset_directory=POINTNET_PART_SEGMENTER_DATASET_DIR, 
                                                        zipped_name="shapenet.zip",
                                                        extracted_folder_name=POINTNET_PART_SEGMENTER_DATASET_NAME)
    
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
            train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(dataset_directory=classifier_dataset_directory,
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

        # Generate the Point Net Model History Callback Logger
        pointnet_history_logger = generate_history_callback(history_file_name='pointnet_classifier_10cls_history.csv',
                                                            history_path_to_save=r'modelling_pipeline\models')
        
        # Train the Point Net Model 
        trained_pointnet_model = train_pointnet_classifier(model=pointnet_model, 
                                                        train_dataset=train_dataset, 
                                                        test_dataset=test_dataset,
                                                        model_history_logger=pointnet_history_logger)

        # Save Trained Point Net Model Weights
        save_model_weights(model=trained_pointnet_model, model_name = 'pointnet_classifier_10cls.h5', path_to_save=r'modelling_pipeline\models')
    
    if TRAIN_POINTNET_PART_SEGMENTATOR:
        print('here')