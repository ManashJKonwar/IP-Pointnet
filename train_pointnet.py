__author__ = "konwar.m"
__copyright__ = "Copyright 2021, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import os, json
import pickle, random
from numpy import load, save
from modelling_pipeline.preprocessing.data_preprocesser import augment_dataset, parse_dataset, parse_segmentation_dataset, preprocess_segmentation_dataset, \
                                                                generate_segmentation_dataset
from modelling_pipeline.modelling.train_pointnet_classifier import generate_pointnet_model, train_pointnet_classifier
from modelling_pipeline.modelling.train_pointnet_part_segmenter import generate_pointnet_segmentation_model, generate_pointnet_segmentation_lr_schedule, train_pointnet_segmenter
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
        if not os.path.exists(r'modelling_pipeline\datasets\preprocessed\pointnet_classifier\train_points.npy') \
        or not os.path.exists(r'modelling_pipeline\datasets\preprocessed\pointnet_classifier\train_labels.npy') \
        or not os.path.exists(r'modelling_pipeline\datasets\preprocessed\pointnet_classifier\test_points.npy') \
        or not os.path.exists(r'modelling_pipeline\datasets\preprocessed\pointnet_classifier\test_labels.npy') \
        or not os.path.exists(r'modelling_pipeline\datasets\preprocessed\pointnet_classifier\class_map.pkl'):
            train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(dataset_directory=classifier_dataset_directory,
                                                                                            num_points=NUM_POINTS)
            
            save(r'modelling_pipeline\datasets\preprocessed\pointnet_classifier\train_points.npy', train_points)
            save(r'modelling_pipeline\datasets\preprocessed\pointnet_classifier\test_points.npy', test_points)
            save(r'modelling_pipeline\datasets\preprocessed\pointnet_classifier\train_labels.npy', train_labels)
            save(r'modelling_pipeline\datasets\preprocessed\pointnet_classifier\test_labels.npy', test_labels)
            with open(r'modelling_pipeline\datasets\preprocessed\pointnet_classifier\class_map.pkl', 'wb') as f:
                pickle.dump(CLASS_MAP, f)
        else:
            train_points = load(r'modelling_pipeline\datasets\preprocessed\pointnet_classifier\train_points.npy')
            test_points = load(r'modelling_pipeline\datasets\preprocessed\pointnet_classifier\test_points.npy')
            train_labels = load(r'modelling_pipeline\datasets\preprocessed\pointnet_classifier\train_labels.npy')
            test_labels = load(r'modelling_pipeline\datasets\preprocessed\pointnet_classifier\test_labels.npy')
            with open(r'modelling_pipeline\datasets\preprocessed\pointnet_classifier\class_map.pkl', 'rb') as f:
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
        VAL_SPLIT = 0.2
        NUM_SAMPLE_POINTS = 1024
        BATCH_SIZE = 32
        EPOCHS = 60
        INITIAL_LR = 1e-3

        with open(os.path.join(part_segmenter_dataset_directory,'metadata.json')) as json_file:
            metadata = json.load(json_file)

        object_name = "Airplane"
        # Data Reading which focus to specific object
        points_dir = os.path.join(part_segmenter_dataset_directory, metadata[object_name]["directory"], 'points')
        labels_dir = os.path.join(part_segmenter_dataset_directory, metadata[object_name]["directory"], 'points_label')
        LABELS = metadata[object_name]["lables"]
        COLORS = metadata[object_name]["colors"]

        # Stucturing Dataset for this paprt segmnetation dataset by reading relevant point cloud data and 
        # point cloud labels
        if not os.path.exists(r'modelling_pipeline\datasets\preprocessed\pointnet_part_segmenter\%s\point_clouds.npy' %(object_name.lower())) \
        or not os.path.exists(r'modelling_pipeline\datasets\preprocessed\pointnet_part_segmenter\%s\point_cloud_labels.npy' %(object_name.lower())) \
        or not os.path.exists(r'modelling_pipeline\datasets\preprocessed\pointnet_part_segmenter\%s\test_point_clouds.npy' %(object_name.lower())) \
        or not os.path.exists(r'modelling_pipeline\datasets\preprocessed\pointnet_part_segmenter\%s\all_labels.npy' %(object_name.lower())):
            if not os.path.exists(os.path.join('modelling_pipeline\datasets\preprocessed\pointnet_part_segmenter',object_name.lower())):
                os.makedirs(os.path.join('modelling_pipeline\datasets\preprocessed\pointnet_part_segmenter',object_name.lower()))

            point_clouds, point_cloud_labels, test_point_clouds, all_labels = parse_segmentation_dataset(dataset_directory=part_segmenter_dataset_directory,
                                                                                                        points_dir=points_dir,
                                                                                                        labels_dir=labels_dir,
                                                                                                        VAL_SPLIT=VAL_SPLIT,
                                                                                                        NUM_SAMPLE_POINTS=NUM_SAMPLE_POINTS,
                                                                                                        BATCH_SIZE=BATCH_SIZE,
                                                                                                        EPOCHS=EPOCHS,
                                                                                                        INITIAL_LR=INITIAL_LR,
                                                                                                        LABELS=LABELS,
                                                                                                        COLORS=COLORS)

            save(r'modelling_pipeline\datasets\preprocessed\pointnet_part_segmenter\%s\point_clouds.npy' %(object_name.lower()) , point_clouds)
            save(r'modelling_pipeline\datasets\preprocessed\pointnet_part_segmenter\%s\point_cloud_labels.npy' %(object_name.lower()), point_cloud_labels)
            save(r'modelling_pipeline\datasets\preprocessed\pointnet_part_segmenter\%s\test_point_clouds.npy' %(object_name.lower()), test_point_clouds)
            save(r'modelling_pipeline\datasets\preprocessed\pointnet_part_segmenter\%s\all_labels.npy' %(object_name.lower()), all_labels)
        else:
            point_clouds = load(r'modelling_pipeline\datasets\preprocessed\pointnet_part_segmenter\%s\point_clouds.npy' %(object_name.lower()), allow_pickle=True)
            point_cloud_labels = load(r'modelling_pipeline\datasets\preprocessed\pointnet_part_segmenter\%s\point_cloud_labels.npy' %(object_name.lower()), allow_pickle=True)
            test_point_clouds = load(r'modelling_pipeline\datasets\preprocessed\pointnet_part_segmenter\%s\test_point_clouds.npy' %(object_name.lower()), allow_pickle=True)
            all_labels = load(r'modelling_pipeline\datasets\preprocessed\pointnet_part_segmenter\%s\all_labels.npy' %(object_name.lower()), allow_pickle=True)
        
        # Preprocess the point cloud to perform random batching and also performing normalization on top of that
        point_clouds, point_cloud_labels, all_labels = preprocess_segmentation_dataset(point_clouds=point_clouds,
                                                                                    point_cloud_labels=point_cloud_labels,
                                                                                    all_labels=all_labels,
                                                                                    NUM_SAMPLE_POINTS=NUM_SAMPLE_POINTS)
        
        # Data Augmentation to convert the numpy arrays of train and test dataset into tensorflow based dataset format
        train_dataset, validation_dataset, total_training_examples = generate_segmentation_dataset(point_clouds=point_clouds,
                                                                                                point_cloud_labels=point_cloud_labels,
                                                                                                VAL_SPLIT=VAL_SPLIT,
                                                                                                BATCH_SIZE=BATCH_SIZE)
        
        x, y = next(iter(train_dataset))
        num_points = x.shape[1]
        num_classes = y.shape[-1]

        # Generate the Segmentation Point Net Model Architecture
        segmentation_pointnet_model = generate_pointnet_segmentation_model(num_points=num_points,
                                                                        num_classes=num_classes)

        # Generate the Segmentation Point Net Model History Callback Logger
        segmentation_pointnet_history_logger = generate_history_callback(history_file_name='pointnet_part_segmenter_history.csv',
                                                                        history_path_to_save=r'modelling_pipeline\models')

        # Learning Rate Schedule
        segmentation_lr_schedule = generate_pointnet_segmentation_lr_schedule(total_training_examples=total_training_examples,
                                                                            BATCH_SIZE=BATCH_SIZE,
                                                                            EPOCHS=EPOCHS,
                                                                            INITIAL_LR=INITIAL_LR)

        # Train the Segmentation Point Net Model 
        trained_segmentation_pointnet_model = train_pointnet_segmenter(model=segmentation_pointnet_model, 
                                                                    train_dataset=train_dataset, 
                                                                    test_dataset=validation_dataset,
                                                                    model_history_logger=segmentation_pointnet_history_logger,
                                                                    lr_schedule=segmentation_lr_schedule,
                                                                    epochs=EPOCHS)
        
        # Save Trained Segmentation Point Net Model Weights
        save_model_weights(model=trained_segmentation_pointnet_model, model_name = 'pointnet_part_segmenter_%s.h5' %(str(object_name.lower())), path_to_save=r'modelling_pipeline\models')