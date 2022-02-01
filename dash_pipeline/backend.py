__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import os 
import json
import pickle
import tensorflow as tf
from numpy import save, load
from dash_pipeline.utility.utility_pointnet_classifier import load_classifier_dataset, load_classifier_model, load_classifier_training_history
from dash_pipeline.utility.utility_pointnet_part_segmenter import load_segmentation_dataset, load_part_segmenter_model, load_part_segmenter_training_history

#region Pointnet Classifier
NUM_POINTS = 2048
NUM_CLASSES = 10
CLASSIFIER_DATA_DIR = r'modelling_pipeline\datasets\ModelNet10'

test_points, test_labels, class_map = None, None, None

if not os.path.exists(r'dash_pipeline\datasets\pointnet_classifier\test_points.npy') \
or not os.path.exists(r'dash_pipeline\datasets\pointnet_classifier\test_labels.npy') \
or not os.path.exists(r'dash_pipeline\datasets\pointnet_classifier\class_map.pkl'):
    test_points, test_labels, class_map = load_classifier_dataset(dataset_directory=CLASSIFIER_DATA_DIR, 
                                                                num_points=NUM_POINTS)
    
    save(r'dash_pipeline\datasets\pointnet_classifier\test_points.npy', test_points)
    save(r'dash_pipeline\datasets\pointnet_classifier\test_labels.npy', test_labels)
    with open(r'dash_pipeline\datasets\pointnet_classifier\class_map.pkl', 'wb') as f:
        pickle.dump(class_map, f)
else:
    test_points = load(r'dash_pipeline\datasets\pointnet_classifier\test_points.npy')
    test_labels = load(r'dash_pipeline\datasets\pointnet_classifier\test_labels.npy')
    with open(r'dash_pipeline\datasets\pointnet_classifier\class_map.pkl', 'rb') as f:
        class_map = pickle.load(f)

trained_classifier_model = None
trained_classifier_model = load_classifier_model(model_wts_filename=r'modelling_pipeline\models\pointnet_classifier_10cls.h5', 
                                                num_points=NUM_POINTS,
                                                num_classes=NUM_CLASSES)

trained_classifier_history = None
trained_classifier_history = load_classifier_training_history(model_history_filename=r'modelling_pipeline\models\pointnet_classifier_10cls_history.csv')
#endregion

#region Pointnet Part Segmenter
VAL_SPLIT = 0.2
NUM_SAMPLE_POINTS = 1024
BATCH_SIZE = 32
EPOCHS = 60
INITIAL_LR = 1e-3
PART_SEGMENTER_DATA_DIR = r'modelling_pipeline\datasets\PartAnnotation'
object_name='Airplane'
PART_SEGMENTER_DICT={}

with open(os.path.join(PART_SEGMENTER_DATA_DIR,'metadata.json')) as json_file:
    metadata = json.load(json_file)
# Data Reading which focus to specific object
points_dir = os.path.join(PART_SEGMENTER_DATA_DIR, metadata[object_name]["directory"], 'points')
labels_dir = os.path.join(PART_SEGMENTER_DATA_DIR, metadata[object_name]["directory"], 'points_label')
LABELS = metadata[object_name]["lables"]
COLORS = metadata[object_name]["colors"]

pps_dataset_path = r'dash_pipeline\datasets\pointnet_part_segmenter\%s' %(object_name.lower())

if not os.path.exists(os.path.join(pps_dataset_path, 'element_spec')):
    if not os.path.exists(os.path.join('dash_pipeline\datasets\pointnet_part_segmenter',object_name.lower())):
        os.makedirs(os.path.join('dash_pipeline\datasets\pointnet_part_segmenter',object_name.lower()))

    pps_validation_dataset = load_segmentation_dataset(dataset_directory=PART_SEGMENTER_DATA_DIR,
                                                points_dir=points_dir,
                                                labels_dir=labels_dir,
                                                VAL_SPLIT=VAL_SPLIT,
                                                NUM_SAMPLE_POINTS=NUM_SAMPLE_POINTS,
                                                BATCH_SIZE=BATCH_SIZE,
                                                EPOCHS=EPOCHS,
                                                INITIAL_LR=INITIAL_LR,
                                                LABELS=LABELS,
                                                COLORS=COLORS)
    tf.data.experimental.save(
        pps_validation_dataset, 
        pps_dataset_path, 
        compression='GZIP'
    )
    with open(os.path.join(pps_dataset_path, 'element_spec'), 'wb') as out_:  # also save the element_spec to disk for future loading
        pickle.dump(pps_validation_dataset.element_spec, out_)
else:
    with open(os.path.join(pps_dataset_path, 'element_spec'), 'rb') as in_:
        es = pickle.load(in_)

    pps_validation_dataset = tf.data.experimental.load(
                            pps_dataset_path, 
                            es, 
                            compression='GZIP'
                        )

trained_part_segmenter_model = None
trained_part_segmenter_model = load_part_segmenter_model(model_wts_filename=r'modelling_pipeline\models\pointnet_part_segmenter_airplane.h5')

trained_part_segmenter_history = None
trained_part_segmenter_history = load_part_segmenter_training_history(model_history_filename=r'modelling_pipeline\models\pointnet_part_segmenter_history.csv')
#endregion