__author__ = "konwar.m"
__copyright__ = "Copyright 2021, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

from modelling_pipeline.preprocessing.data_preprocesser import parse_dataset
from modelling_pipeline.utility.utility_datatransformation import download_dataset

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

        train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(dataset_directory=dataset_directory,
                                                                                        num_points=NUM_POINTS)