__author__ = "konwar.m"
__copyright__ = "Copyright 2021, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import os
import trimesh
import tensorflow as tf

def download_dataset(dataset_url=None, dataset_directory=None):
    """
    This method downloads the point cloud dataset from relevant url provided and 
    dumps this in the dataset folder
    Parameters: 
        dataset_url (str): http link for downloading point cloud dataset
    Returns: 
        None
    """
    try:
        if dataset_url is not None:
            if os.path.exists(os.path.join(dataset_directory, 'datasets', os.path.basename(dataset_url).split('.')[0])):
                return os.path.join(dataset_directory, 'datasets', os.path.basename(dataset_url).split('.')[0])
            
            DATA_DIR = tf.keras.utils.get_file(
                "modelnet.zip",
                dataset_url,
                extract=True,
                cache_dir=dataset_directory
            )
            DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), os.path.basename(dataset_url).split('.')[0])
            return DATA_DIR
    except Exception:
        print('Caught Exception while downloading dataset', exc_info=True)

def visualize_dataset(dataset_directory=None, mesh_filename_path=None):
    """
    This method loads the mesh file from dataset directory and helps us to visualize it
    Parameters: 
        dataset_directory (str): root dataset directory
        mesh_filename_path (str): mesh file name to process
    Returns: 
        
    """
    try:
        if dataset_directory is not None and mesh_filename_path is not None:
            mesh = trimesh.load(os.path.join(dataset_directory, mesh_filename_path))
            return mesh
    except Exception:
        print('Caught Exception while reading meshfile', exc_info=True)

def save_model_weights(model=None, model_name=None, path_to_save=None):
    if model is not None and model_name is not None and path_to_save is not None:
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        model.save_weights(filepath=os.path.join(path_to_save, model_name), overwrite=True)

def generate_history_callback(history_file_name=None, history_path_to_save=None):
    if history_file_name is not None and history_path_to_save is not None:
        history_logger = tf.keras.callbacks.CSVLogger(os.path.join(history_path_to_save, history_file_name), separator=",", append=True)
        return history_logger