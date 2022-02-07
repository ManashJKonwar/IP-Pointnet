__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
Each convolution and fully-connected layer (with exception for end layers) 
consits of Convolution / Dense -> Batch Normalization -> ReLU Activation.
"""
def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

"""
PointNet consists of two core components. The primary MLP network, and the 
transformer net (T-net). The T-net aims to learn an affine transformation 
matrix by its own mini network. The T-net is used twice. The first time to 
transform the input features (n, 3) into a canonical representation. 
The second is an affine transformation for alignment in feature space (n, 3). 
As per the original paper we constrain the transformation to be close to an 
orthogonal matrix (i.e. ||X*X^T - I|| = 0).
"""
class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

def tnet(inputs, num_features):
    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

"""
The main network can be then implemented in the same manner where the t-net mini 
models can be dropped in a layers in the graph. Here we replicate the network architecture 
published in the original paper but with half the number of weights at each layer as we 
are using the smaller 10 class ModelNet dataset.
"""
def generate_pointnet_model(num_points=2048, num_classes=10):
    try:
        inputs = keras.Input(shape=(num_points, 3))

        x = tnet(inputs, 3)
        x = conv_bn(x, 32)
        x = conv_bn(x, 32)
        x = tnet(x, 32)
        x = conv_bn(x, 32)
        x = conv_bn(x, 64)
        x = conv_bn(x, 512)
        x = layers.GlobalMaxPooling1D()(x)
        x = dense_bn(x, 256)
        x = layers.Dropout(0.3)(x)
        x = dense_bn(x, 128)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(num_classes, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
        model.summary()
        return model
    except Exception as ex:
        print('Caught Exception while Generating Pointnet Model Architecture: %s' %(str(ex)))

def train_pointnet_classifier(**kwargs):
    # DL Model to train
    model=kwargs.get('model') if 'model' in kwargs.keys() else None
    
    # Training and Validation Dataset
    train_dataset=kwargs.get('train_dataset') if 'train_dataset' in kwargs.keys() else None
    validation_dataset=kwargs.get('validation_dataset') if 'validation_dataset' in kwargs.keys() else None
    
    # Model Training Callbacks
    callbacks=kwargs.get('callbacks') if 'callbacks' in kwargs.keys() else {}
    
    # Training Parameters
    training_parameters=kwargs.get('training_parameters') if 'training_parameters' in kwargs.keys() else \
                        {
                            'learning_rate':0.001,
                            'epochs':20 
                        }
    try:
        if model is not None and train_dataset is not None and validation_dataset is not None:
            callback_list = [callback_package['callback_item'] for callback_package in callbacks.values() if callback_package['callback_flag']==True]
            
            model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=keras.optimizers.Adam(learning_rate=training_parameters['learning_rate']),
                metrics=["sparse_categorical_accuracy"],
            )
            model.fit(train_dataset, epochs=training_parameters['epochs'], callbacks=callback_list, validation_data=validation_dataset)
            return model
    except Exception as ex:
        print('Caught Exception while training the Pointnet Model: %s' %(str(ex)))