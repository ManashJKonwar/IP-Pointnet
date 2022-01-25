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
Start by implementing the basic blocks i.e., the convolutional block and 
the multi-layer perceptron block.
"""
def conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = layers.Conv1D(filters, kernel_size=1, padding="valid", name=f"{name}_conv")(x)
    x = layers.BatchNormalization(momentum=0.0, name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)

def mlp_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = layers.Dense(filters, name=f"{name}_dense")(x)
    x = layers.BatchNormalization(momentum=0.0, name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)

"""
Implement a regularizer (taken from this example) to enforce orthogonality 
in the feature space. This is needed to ensure that the magnitudes of the transformed 
features do not vary too much.
"""
class OrthogonalRegularizer(keras.regularizers.Regularizer):
    """Reference: https://keras.io/examples/vision/pointnet/#build-a-model"""

    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.identity = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.identity))

    def get_config(self):
        config = super(TransformerEncoder, self).get_config()
        config.update({"num_features": self.num_features, "l2reg_strength": self.l2reg})
        return config

def transformation_net(inputs: tf.Tensor, num_features: int, name: str) -> tf.Tensor:
    """
    Reference: https://keras.io/examples/vision/pointnet/#build-a-model.

    The `filters` values come from the original paper:
    https://arxiv.org/abs/1612.00593.
    """
    x = conv_block(inputs, filters=64, name=f"{name}_1")
    x = conv_block(x, filters=128, name=f"{name}_2")
    x = conv_block(x, filters=1024, name=f"{name}_3")
    x = layers.GlobalMaxPooling1D()(x)
    x = mlp_block(x, filters=512, name=f"{name}_1_1")
    x = mlp_block(x, filters=256, name=f"{name}_2_1")
    return layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=keras.initializers.Constant(np.eye(num_features).flatten()),
        activity_regularizer=OrthogonalRegularizer(num_features),
        name=f"{name}_final",
    )(x)


def transformation_block(inputs: tf.Tensor, num_features: int, name: str) -> tf.Tensor:
    transformed_features = transformation_net(inputs, num_features, name=name)
    transformed_features = layers.Reshape((num_features, num_features))(
        transformed_features
    )
    return layers.Dot(axes=(2, 1), name=f"{name}_mm")([inputs, transformed_features])

def get_shape_segmentation_model(num_points: int, num_classes: int) -> keras.Model:
    input_points = keras.Input(shape=(None, 3))

    # PointNet Classification Network.
    transformed_inputs = transformation_block(
        input_points, num_features=3, name="input_transformation_block"
    )
    features_64 = conv_block(transformed_inputs, filters=64, name="features_64")
    features_128_1 = conv_block(features_64, filters=128, name="features_128_1")
    features_128_2 = conv_block(features_128_1, filters=128, name="features_128_2")
    transformed_features = transformation_block(
        features_128_2, num_features=128, name="transformed_features"
    )
    features_512 = conv_block(transformed_features, filters=512, name="features_512")
    features_2048 = conv_block(features_512, filters=2048, name="pre_maxpool_block")
    global_features = layers.MaxPool1D(pool_size=num_points, name="global_features")(
        features_2048
    )
    global_features = tf.tile(global_features, [1, num_points, 1])

    # Segmentation head.
    segmentation_input = layers.Concatenate(name="segmentation_input")(
        [
            features_64,
            features_128_1,
            features_128_2,
            transformed_features,
            features_512,
            global_features,
        ]
    )
    segmentation_features = conv_block(
        segmentation_input, filters=128, name="segmentation_features"
    )
    outputs = layers.Conv1D(
        num_classes, kernel_size=1, activation="softmax", name="segmentation_head"
    )(segmentation_features)
    return keras.Model(input_points, outputs)

def generate_pointnet_segmentation_model(num_points=1024, num_classes=5):
    try:
        segmentation_model = get_shape_segmentation_model(num_points, num_classes)
        segmentation_model.summary()
        return segmentation_model
    except Exception as ex:
        print('Caught Exception while Generating Pointnet Segmentation Model Architecture: %s' %(str(ex)))

"""
For the training the authors of pointnet segmnetation recommend using a learning rate schedule 
that decays the initial learning rate by half every 20 epochs. For this experiment, we use to 
change it every 15 epochs.
"""
def generate_pointnet_segmentation_lr_schedule(**kwargs):
    total_training_examples=kwargs.get('total_training_examples')
    BATCH_SIZE=kwargs.get('BATCH_SIZE')
    EPOCHS=kwargs.get('EPOCHS')
    INITIAL_LR=kwargs.get('INITIAL_LR')

    training_step_size = total_training_examples // BATCH_SIZE
    total_training_steps = training_step_size * EPOCHS
    print(f"Total training steps: {total_training_steps}.")

    lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[training_step_size * 15, training_step_size * 15],
        values=[INITIAL_LR, INITIAL_LR * 0.5, INITIAL_LR * 0.25],
    )
    return lr_schedule

def train_pointnet_segmenter(model=None, train_dataset=None, test_dataset=None, model_history_logger=None, lr_schedule=None, epochs=None):
    try:
        if model is not None and train_dataset is not None and test_dataset is not None and model_history_logger is not None \
        and lr_schedule is not None and epochs is not None:

            model.compile(
                loss=keras.losses.CategoricalCrossentropy(),
                optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                metrics=["accuracy"],
            )

            checkpoint_filepath = r"modelling_pipeline\models\checkpoint"
            checkpoint_callback = keras.callbacks.ModelCheckpoint(
                                                                checkpoint_filepath,
                                                                monitor="val_loss",
                                                                save_best_only=True,
                                                                save_weights_only=True,
                                                            )
                
            model.fit(train_dataset,
                    epochs=epochs,
                    validation_data=test_dataset,
                    callbacks=[model_history_logger, checkpoint_callback])

            model.load_weights(checkpoint_filepath)
            return model
    except Exception as ex:
        print('Caught Exception while training the Segmentation Pointnet Model: %s' %(str(ex)))