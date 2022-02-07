__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import dash
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output, State
from dash_pipeline.calback_manager import CallbackManager
from dash_pipeline.backend import *
from dash_pipeline.utility.utility_app import find_index
from dash_pipeline.backend import trained_classifier_dict, class_map

callback_manager = CallbackManager()

"""
Callback for Learn More Button for classifier
"""
@callback_manager.callback(Output(component_id='modal', component_property='is_open'),
                        [Input(component_id='learn-more-button', component_property='n_clicks'), 
                        Input(component_id='classifier-modal-close', component_property='n_clicks')],
                        State(component_id='modal', component_property='is_open'))
def update_click_output(button_click, close_click, is_open):
    ctx = dash.callback_context
    prop_id = ""
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if prop_id == 'learn-more-button':
        return True
    elif prop_id == 'classifier-modal-close':
        return False
    else:
        return is_open

"""
Callback for plotting training and validation 
data during classifier training
"""
@callback_manager.callback(Output(component_id='model-training-history', component_property='figure'),
                        Input(component_id='dropdown-model-selection', component_property='value'))
def update_training_characteristics(selected_model):
    if selected_model is not None or selected_model!='':
        trained_classifier_history = trained_classifier_dict[selected_model+'.h5']['history']
        
        fig = make_subplots(rows=2, cols=1)

        final_loss_cols = [col for col in list(trained_classifier_history.columns) if 'loss' in col]
        final_acc_cols = [col for col in list(trained_classifier_history.columns) if 'accuracy' in col]

        for loss_col in final_loss_cols:
            fig.append_trace(go.Scatter(x=trained_classifier_history.epoch,
                                        y=trained_classifier_history[loss_col],
                                        name=loss_col)
                                        , row=1, col=1)
        for acc_col in final_acc_cols:
            fig.append_trace(go.Scatter(x=trained_classifier_history.epoch,
                                        y=trained_classifier_history[acc_col],
                                        name=acc_col)
                                        , row=2, col=1)

        fig.update_traces(mode='lines+markers')
        return fig
    else:
        return dash.no_update

"""
Callback for labelling prediction and validation data 
based on trained classifier model
"""
@callback_manager.callback(Output(component_id='div-detection-mode', component_property='children'),
                        [Input(component_id='dropdown-image-selection', component_property='value'),
                        Input(component_id='dropdown-model-selection', component_property='value')],
                        State(component_id='dropdown-class-selection', component_property='value'))
def update_detection_mode(image_value, selected_model, class_value):
    class_index = list(class_map.keys())[list(class_map.values()).index(class_value)]
    start_index, end_index = find_index(test_labels, len(test_labels), class_index)

    if start_index != None and end_index != None:
        trained_classifier_model=trained_classifier_dict[selected_model+'.h5']['model']

        total_images = end_index-start_index+1
        image_options = [class_value+'-'+str(image_no) for image_no in range(total_images)]
        image_index = image_options.index(image_value) + start_index

        test_image_array = test_points[image_index]

        # Create Prediction Data for finding pointnet classifier prediction
        test_dataset = tf.data.Dataset.from_tensor_slices((test_points, np.array([class_index]*len(test_points))))
        test_dataset = test_dataset.shuffle(len(test_points)).batch(32)
        
        tensor_points, tensor_labels = list(test_dataset.take(1))[0]

        preds = trained_classifier_model.predict(tensor_points)
        preds = tf.math.argmax(preds, -1)
        predicted_class = class_map[preds[0].numpy()]

    else:
        return dash.no_update

    return [
        html.Div(
            children=[
                dbc.Row("Classifier Prediction: %s" %(predicted_class)),
                dbc.Row("Actual Label: %s" %(class_map[tensor_labels.numpy()[0]]))
                # dcc.Graph(id="bar-score-graph"),
            ]
        )
    ]

"""
Callback for populating dropdown for image labels
in specific class under classifier model
"""
@callback_manager.callback([Output(component_id='dropdown-image-selection', component_property='options'),
                        Output(component_id='dropdown-image-selection', component_property='value')],
                        Input(component_id='dropdown-class-selection', component_property='value'))
def update_image_selection_options(class_value):
    class_index = list(class_map.keys())[list(class_map.values()).index(class_value)]
    start_index, end_index = find_index(test_labels, len(test_labels), class_index)

    if start_index != None and end_index != None:
        total_images = end_index-start_index+1
        image_options = [class_value+'-'+str(image_no) for image_no in range(total_images)]
        return [{'label': image_value.capitalize(), 'value': image_value} for image_value in image_options], image_options[0]
    else:
        return dash.no_update, dash.no_update

"""
Callback for plotting prediction and validation data 
based on trained classifier
"""
@callback_manager.callback(Output(component_id='model-output', component_property='figure'),
                        Input(component_id='dropdown-image-selection', component_property='value'),
                        State(component_id='dropdown-class-selection', component_property='value'))
def update_image(image_value, class_value):
    class_index = list(class_map.keys())[list(class_map.values()).index(class_value)]
    start_index, end_index = find_index(test_labels, len(test_labels), class_index)

    if start_index != None and end_index != None:
        total_images = end_index-start_index+1
        image_options = [class_value+'-'+str(image_no) for image_no in range(total_images)]
        image_index = image_options.index(image_value) + start_index

        test_image_array = test_points[image_index]
        df_test_array = pd.DataFrame()
        for arr_points in test_image_array:
            df_test_array = df_test_array.append(
                                                {'x': round(arr_points[0], 2), 
                                                'y': round(arr_points[1], 2),
                                                'z': round(arr_points[2], 2), 
                                                'image-label': image_value,
                                                'marker_size': 5.0}
                                            ,ignore_index=True)
        
        fig = px.scatter_3d(df_test_array, x='x', y='y', z='z', size='marker_size', size_max=5)
        return fig
    else:
        return dash.no_update