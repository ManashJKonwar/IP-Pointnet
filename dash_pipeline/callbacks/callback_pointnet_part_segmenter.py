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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output, State
from dash_pipeline.calback_manager import CallbackManager
from dash_pipeline.backend import trained_part_segmenter_model, trained_part_segmenter_history, pps_validation_dataset, LABELS

callback_manager = CallbackManager()

"""
Callback for Learn More Button for part segmenter
"""
@callback_manager.callback(Output(component_id='pps-modal', component_property='is_open'),
                        [Input(component_id='pps-learn-more-button', component_property='n_clicks'), 
                        Input(component_id='pps-modal-close', component_property='n_clicks')],
                        State(component_id='pps-modal', component_property='is_open'))
def update_click_output(button_click, close_click, is_open):
    ctx = dash.callback_context
    prop_id = ""
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if prop_id == 'learn-more-button':
        return True
    elif prop_id == 'pps-modal-close':
        return False
    else:
        return is_open

"""
Callback for plotting training and validation 
data during part segmenter training
"""
@callback_manager.callback(Output(component_id='pps-model-training-history', component_property='figure'),
                        Input(component_id='pps-dropdown-model-selection', component_property='value'))
def update_training_characteristics(selected_model):
    if selected_model.__eq__('pointnet_airplane_ps_60epochs'):
        fig = make_subplots(rows=2, cols=1)

        final_loss_cols = [col for col in list(trained_part_segmenter_history.columns) if 'loss' in col]
        final_acc_cols = [col for col in list(trained_part_segmenter_history.columns) if 'accuracy' in col]

        for loss_col in final_loss_cols:
            fig.append_trace(go.Scatter(x=trained_part_segmenter_history.epoch,
                                        y=trained_part_segmenter_history[loss_col],
                                        name=loss_col)
                                        , row=1, col=1)
        for acc_col in final_acc_cols:
            fig.append_trace(go.Scatter(x=trained_part_segmenter_history.epoch,
                                        y=trained_part_segmenter_history[acc_col],
                                        name=acc_col)
                                        , row=2, col=1)

        fig.update_traces(mode='lines+markers')
        return fig
    else:
        return dash.no_update

"""
Callback for populating dropdown for image labels
in specific class under classifier model
"""
@callback_manager.callback([Output(component_id='pps-dropdown-image-selection', component_property='options'),
                        Output(component_id='pps-dropdown-image-selection', component_property='value')],
                        Input(component_id='pps-dropdown-class-selection', component_property='value'))
def update_image_selection_options(object_category):
    image_options=[]
    
    if object_category is not None:
        for index in range(len(pps_validation_dataset)):
            image_options.append(object_category+'-'+str(index))
        
        return [{'label': image_value.capitalize(), 'value': image_value} for image_value in image_options], image_options[0]
    else:
        return dash.no_update, dash.no_update

"""
Callback for plotting prediction and validation data 
based on trained part segmenter
"""
@callback_manager.callback([Output(component_id='pps-model-output-groundtruth', component_property='figure'),
                        Output(component_id='pps-model-output-prediction', component_property='figure')],
                        Input(component_id='pps-dropdown-image-selection', component_property='value'),
                        State(component_id='pps-dropdown-class-selection', component_property='value'))
def update_part_segmenter_results(image_value, object_category):
    
    if object_category is not None and image_value is not None:
        extracted_image_index = int(image_value.split('-')[1])

        validation_batch = next(iter(pps_validation_dataset))
        val_predictions = trained_part_segmenter_model.predict(validation_batch[0])

        idx = np.random.choice(len(validation_batch[0]))
        print(f"Index selected: {idx}")

        df_ground_truth = pd.DataFrame()
        label_map = LABELS + ["none"]
        point_clouds = validation_batch[0]
        label_clouds = validation_batch[1]

        def visualize_single_point_cloud(point_clouds, label_clouds, idx):
            label_map = LABELS + ["none"]
            point_cloud = point_clouds[idx]
            label_cloud = label_clouds[idx]

            df_point = pd.DataFrame(
                            data={
                                "x": point_cloud[:, 0],
                                "y": point_cloud[:, 1],
                                "z": point_cloud[:, 2],
                                "label": [label_map[np.argmax(label)] for label in label_cloud],
                                'marker_size': 3.0
                            }
                        )

            return px.scatter_3d(df_point, x='x', y='y', z='z', color='label', size='marker_size', size_max=5)

        # Plotting with ground-truth.
        ground_truth_fig = visualize_single_point_cloud(validation_batch[0], validation_batch[1], idx)

        # Plotting with predicted labels.
        prediction_fig = visualize_single_point_cloud(validation_batch[0], val_predictions, idx)
        
        return ground_truth_fig, prediction_fig
    else:
        return dash.no_update