__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import dash
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output, State
from dash_pipeline.calback_manager import CallbackManager
from dash_pipeline.backend import trained_part_segmenter_model, trained_part_segmenter_history

callback_manager = CallbackManager()

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