__author__ = "konwar.m"
__copyright__ = "Copyright 2021, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash_pipeline.calback_manager import CallbackManager

callback_manager = CallbackManager()

@callback_manager.callback(Output(component_id='div-detection-mode', component_property='children'),
                        Input(component_id='dropdown-graph-view-mode', component_property='value'))
def update_detection_mode(value):
    if value == "detection":
        return [
            html.Div(
                children=[
                    html.P(
                        children="Detection Score of Most Probable Objects",
                        className="plot-title",
                    ),
                    dcc.Graph(id="bar-score-graph"),
                ]
            )
        ]
    return []

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