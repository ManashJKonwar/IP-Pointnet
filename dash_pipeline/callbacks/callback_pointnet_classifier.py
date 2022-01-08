__author__ = "konwar.m"
__copyright__ = "Copyright 2021, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

from dash import dcc
from dash import html
from dash.dependencies import Input, Output
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