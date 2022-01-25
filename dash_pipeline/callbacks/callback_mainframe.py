__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import dash_bootstrap_components as dbc 
from dash import html
from dash.dependencies import Input, Output
from dash_pipeline.calback_manager import CallbackManager
from dash_pipeline.layouts import layout_mainframe, layout_pointnet_classifier   

callback_manager = CallbackManager()

# set the content according to the current pathname
@callback_manager.callback(Output(component_id='page-content', component_property='children'), 
                        Input(component_id='url', component_property='pathname'))
def render_page_content(pathname):
    if pathname == "/":
        return layout_pointnet_classifier.layout
    elif pathname == "/pointnet-classification":
        return layout_pointnet_classifier.layout
    elif pathname == "/pointnet-part-segmentation": 
        return html.P("This is pointnet part segmentation page!")
    elif pathname == "/pointnet-semantic-segmentation":
        return html.P("This is pointnet semantic segmentation page!")
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )