__author__ = "konwar.m"
__copyright__ = "Copyright 2021, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import dash
import logging, logging.config
import dash_bootstrap_components as dbc
from flask import Flask
from dash_pipeline.callbacks.callback_mainframe import callback_manager as mainframe_callback_manager
from dash_pipeline.callbacks.callback_pointnet_classifier import callback_manager as classifier_callback_manager

FA = "https://use.fontawesome.com/releases/v5.15.1/css/all.css"
external_stylesheets = [dbc.themes.BOOTSTRAP, FA]
server = Flask(__name__) 
app = dash.Dash(external_stylesheets=external_stylesheets, server=server)
app.config.suppress_callback_exceptions = True

mainframe_callback_manager.attach_to_app(app)
classifier_callback_manager.attach_to_app(app)