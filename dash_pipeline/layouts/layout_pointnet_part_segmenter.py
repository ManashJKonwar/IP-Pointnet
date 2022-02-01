__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from textwrap import dedent
from dash_pipeline.backend import *

def part_segmenter_modal():
    return html.Div([
                dbc.Modal(
                    [
                        dbc.ModalHeader("HEADER"),
                        dbc.ModalBody(
                                html.Div(
                                    children=[
                                        dcc.Markdown(
                                            children=dedent(
                                                """
                                            ##### What am I looking at?
                                            
                                            This app enhances visualization of objects detected using state-of-the-art Pointnet Classification.
                                            Most user generated videos are dynamic and fast-paced, which might be hard to interpret. A confidence
                                            heatmap stays consistent through the video and intuitively displays the model predictions. The pie chart
                                            lets you interpret how the object classes are divided, which is useful when analyzing videos with numerous
                                            and differing objects.
                                            ##### More about this Dash app
                                            
                                            The purpose of this demo is to explore alternative visualization methods for object detection. Therefore,
                                            the visualizations, predictions and videos are not generated in real time, but done beforehand. To read
                                            more about it, please visit the [project repo](https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-object-detection).
                                            """
                                            )
                                        )
                                    ],
                                ),
                                style={"height": "30vh"}
                        ),
                        dbc.ModalFooter(
                            dbc.Button("CLOSE BUTTON", id="pps-modal-close", className="ml-auto")
                        ),
                    ],
                    id="pps-modal", # Give the modal an id name 
                    is_open=False,  # Open the modal at opening the webpage.
                    size="xl",  # "sm", "lg", "xl" = small, large or extra large
                    backdrop=True,  # Modal to not be closed by clicking on backdrop
                    scrollable=True,  # Scrollable in case of large amount of text
                    centered=True,  # Vertically center modal 
                    keyboard=True,  # Close modal when escape is pressed
                    fade=True,  # True, False
                    # style={"max-width": "none", "width": "50%"}
                )
            ])

layout = html.Div(
            children=[
                dbc.Row([
                    dbc.Col([
                        html.Div(
                            id="pps-left-side-column",
                            children=[
                                html.Div(
                                    id="pps-header-section",
                                    children=[
                                        html.H4("Pointnet Part Segmenter Explorer"),
                                        html.P(
                                            "To get started, select the object category and choose the image no after that, you will be able to see " 
                                            "the pointnet part segmenter structure of the same in the graphical area, further the groundtruth and prediction "
                                            "plots could be observed in the right bottom corner of your screen. You can also see the model training characteristics "
                                            "on the right top corner of your screen. Select the appropriate model type after training the pointnet part segmenter "
                                            "model"
                                        ),
                                        html.Button(
                                            "Learn Part Segmenter", id="pps-learn-more-button", n_clicks=0
                                        ),
                                    ],
                                ),
                                html.Div(
                                    children=[
                                        html.Div(children=["Model Selection:"]),
                                        dcc.Dropdown(
                                            id="pps-dropdown-model-selection",
                                            options = [{'label': classifier_item, 'value': classifier_item} for classifier_item in ['pointnet_airplane_ps_60epochs']],
                                            value='pointnet_airplane_ps_60epochs',
                                            clearable=False,
                                        ),
                                    ],
                                ),
                                html.Div(
                                    dcc.Loading(
                                        children=[
                                            dbc.Row([
                                                dbc.Col(
                                                    dcc.Graph(id='pps-model-output-groundtruth', style={"height": "70vh"})
                                                ),
                                                dbc.Col(
                                                    dcc.Graph(id='pps-model-output-prediction', style={"height": "70vh"})
                                                )
                                            ])
                                        ],
                                        type="cube",
                                    )

                                ),
                                html.Div(
                                    children=[
                                        html.Div(
                                            children=[
                                                html.Div(
                                                    children=["Minimum Confidence Threshold:"]
                                                ),
                                                html.Div(
                                                    dcc.Slider(
                                                        id="pps-slider-minimum-confidence-threshold",
                                                        min=20,
                                                        max=90,
                                                        marks={
                                                            i: f"{i}%"
                                                            for i in range(20, 91, 10)
                                                        },
                                                        value=30,
                                                        updatemode="drag",
                                                    )
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            children=[
                                                html.Div(children=["Class Selection:"]),
                                                dcc.Dropdown(
                                                    id="pps-dropdown-class-selection",
                                                    options = [{'label': class_item.capitalize(), 'value': class_item} for class_item in [object_name]],
                                                    value=object_name,
                                                    clearable=False,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            children=[
                                                html.Div(children=["Image Selection:"]),
                                                dcc.Dropdown(
                                                    id="pps-dropdown-image-selection",
                                                    searchable=False,
                                                    clearable=False,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        )
                    ]),
                    dbc.Col([
                        html.Div(
                            id="pps-right-side-column",
                            children=[
                                html.Div(children=["Pointnet Part Segmenter Insights:"]),
                                dbc.Row([
                                    dcc.Loading(
                                        dcc.Graph(id='pps-model-training-history', style={"height": "70vh"}),
                                        type="cube",
                                    )
                                ]),
                                dbc.Row([
                                    html.Div(id="pps-div-detection-mode")
                                ])
                            ],
                        )
                    ]   )    
                ], style={'padding':'10px'}),
                part_segmenter_modal(),
            ])