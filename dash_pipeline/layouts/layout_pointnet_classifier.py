__author__ = "konwar.m"
__copyright__ = "Copyright 2021, AI R&D"
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

def classifier_modal():
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
                            dbc.Button("CLOSE BUTTON", id="classifier-modal-close", className="ml-auto")
                        ),
                    ],
                    id="modal", # Give the modal an id name 
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
                # dcc.Interval(id="interval-updating-graphs", interval=1000, n_intervals=0),
                # html.Div(id="top-bar", className="row"),
                dbc.Row([
                    dbc.Col(
                        html.Div(
                            id="left-side-column",
                            # className="eight columns",
                            children=[
                                html.Div(
                                    id="header-section",
                                    children=[
                                        html.H4("Pointnet Classifier Explorer"),
                                        html.P(
                                            "To get started, select the footage you want to view, and choose the display mode (with or without "
                                            "bounding boxes). Then, you can start playing the video, and the result of objects detected "
                                            "will be displayed in accordance to the current video-playing time."
                                        ),
                                        html.Button(
                                            "Learn More", id="learn-more-button", n_clicks=0
                                        ),
                                    ],
                                ),
                                html.Div(
                                    dcc.Graph(id='model-output', style={"height": "70vh"}), 
                                    # className="row"
                                ),
                                html.Div(
                                    # className="control-section",
                                    children=[
                                        html.Div(
                                            className="control-element",
                                            children=[
                                                html.Div(
                                                    children=["Minimum Confidence Threshold:"]
                                                ),
                                                html.Div(
                                                    dcc.Slider(
                                                        id="slider-minimum-confidence-threshold",
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
                                            # className="control-element",
                                            children=[
                                                html.Div(children=["Footage Selection:"]),
                                                dcc.Dropdown(
                                                    id="dropdown-footage-selection",
                                                    options=[
                                                        {
                                                            "label": "Drone recording of canal festival",
                                                            "value": "DroneCanalFestival",
                                                        },
                                                        {
                                                            "label": "Drone recording of car festival",
                                                            "value": "car_show_drone",
                                                        },
                                                        {
                                                            "label": "Drone recording of car festival #2",
                                                            "value": "DroneCarFestival2",
                                                        },
                                                        {
                                                            "label": "Drone recording of a farm",
                                                            "value": "FarmDrone",
                                                        },
                                                        {
                                                            "label": "Lion fighting Zebras",
                                                            "value": "zebra",
                                                        },
                                                        {
                                                            "label": "Man caught by a CCTV",
                                                            "value": "ManCCTV",
                                                        },
                                                        {
                                                            "label": "Man driving expensive car",
                                                            "value": "car_footage",
                                                        },
                                                        {
                                                            "label": "Restaurant Robbery",
                                                            "value": "RestaurantHoldup",
                                                        },
                                                    ],
                                                    value="car_show_drone",
                                                    clearable=False,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            # className="control-element",
                                            children=[
                                                html.Div(children=["Video Display Mode:"]),
                                                dcc.Dropdown(
                                                    id="dropdown-video-display-mode",
                                                    options=[
                                                        {
                                                            "label": "Regular Display",
                                                            "value": "regular",
                                                        },
                                                        {
                                                            "label": "Display with Bounding Boxes",
                                                            "value": "bounding_box",
                                                        },
                                                    ],
                                                    value="bounding_box",
                                                    searchable=False,
                                                    clearable=False,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            # className="control-element",
                                            children=[
                                                html.Div(children=["Graph View Mode:"]),
                                                dcc.Dropdown(
                                                    id="dropdown-graph-view-mode",
                                                    options=[
                                                        {
                                                            "label": "Visual Mode",
                                                            "value": "visual",
                                                        },
                                                        {
                                                            "label": "Detection Mode",
                                                            "value": "detection",
                                                        },
                                                    ],
                                                    value="detection",
                                                    searchable=False,
                                                    clearable=False,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        )
                    ),
                    dbc.Col(
                        html.Div(
                            id="right-side-column",
                            # className="four columns",
                            children=[
                                html.Div(id="div-visual-mode"),
                                html.Div(id="div-detection-mode"),
                            ],
                        )
                    )
                ]),
                classifier_modal(),
            ])