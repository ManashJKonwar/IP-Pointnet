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

def classifier_modal():
    return html.Div([
                dbc.Modal(
                    [
                        dbc.ModalHeader("UNDERSTANDING POINTNET CLASSIFIER"),
                        dbc.ModalBody(
                            html.Div([
                                    dcc.Markdown(
                                        '''
                                        ## What are you looking at?
                                        
                                        This app helps us to visualize pointnet objects and also helps us in running predictions for them.
                                        Classification, detection and segmentation of unordered 3D point sets i.e. point clouds is a core problem in computer vision. 
                                        This case study implements the seminal point cloud deep learning paper PointNet (Qi et al., 2017). 
                                        
                                        ## Dataset Utilzation and Data Transformation:
                                        ***

                                        Please refer to the pointers below:
                                        
                                        1. Once the ModelNet 10 dataset is downloaded, we parse each of the data folders. Each of this file is loaded and further sampled into  
                                        a point cloud before further converting them to numpy array. Along with the pointcloud representation, we also store the object label of each object  
                                        into a dictionary and utilize it as part of prediction pipeline.
                                        2. Set Global Parameters required for training such as  
                                        &nbsp;&nbsp;&nbsp;&nbsp; NUM_POINTS -> Number of points to sample for each object under each class.  
                                        &nbsp;&nbsp;&nbsp;&nbsp; NUM_CLASSES -> Total number of object classes to be considered for training the classifier.  
                                        &nbsp;&nbsp;&nbsp;&nbsp; BATCH_SIZE -> Number of samples that will be propagated through the network.  
                                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Advantages of using a batch size < number of all samples:  
                                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp a. It requires less memory. Since you train the network using fewer samples, the overall training procedure requires less memory.  
                                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp That's especially important if you are not able to fit the whole dataset in your machine's memory.  
                                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp b. Typically networks train faster with mini-batches. That's because we update the weights after each propagation. In our example we've  
                                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp propagated 32 batches (32 of them had 32 samples) and after each of them we've updated our network's parameters.  
                                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp If we used all samples during propagation we would make only 1 update for the network's parameter.  
                                        3. Data Augmentation is a very crucial step for training pipeline as it increases the total sample size and also help us to treat difefrent point cloud  
                                        scenarios.  
                                        4. Next Step would be to build the pointnet model in keras based on the Architecture defined in original journal paper. 

                                        ## Model Building
                                        ***
                                        '''
                                    ),
                                    html.Img(
                                        id='pointnet_classifier',
                                        src=r'/assets/Pointnet_Classifier.png', 
                                        style={'width':'60%'}
                                    ),
                                    dcc.Markdown(
                                        '''
                                        &nbsp;&nbsp;&nbsp;&nbsp; In the classifier architecture above, Each convolution and fully-connected layer (with exception for end layers) consists of  
                                        Convolution / Dense -> Batch Normalization -> ReLU Activation.
                                        '''
                                    ),
                                    html.Img(
                                        id='pointnet_classifier_cl_fc',
                                        src=r'/assets/Pointnet_Classifier_CL_FC.png', 
                                        style={'width':'60%'}
                                    ),
                                    dcc.Markdown(
                                        '''
                                        ##### More about this Dash app
                                        
                                        The purpose of this demo is to explore alternative visualization methods for object detection. Therefore,
                                        the visualizations, predictions and videos are not generated in real time, but done beforehand. To read
                                        more about it, please visit the [project repo](https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-object-detection).
                                        '''
                                    )
                                ]
                            ),
                            style={"height": "70vh"}
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
                dbc.Row([
                    dbc.Col([
                        html.Div(
                            id="left-side-column",
                            children=[
                                html.Div(
                                    id="header-section",
                                    children=[
                                        html.H4("Pointnet Classifier Explorer"),
                                        html.P(
                                            "To get started, select the object category and choose the image no after that, you will be able to see " 
                                            "the pointnet structure of the same in the graphical area, further the groundtruth and prediction labels "
                                            "could be observed in the right bottom corner of your screen. You can also see the model training characteristics "
                                            "on the right top corner of your screen. Select the appropriate model type after training the pointnet classifier "
                                            "model"
                                        ),
                                        html.Button(
                                            "Learn Pointnet Classifier", id="learn-more-button", n_clicks=0
                                        ),
                                    ],
                                ),
                                html.Div(
                                    children=[
                                        html.Div(children=["Model Selection:"]),
                                        dcc.Dropdown(
                                            id="dropdown-model-selection",
                                            options = [{'label': classifier_item, 'value': classifier_item} for classifier_item in ['pointnet_10cls_20epochs']],
                                            value='pointnet_10cls_20epochs',
                                            clearable=False,
                                        ),
                                    ],
                                ),
                                html.Div(
                                    dcc.Loading(
                                        dcc.Graph(id='model-output', style={"height": "70vh"}),
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
                                            children=[
                                                html.Div(children=["Class Selection:"]),
                                                dcc.Dropdown(
                                                    id="dropdown-class-selection",
                                                    options = [{'label': class_item.capitalize(), 'value': class_item} for class_item in class_map.values()],
                                                    value=list(class_map.values())[0],
                                                    clearable=False,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            children=[
                                                html.Div(children=["Image Selection:"]),
                                                dcc.Dropdown(
                                                    id="dropdown-image-selection",
                                                    searchable=False,
                                                    clearable=False,
                                                ),
                                            ],
                                        ),
                                        # html.Div(
                                        #     children=[
                                        #         html.Div(children=["Graph View Mode:"]),
                                        #         dcc.Dropdown(
                                        #             id="dropdown-graph-view-mode",
                                        #             options=[
                                        #                 {
                                        #                     "label": "Visual Mode",
                                        #                     "value": "visual",
                                        #                 },
                                        #                 {
                                        #                     "label": "Detection Mode",
                                        #                     "value": "detection",
                                        #                 },
                                        #             ],
                                        #             value="detection",
                                        #             searchable=False,
                                        #             clearable=False,
                                        #         ),
                                        #     ],
                                        # ),
                                    ],
                                ),
                            ],
                        )
                    ]),
                    dbc.Col([
                        html.Div(
                            id="right-side-column",
                            children=[
                                html.Div(children=["Pointnet Classifier Insights:"]),
                                dbc.Row([
                                    dcc.Loading(
                                        dcc.Graph(id='model-training-history', style={"height": "70vh"}),
                                        type="cube",
                                    )
                                ]),
                                # html.Div(id="div-visual-mode"),
                                dbc.Row([
                                    html.Div(id="div-detection-mode")
                                ])
                            ],
                        )
                    ]   )    
                ], style={'padding':'10px'}),
                classifier_modal(),
            ])