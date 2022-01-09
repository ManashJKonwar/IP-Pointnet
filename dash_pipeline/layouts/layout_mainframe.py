__author__ = "konwar.m"
__copyright__ = "Copyright 2021, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import Input, Output, State, html

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

# make a reuseable dropdown for the different examples
dropdown = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem("Pointnet Classification", href="/pointnet-classification"),
        dbc.DropdownMenuItem(divider=True),
        dbc.DropdownMenuItem("Pointnet Part Segmentation", href="/pointnet-part-segmentation"),
        dbc.DropdownMenuItem(divider=True),
        dbc.DropdownMenuItem("Pointnet Semantic Segmentation", href="/pointnet-semantic-segmentation"),
    ],
    nav=True,
    in_navbar=True,
    label="Menu",
)

# this example that adds a logo to the navbar brand
navbar = dbc.Navbar(
            dbc.Container(
                [
                    html.A(
                        # Use row and col to control vertical alignment of logo / brand
                        dbc.Row(
                            [
                                dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                                dbc.Col(dbc.NavbarBrand("Pointnet Platter", className="ms-2")),
                            ],
                            align="center",
                            className="g-0",
                        ),
                        href="https://plotly.com",
                        style={"textDecoration": "none"},
                    ),
                    dbc.NavbarToggler(id="navbar-toggler2", n_clicks=0),
                    dbc.Collapse(   
                        dbc.Nav(
                            # [nav_item, dropdown],
                            [dbc.NavLink('Pointnet Classification', href="/pointnet-classification", active='exact'),
                            dbc.NavLink('Pointnet Part Segmentation', href="/pointnet-part-segmentation", active='exact'),
                            dbc.NavLink('Pointnet Semantic Segmentation', href="/pointnet-semantic-segmentation", active='exact')],
                            # dropdown],
                            className="ms-auto",
                            navbar=True,
                        ),
                        id="navbar-collapse2",
                        navbar=True,
                    ),
                ],
                fluid=True
            ),
            color="dark",
            dark=True,
            className="mb-5"
        )

content = html.Div(id="page-content", className="content")
layout = html.Div([dcc.Location(id="url"), navbar, content])