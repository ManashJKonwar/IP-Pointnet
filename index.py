__author__ = "konwar.m"
__copyright__ = "Copyright 2021, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

from dash_pipeline.app import app
from dash_pipeline.layouts import layout_mainframe

app.title = 'Pointnet Implications'
app.layout = layout_mainframe.layout

if __name__ == '__main__':
     app.run_server(debug=True)