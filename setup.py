from setuptools import find_packages
from cx_Freeze import setup, Executable
import os
import sys


def find_data_file(filename):
    if getattr(sys, "frozen", False):
        # The application is frozen
        datadir = os.path.dirname(sys.executable)
    else:
        # The application is not frozen
        # # Change this bit to match where you store your data files:
        datadir = os.path.dirname(__file__)
    return os.path.join(datadir, filename)


options = {
    'build_exe': {
        'includes': [
            'cx_Logging', 'idna',
        ],
        'packages': [
            'asyncio', 'flask', 'jinja2', 'dash', 'plotly', 'waitress', 'tensorflow', 'keras', 'os', 'pye57', 
            'laspy', 'sklearn', 'matplotlib', 'pandas', 'numpy', 'dash_core_components', 'dash_html_components', 
            'dash_extensions', 'dash_bootstrap_components', 'os', 'base64', 'datetime', 'ezdxf'
        ],
        'excludes': [
            'tkinter'
        ],
        "include_files": [
            'assets/image4.png','model_VGG.h5','assets/logo.png', 'model.py', 'assets/','final_model.h5',
            'assets/image3.png','assets/logo2.png','assets/style.css', 'assets/favicon.ico', 'ico.ico'
        ]
    }
}

executables = [
    Executable('server.py',
               base='console',
               targetName='Detection_Catenaires.exe',
               icon = "ico.ico")
]

setup(
    name='Detection_Catenaires',
    packages=find_packages(),
    version='0.4.0',
    description='rig',
    executables=executables,
    options=options
)