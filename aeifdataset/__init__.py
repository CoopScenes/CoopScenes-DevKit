from .core import Dataloader, DataRecord
from data import *
from utils import *

import re
import pathlib

# Path to setup.py
setup_path = pathlib.Path(__file__).parent.parent / 'setup.py'


# Function to extract version
def get_version():
    version_pattern = re.compile(r"version=['\"]([^'\"]*)['\"]")
    with open(setup_path, 'r') as file:
        setup_content = file.read()
        match = version_pattern.search(setup_content)
        if match:
            return match.group(1)
        return 'unknown'


# Set the version attribute
__version__ = get_version()
