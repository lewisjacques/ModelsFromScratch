# Required to handle ./functions as a package
# Handles aggregating all of the wrapped functions for import

from importlib import import_module
from os import listdir
import re

# Get current file path
current_file_path = "/".join(__file__.split("/")[:-1])
adjacent_files = listdir(current_file_path)

# Find all function-files
function_files = [
    f[:-3] for f in adjacent_files \
    if re.match(r"[a-z0-9_]+functions.py",f) and \
    not re.match(r"_all_functions.py", f)
]

# Build a dictionary of all imports
function_dict = {
    file_name: 
        import_module(f"functions.{file_name}")._fh.get_functions() \
        for file_name in function_files
}

print(function_dict)