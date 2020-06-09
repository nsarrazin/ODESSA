import inspect
import os
from pathlib import Path
import importlib

"""This is used to pick up modules classes automatically
    and put them in a dictionary where they can be easily accessed by ODESSA dynamically
    rather than doing some import voodoo.

    import index from .modules
    to access the dictionary
"""
index = {}

# get the absolute path to modules folder
root_path = Path(os.path.abspath(__file__)).parent
walk = os.walk(root_path)


while True:
    try:  # go through the generator until it stops
        current = next(walk)
    except StopIteration:
        break

    if current[0].endswith("__pycache__"):  # skip the folders named __pycache__
        continue

    # get the path of the folder relative to the modules folder
    relpath = os.path.relpath(current[0], root_path).replace("\\", '.')

    for file in current[2]:  # iterate through files in the current subfolder

        # turns "folder/subfolder/file.py" into "folder.subfolder.file"
        importpath = (relpath + "/" + file).replace("/", ".")[:-3].lstrip(".")

        if "__" in importpath:
            continue  # ignore files and folders containing __

        # import the file and store its content in the module variable
        module = importlib.import_module(
            ".modules.{}".format(importpath), package="odessa")

        # go through the classes in the file namespace
        clsmembers = inspect.getmembers(module, inspect.isclass)

        for cls in clsmembers:
            # skip the numpy_array class that exists in the file namespace for some reason
            if cls[0] == 'numpy_array':
                continue

            # turns folder.subfolder.file into folder.subfolder.classname
            # we skip the filename bc you usually only have one class per file anyway and it creates repetition like
            # "gravity.gravity.gravityRTS" so now it would be "gravity.gravityRTS"

            indexpath = ".".join(importpath.split(".")[:-1])
            indexpath = (indexpath + "." + cls[0]).lstrip(".")

            # link a reference to the CLASS (not the object) in the index
            index[indexpath] = cls[1]
