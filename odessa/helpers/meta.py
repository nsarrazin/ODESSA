from collections import defaultdict
from pathlib import Path
from odessa.helpers.interpolators import interpolators

import numpy as np
import csv
import os
import sys
import json

def in_notebook():
    """
    Returns ``True`` if the module is running in IPython kernel,
    ``False`` if in IPython shell or other Python shell.
    """
    return 'ipykernel' in sys.modules


def set_data(obj, params):
    """Takes a python object and set attributes based on a dictionary of parameters.
        This is at the core of ODESSA when it loads. The JSON is used to set attributes of the modules
        directly. 
        
        It checks for special cases such as :

        - list > They get turned into numpy arrays.
        - if the value is a dictionary that contains a "method" attribute that is in the 'interpolators' dictionary
            then it loads the interpolator class
        - if the key is "JSON" then it loads the JSON as part of this JSON. 
            This allows for splitting really big JSON config in multiple smaller files.
        - If the value is something like 'file("data/thrust.csv")' then the CSV gets loaded as a numpy array.

    Args:
        obj ([object]): The object whose attributes need to be set
        params ([dict]): The dictionary containing the attributes to be set

    Returns:
        [object]: The object with its attributes set.
    """
    for key in params.keys():
        if type(params[key]) == type(dict()) and params[key]["method"] in interpolators.keys():
            interp = interpolators[params[key]["method"]]() # create the interpolator object based on the method string of the key
            del params[key]["method"]
            interp = set_data(interp, params[key])
            setattr(obj, key, interp)
            continue

        # check for list to turn into arrays
        if type(params[key]) == type(list()):
            setattr(obj, key, np.ascontiguousarray(params[key], dtype=np.float64))
            continue
        # check for files to read
        if type(params[key]) == type(str()) and params[key].startswith("file('"):
            setattr(obj, key, read_file(params[key]))
            continue
        
        if key == "JSON":
            config = read_json(params[key])
            set_data(obj, config)
            continue

        # check for infinity
        if params[key] == 'inf':
            setattr(obj, key, np.inf)
            continue
        setattr(obj, key, params[key])
    return obj

def read_json(string):
    """Takes the path of a JSON and returns the equivalent object

    Args:
        string ([string]): Path to the JSON

    Returns:
        [dict]: The loaded JSON as a python object
    """
    path = string.split("json('")[1].rstrip("')")

    if in_notebook():   # FIXME: Loading notebooks fucks up everything bc
        # they're designed to hide their own path
        cwd = "/home/nsarrazin/Documents/DARE/StratosIV/Libraries/ODESSA/test/validation/proteus6"
    else:
        cwd = Path().cwd()

    filename = os.path.join(cwd, path)

    with open(filename) as json_file:
        config = json.loads(json_file.read(), strict=False)

    return config

def read_file(string):
    """Take the path of a csv file and load it as a numpy array
       while skipping the first row IF it contains headers.

       Uses a comma as a delimiter

    Args:
        string ([type]): Path to the CSV

    Returns:
        [nparray]: The loaded numpy array
    """
    path = string.split("file('")[1].split('[')[0].rstrip("')")

    if in_notebook():   # FIXME: Loading notebooks fucks up everything bc
        # they're designed to hide their own path
        cwd = "/home/nsarrazin/Documents/DARE/ODESSA/scripts/"
    else:
        cwd = Path().cwd()

    filename = os.path.join(cwd, path)

    columns = defaultdict(list)
    index = int(string.rsplit("[")[-1].rstrip(']'))

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for n, row in enumerate(csv_reader):
            if n == 0:  # skip header
                try:
                    for v in row:  # if first row cant be cast to float then
                        float(v)   # its a header row and we skip it.
                except ValueError:
                    continue
                pass
            for (i, v) in enumerate(row):
                columns[i].append(v)

    return np.array(columns[index], dtype=np.float64)


def find_key(mydict, value):
    """Finds a dictionary key based on the value. Big yikes but it works

    Args:
        mydict ([dict]): The dictionary
        value ([*]): The value to lookup

    Returns:
        [*]: The key corresponding to that value
    """
    return list(mydict.keys())[list(mydict.values()).index(value)]