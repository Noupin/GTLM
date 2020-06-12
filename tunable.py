__author__ = "Noupin, TensorFlow"

#Third Party Imports
import json

class Tunable:

    with open(r"C:\Coding\Python\ML\Text\transformer\tunable.json") as jsonFile:
        tunableVars = json.load(jsonFile)