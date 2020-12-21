###
# graphize.py
#
# Generate a NetworkX graph object from the json.
# Save it to a pickle for digesting in other scripts.
#
# Usage:
#
#   python3 graphize.py /path/to/graph.json
###

import os
import sys
import json
from helpers import getConfig, getDict, getGraph, pickleIt, mkdir


config = getConfig()
jsonFile = sys.argv[1]

def main():
    if config is False:
        print("Config file error. Exiting.")
        return
    if not os.path.isfile(jsonFile):
        print("Argument %s is not a file. Exiting." % (jsonFile))
        return
    graphJson = getDict(jsonFile)
    graph = getGraph(graphJson)
    mkdir(config["pickle_jar"])
    pickleIt(graph, '%s/%s.p' % (config['pickle_jar'], graphJson['timestamp']))

if __name__ == '__main__':
    main()
