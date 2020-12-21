###
# helpers.py
#
# Helper functions for use in this project
#
# Usage: from helpers import *
###

from shutil import which
import subprocess
import yaml
import json
import glob
import os.path
import pickle
from datetime import datetime
import networkx as nx

configPath = "./config.yml"

def getConfig():
    if not os.path.isfile(configPath):
        print("%s does not exist. Copy example.config.yml to this path and edit with your configuration.\n\n    E.x. cp example.config.yml %s\n" % (configPath, configPath))
        return False
    with open(configPath) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def sh(command):
    subprocess.call(command.split(' '))

def grab(command):
    return str(subprocess.check_output(command.split(' ')), 'utf-8')

def isCommand(name):
    """Check whether `name` is on PATH and marked as executable."""
    return which(name) is not None

def getDict(f):
    with open(f) as j:
        return json.load(j)

def getGraph(graphJson):
    # Create an empty graph
    G = nx.Graph()

    # Parse and add nodes
    for node in graphJson['graph']['nodes']:
        G.add_node(
            node['pub_key'],
            alias=node['alias'],
            addresses=node['addresses'],
            color=node['color'],
            last_update=['last_update']
        )

    # Parse and add edges
    for edge in graphJson['graph']['edges']:
        G.add_edge(
            edge['node1_pub'],
            edge['node2_pub'],
            channel_id=edge['channel_id'],
            chan_point=edge['chan_point'],
            last_update=edge['last_update'],
            capacity=edge['capacity'],
            node1_policy=edge['node1_policy'],
            node2_policy=edge['node2_policy']
        )

    return G

def pickleIt(g, f):
    pickle.dump(g, open(f, "wb"))

def depickle(f):
    return pickle.load(open(f, "rb"))

def mkdir(d):
    """Makes a directory if it does not exist"""
    if not os.path.exists(d):
        os.makedirs(d)

def picklePicker(jarPath):
    pickles = glob.glob("%s/*.p" % (jarPath))
    if len(pickles) == 0:
        print("No pickled graphs found. Run graphize.py or check your pickle jar. Exiting.")
        return
    if len(pickles) == 1:
        return depickle(pickles[0])
    else:
        print("Showing all saved pickles:\n")
        for i, pickle in enumerate(pickles):
            file = pickle.split("/")[-1]
            date = datetime.utcfromtimestamp(int(file.split('.')[0])).strftime('%Y-%m-%d %H:%M:%S')
            print("%s - %s from %s" % (i, file, date))
        choice = input("\nPlease select a pickle. Enter a number 0-%s: " % (len(pickles)-1))
        return depickle(pickles[int(choice)])
