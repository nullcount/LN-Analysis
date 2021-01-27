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
from os.path import join
import pickle
from datetime import datetime
import networkx as nx
from bs4 import BeautifulSoup as bs
import requests
from many_requests import ManyRequests

configPath = "./config.yml"


def getConfig():
    if not os.path.isfile(configPath):
        print(
            "%s does not exist. Copy example.config.yml to this path and edit with your configuration.\n\n    E.x. cp example.config.yml %s\n" % (
                configPath, configPath))
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
        if node['last_update'] == 0:
            continue
        G.add_node(
            node['pub_key'],
            alias=node['alias'],
            addresses=node['addresses'],
            color=node['color'],
            last_update=['last_update']
        )

    # Parse and add edges
    for edge in graphJson['graph']['edges']:
        if edge['last_update'] == 0:
            continue
        if edge['node1_policy'] is None or edge['node2_policy'] is None:
            continue
        if edge['node1_policy']['disabled'] is None or edge['node2_policy']['disabled'] is None:
            continue
        G.add_edge(
            edge['node1_pub'],
            edge['node2_pub'],
            weight=1,
            channel_id=edge['channel_id'],
            chan_point=edge['chan_point'],
            last_update=edge['last_update'],
            capacity=edge['capacity'],
            node1_policy=edge['node1_policy'],
            node2_policy=edge['node2_policy']
        )
    G.remove_nodes_from([x for x in G.nodes() if G.degree[x] == 0])

    return G


def remove_subs(G):
    """
    Removes all subgraphs besides the largest one.
    """
    components = list(nx.connected_components(G))
    mainnet = max(components, key=len)
    components.remove(mainnet)
    for component in components:
        G.remove_nodes_from(component)
    return G


def mkdir(d):
    """Makes a directory if it does not exist"""
    if not os.path.exists(d):
        os.makedirs(d)


def graphSelector():
    config = getConfig()
    graphs = glob.glob("%s/*" % (config['json_archive']))
    if len(graphs) == 0:
        print(
            "No graphs found in %s. Run snapshot.py on your lightning node or grab a snapshot from somewhere else. Exiting." %
            config['json_archive'])
        return
    if len(graphs) == 1:
        return getGraph(getDict(graphs[0]))
    else:
        print("Showing all archived graph json:\n")
        for i, graph in enumerate(graphs):
            file = graph.split("/")[-1]
            date = datetime.utcfromtimestamp(int(file.split('-')[0])).strftime('%Y-%m-%d %H:%M:%S')
            print("%s - %s from %s" % (i, file, date))
        choice = input("\nPlease select a snapshot. Enter a number 0-%s: " % (len(graphs) - 1))
        return getGraph(getDict(graphs[int(choice)]))


def get_merchants():
    # go to directory
    # grab all categories
    # go to all categories
    # grab all pub keys on each page
    # return a list of pubkeys

    if os.path.isfile("merchants.txt"):
        print("Merchants list found, delete it to update.")
        with open("merchants.txt", "r") as fileobj:
            pub_keys = fileobj.read().splitlines()
    else:
        print("Merchant list updating...")
        base_url = "https://1ml.com"
        directory_link = join(base_url, "directory")
        response = requests.get(directory_link)
        directory_soup = bs(response.content, "html.parser")
        categories = directory_soup.find_all("li", {"class": "list-group-item"})[1:]
        links = []

        for category in categories:
            links.extend(category.find_all("a", {"title": True}))
        links = [link["href"] for link in links]
        print(links)
        links = [base_url + link for link in links]  # idk why join doesnt work here
        print(links)

        responses = ManyRequests(n_workers=10, n_connections=10)(
            method='GET', url=links)

        pub_keys = []
        for response in responses:
            soup = bs(response.content, "html.parser")
            pub_keys.extend(soup.find_all("strong", {"class": "small selectable"}))

        pub_keys = list(set([pub_key.text for pub_key in pub_keys]))

        with open("merchants.txt", "w") as fileobj:
            fileobj.writelines([pub_key+"\n" for pub_key in pub_keys])
    return pub_keys
