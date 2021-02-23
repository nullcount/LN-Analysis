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
from pathlib import Path
from copy import deepcopy

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
    G = nx.Graph(name=graphJson["timestamp"])

    # Parse and add nodes
    for node in graphJson['graph']['nodes']:
        if node.get('last_update', 0) == 0:
            continue
        G.add_node(
            node['pub_key'],
            alias=node['alias'],
            addresses=node['addresses'],
            color=node['color'],
            last_update=node['last_update']
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
            # weight=1,
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


def clean_graph(graph, save=True):
    # take NetworkX Graph as input
    articulation_points = list(nx.articulation_points(graph))
    # remove articulation points, then remove components, then add back in articulation points
    graph_copy = deepcopy(graph)  # because adding back the articulation points does not add back the edges
    graph_copy.remove_nodes_from(articulation_points)
    components = list(nx.connected_components(graph_copy))
    mainnet = max(components, key=len)
    kept_nodes = mainnet | set(articulation_points)
    removed_nodes = set(graph.nodes()) - kept_nodes
    graph.remove_nodes_from(removed_nodes)
    graph = remove_subs(graph)

    # save graph in cleaned_graphs dir
    if save:
        config = getConfig()
        save_graph(graph, loc=config["clean_dir"])
    return graph


def save_graph(graph, loc):
    graph_data = nx.readwrite.node_link_data(graph, {"name": "pub_key",
                                                     "link": "edges",
                                                     "source": "node1_pub",
                                                     "target": "node2_pub"})
    graph_data = {"graph": graph_data, "timestamp": graph.name}
    del graph_data["graph"]["directed"]
    del graph_data["graph"]["multigraph"]
    del graph_data["graph"]["graph"]
    mkdir(loc)
    with open(join(loc, str(graph.name) + '-graph.json'), 'w') as outfile:
        json.dump(graph_data, outfile)


def mkdir(d):
    """Makes a directory if it does not exist"""
    Path(d).mkdir(exist_ok=True)


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
            file = graph.split("/")[-1].lstrip("//graphs\\")
            date = datetime.utcfromtimestamp(int(file.split('-')[0])).strftime('%Y-%m-%d %H:%M:%S')
            print("%s - %s from %s" % (i, file, date))
        choice = input("\nPlease select a snapshot. Enter a number 0-%s: " % (len(graphs) - 1))
        return getGraph(getDict(graphs[int(choice)]))


def get_merchant_data():
    # go to directory
    # grab all categories
    # go to all categories
    # grab all pub keys on each page
    # get json data of each merchant
    # return a list of merchant data

    if os.path.isfile("merchants.json"):
        print("Merchants json found, delete it to update.")
        with open("merchants.json", "r") as fileobj:
            merchant_dict = json.load(fileobj)
    else:
        print("Merchants json updating...")
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

        merchant_data = get_1ml_data(pub_keys)
        merchant_dict = convert_data(merchant_data)

        with open("merchants.json", "w") as fileobj:
            json.dump(merchant_dict, fileobj)
    return merchant_dict


def get_1ml_data(pubkeys: list) -> list:
    base_url = "https://1ml.com/node/{}/json"
    urls = [base_url.format(pubkey) for pubkey in pubkeys]
    all_json = []
    responses = ManyRequests(n_workers=20, n_connections=20)(
        method='GET', url=urls)
    for response in responses:
        son = response.json()
        all_json.append(son)
    return all_json


def convert_data(merchant_data: list) -> dict:
    # map each element in the list to its pubkey and return the resulting dict
    merchant_dict = {}
    for merchant in merchant_data:
        pubkey = merchant["pub_key"]
        merchant.pop("pub_key", None)
        merchant_dict[pubkey] = merchant
    return merchant_dict
