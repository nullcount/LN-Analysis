from helpers import graphSelector,remove_subs
import networkx as nx
from copy import deepcopy

import subprocess
from subprocess import call
import threading
import base64
from concurrent.futures import ThreadPoolExecutor as PoolExecutor


def main():
    G = graphSelector()
    G = remove_subs(G)# remove all subgraphs besides the largest (mainnet)

    allArticulators = list(nx.articulation_points(G))
    #all_nodes = []
    #for x,y in allBridges:
    #    all_nodes.extend([x,y])
    #print(allArticulators)
    #all_nodes = list(set(allBridges))
    print(f"{len(allArticulators)} * {len(G.edges())}")
    filtered_a_nodes = list(filter(lambda n: G.degree[n]>1,allArticulators))
    #print(len(all_nodes))
    #btwn_centralities = nx.betweenness_centrality(G) # slowwwwww
    #filtered_a_nodes = sorted(filtered_a_nodes,key = lambda x: btwn_centralities[x])
    #print(btwn_centralities[all_nodes[0]])
    all_components = get_all_components(filtered_a_nodes,G)
    flat_list =[]
    for component_set in all_components:
        flat_list.extend(list(component_set))

    print(all_components)

    print(len(flat_list),len(all_components))
    new_G = deepcopy(G).remove_nodes_from(flat_list)
    print(len(G.nodes()))
    print(len(new_G.nodes()),len(new_G.edges()))

def get_all_components(articulation_nodes,G):
    all_components = []
    for node in articulation_nodes:
        components = temp_remove(G, node)
        mainnet = max(components,key=len)
        components.remove(mainnet)
        nodes_in_components = sum([len(x) for x in components])
        nbr_of_components = len(components)

        print(G.degree[node],nbr_of_components,nodes_in_components)
        if nodes_in_components/nbr_of_components > 1:
            print(node)
        all_components.extend(components)
    return all_components

def temp_remove(G,v_id):
    temp_G = deepcopy(G)
    temp_G.remove_nodes_from([v_id])
    components = list(nx.connected_components(temp_G)) # AT LEAST 2
    return  components
# create a copy of the graph, remove the node, identify the largest component
# the sum of the sizes of the rest is the size of the cluster depenedent on that articulation_point





if __name__ == '__main__':
    main()
