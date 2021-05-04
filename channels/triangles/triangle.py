import json
import networkx as nx
import sys

filename = 'graph.json'

pubkey = sys.argv[1]

def getGraph(graphJson):
    # Create an empty graph
    G = nx.Graph(name='graph')

    # Parse and add nodes
    for node in graphJson['nodes']:
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
    for edge in graphJson['edges']:
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

def getPeers(G, pubkey):
    peers = []
    for chan in G.edges(pubkey):
        for point in chan:
            if point != pubkey:
                peers.append(point)
    return peers

neighbors = {}

with open(filename, 'r') as f:
    graph = json.load(f)
    G = getGraph(graph)
    peers = getPeers(G, pubkey)
    for peer in peers:
        for neighbor in getPeers(G, peer):
            if neighbor in neighbors and neighbor != pubkey:
                neighbors[neighbor]['count'] += 1
                neighbors[neighbor]['peers'].append(peer)
            else:
                neighbors[neighbor] = {'count': 0, 'peers':[peer] }
    sort = {}
    for neighbor in neighbors:
        count = neighbors[neighbor]['count']
        if count > 0 and neighbor != pubkey:
            sort[neighbor] = count

    for triangle in sorted(sort, key=sort.__getitem__):
        tri = neighbors[triangle]
        print(triangle)
        for p in tri['peers']:
            print(G.nodes[p]['alias'])
        print()
