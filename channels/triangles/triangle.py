import json
import networkx as nx
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import *

def getPeers(G, pubkey):
    peers = []
    for chan in G.edges(pubkey):
        for point in chan:
            if point != pubkey:
                peers.append(point)
    return peers


def main():
    get = {
        'pubkey': (len(sys.argv) >= 2, lambda: input("Enter node public key: "))
    }
    pubkey = get_arguments(get)[0]
    print(pubkey)
    neighbors = {}
    graph = graphSelector()
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

if __name__ == '__main__':
    main()
