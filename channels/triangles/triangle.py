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
        'pubkey': (len(sys.argv) >= 2, lambda: input("Enter node public key: ")),
        'max_results': (len(sys.argv) >= 3, lambda: int(input("Max results to return: "))),
        'graph': (len(sys.argv) >= 4, lambda: graphSelector()),
    }
    pubkey, max_results, graph= get_arguments(get)
    max_results = int(max_results)
    neighbors = {}
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

    ranked = sorted(sort, key=sort.__getitem__)
    for triangle in ranked[len(ranked)-max_results:len(ranked)]:
        print()
        tri = neighbors[triangle]
        print(str(len(tri['peers'])) + " triangles with:")
        print(triangle)
        print()
        for p in tri['peers']:
            if 'alias' in G.nodes[p]:
                print(G.nodes[p]['alias'])
            else:
                print(p)
        print()

if __name__ == '__main__':
    main()
