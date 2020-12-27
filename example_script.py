from helpers import graphSelector,remove_subs
import networkx as nx
from copy import deepcopy
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
    all_components = []
    for node in filtered_a_nodes:
        components = temp_remove(G, node)
        mainnet = max(components,key=len)
        components.remove(mainnet)
        print(G.degree[node],len(components),sum([len(x) for x in components]))
        all_components.extend(components)

    print([len(x) for x in all_components])

def temp_remove(G,v_id):
    temp_G = deepcopy(G)
    temp_G.remove_nodes_from([v_id])
    components = list(nx.connected_components(temp_G)) # AT LEAST 2
    return  components
# create a copy of the graph, remove the node, identify the largest component
# the sum of the sizes of the rest is the size of the cluster depenedent on that articulation_point





if __name__ == '__main__':
    main()
