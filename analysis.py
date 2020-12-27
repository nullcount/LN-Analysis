from helpers import getConfig, picklePicker
import networkx as nx

config = getConfig()

def main():
    G = picklePicker(config['pickle_jar'])
    print(list(nx.bridges(G)))

if __name__ == '__main__':
    main()
