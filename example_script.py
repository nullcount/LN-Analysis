from helpers import getConfig, picklePicker

config = getConfig()

def main():
    G = picklePicker(config['pickle_jar'])
    print(list(G.nodes))

if __name__ == '__main__':
    main()
