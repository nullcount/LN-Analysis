###
# snapshot.py
#
# Generate a timestamped snapshot of the Lightning Netowrk graph
# Must be run by a node with lnd installed
# Optional: copy the graph.json snapshot over ssh 
#
# Usage: Run script with cron
#
#   python3 snapshot.py 
#   python3 snapshot.py --remote-write
###

import os
import time
import json
import sys
from helpers import isCommand, sh, grab, getConfig 

config = getConfig()

def getSnapshot():
    """Get LND to describe the network graph"""
    data = {}
    data["timestamp"] = int(time.time())
    data["graph"] = json.loads(grab("lncli describegraph"))
    return data

def remoteWrite(string):
    open(config["tempfile"], "w").write(string)
    sh('scp %s %s:%s' % (config["tempfile"], config["remote_host"], config["remote_file"]))
    os.remove(config["tempfile"])

def localWrite(string):
    open(config["localfile"], "w").write(string)

def main():
    if config is False:
        print("Config file error. Exiting.")
        return 
    if not isCommand("lncli"):
        print("LND is not installed. Exiting.")
        return
    print("Getting snapshot... please wait...")
    snapshot = json.dumps(getSnapshot())
    if "--remote-write" in sys.argv:
        remoteWrite(snapshot)
    else:
        localWrite(snapshot)

if __name__ == '__main__':
    main()
