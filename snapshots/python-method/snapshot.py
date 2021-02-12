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
from helpers import isCommand, sh, grab, mkdir, getConfig 

config = getConfig()

timestamp = int(time.time())
localFile = "%s-%s" % (timestamp, config['localfile'])

def getSnapshot():
    """Get LND to describe the network graph"""
    data = {}
    data["timestamp"] = timestamp
    data["graph"] = json.loads(grab("%s describegraph" % (config['lncli_path'])))
    return data

def remoteWrite(string):
    open(config["tempfile"], "w").write(string)
    sh('scp %s %s:%s' % (config["tempfile"], config["remote_host"], config["remote_file"]))
    if "--archive" in sys.argv:
        mkdir(config['json_archive'])
        sh("cp %s %s/%s" % (config["tempfile"], config['json_archive'], localFile))
    os.remove(config["tempfile"])

def localWrite(string):
    open(localFile, "w").write(string)
    if "--archive" in sys.argv:
        mkdir(config['json_archive'])
        sh("mv %s %s/" % (localFile, config['json_archive']))

def main():
    if config is False:
        print("Config file error. Exiting.")
        return 
    if not isCommand(config["lncli_path"]):
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
