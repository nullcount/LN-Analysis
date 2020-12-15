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
import os.path

configPath = "config.yml"

def getConfig():
    if not os.path.isfile(configPath):
        print("%s does not exist. Copy example.config.yml to this path and edit with your configuration.\n\n    E.x. cp example.config.yml %s\n" % (configPath, configPath))
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
