import sys
import os

# Note that we import from the function subdirectory in the function directory.
# The function directory is created in the build directory by the faas-cli
# build commmand.
from function import readfile

def handle(req):

    current_dir = os.path.dirname(__file__)

    # Testing two ways to reference the message.txt file.

    # 1. Using the current directory.
    return readfile.readFile(os.path.join(current_dir, "message.txt"))

    # Or,
    # 2. Knowing we're running in openfass make it relative to the docker container root.
    #return readfile.readFile('function/message.txt')