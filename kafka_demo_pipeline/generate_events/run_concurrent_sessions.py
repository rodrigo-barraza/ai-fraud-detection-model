# Script that creates multiple instances of a random user event session so you can spin of n concurrent random sessions.

import sys
import subprocess

import argparse

import os

import wait

wait.for_topics(['events'], host='kafka-rest',port='29080')

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
filename = os.path.join(__location__, 'produce_random_session.py')

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--n_sessions', required=True)
args = parser.parse_args()

procs = []
for i in range(int(args.n_sessions)):
    proc = subprocess.Popen([sys.executable, filename])
    procs.append(proc)

for proc in procs:
    proc.wait()