#!/usr/bin/env python
import argparse
import multiprocessing as mp
import torch

from agent.evaluation import Evaluation
from agent.utils import populate_config

import sys

class Logger(object):
    def __init__(self, path="logfile.log"):
        self.terminal = sys.stdout
        self.log = open(path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

if __name__ == '__main__':
    # mp.set_start_method('spawn')
    argparse.ArgumentParser(description="")
    parser = argparse.ArgumentParser(description='Deep reactive agent.')
    parser.add_argument('--h5_file_path', type = str, default='/app/data/{scene}_keras.h5')
    parser.add_argument('--checkpoint_path', type = str, default=None)
    parser.add_argument('--csv_file', type = str, default=None)
    parser.add_argument('--log_arg', type = int, default=0 )

    # Use experiment.json
    parser.add_argument('--exp', '-e', type = str, help='Experiment parameters.json file', required=True)

    args = vars(parser.parse_args())
    if args['checkpoint_path'] is not None:
        args = populate_config(args, mode='eval', checkpoint=False)
    else:
        args = populate_config(args, mode='eval')

    sys.stdout = Logger(args['base_path'] + 'eval' + str(args['log_arg']) + '.log')

    t = Evaluation.load_checkpoint(args)
    t.run()

