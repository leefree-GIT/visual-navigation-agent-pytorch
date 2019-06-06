#!/usr/bin/env python
import torch
import argparse
import multiprocessing as mp

from agent.evaluation import Evaluation

if __name__ == '__main__':
    # mp.set_start_method('spawn')
    argparse.ArgumentParser(description="")
    parser = argparse.ArgumentParser(description='Deep reactive agent.')
    parser.add_argument('--h5_file_path', type = str, default='/app/data/{scene}_keras.h5')
    parser.add_argument('--checkpoint_path', type = str, default='/model/checkpoint-{checkpoint}.pth')
    parser.add_argument('--csv_file', type = str, default=None)
    parser.add_argument('--resnet', action='store_true')
    parser.add_argument('--cuda', type = str, default='0')
    args = vars(parser.parse_args())

    t = Evaluation.load_checkpoint(args)
    t.run()

