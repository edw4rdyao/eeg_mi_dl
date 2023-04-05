from experiments.bci2a import bci2a
from experiments.physionet import physionet
import argparse
from utils import read_yaml
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bci2a', choices=['bci2a', 'physionet'])
    parser.add_argument('--dataset_cfg', type=str, default='default.yaml')
    parser.add_argument('--model', type=str, default='EEGNet', choices=['EEGNet', 'EEGNetRp', 'ST_GCN'])
    parser.add_argument('--model_cfg', type=str, default='default.yaml')
    parser.add_argument('--strategy', type=str, default='cross-subject',
                        choices=['cross-subject', 'within-subject'])
    parser.add_argument('--train_cfg', type=str, default='default.yaml')
    args = parser.parse_args()
    config = {
        'dataset': read_yaml(os.getcwd() + '\\config\\dataset\\' + args.dataset_cfg),
        'model': read_yaml(os.getcwd() + '\\config\\model\\' + args.model_cfg),
        'train': read_yaml(os.getcwd() + '\\config\\train\\' + args.train_cfg)
    }
    if args.dataset == 'bci2a':
        bci2a(args.model, args.strategy, config)
    elif args.dataset == 'physionet':
        physionet(args.model, args.strategy, config)
