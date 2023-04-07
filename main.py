from experiments.bci2a import bci2a
from experiments.physionet import physionet
import argparse
from utils import read_yaml, save_config
import time
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bci2a', choices=['bci2a', 'physionet'])
    parser.add_argument('--dataset_cfg', type=str, default='default.yaml')
    parser.add_argument('--model', type=str, default='EEGNet', choices=['EEGNet', 'EEGNetRp', 'ST_GCN', 'ASTGCN'])
    parser.add_argument('--strategy', type=str, default='cross-subject',
                        choices=['cross-subject', 'within-subject'])
    parser.add_argument('--fit_cfg', type=str, default='default.yaml')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    config = {
        'dataset': read_yaml(f"{os.getcwd()}\\config\\dataset\\{args.dataset_cfg}"),
        'fit': read_yaml(f"{os.getcwd()}\\config\\fit\\{args.fit_cfg}"),
    }
    save_dir = f"{os.getcwd()}\\save\\{args.dataset}\\{int(time.time())}\\"
    print(config)
    if args.save:
        save_config(config, save_dir)
    args.save_dir = save_dir
    if args.dataset == 'bci2a':
        bci2a(args, config)
    elif args.dataset == 'physionet':
        physionet(args, config)
