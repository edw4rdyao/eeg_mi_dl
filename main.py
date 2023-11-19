import argparse
import os
import time

from experiments.bci2a import BCI2aExperiment
from experiments.physionet import physionet
from utils import read_yaml, save_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bci2a', choices=['bci2a', 'physionet'])
    parser.add_argument('--config', type=str, default='default.yaml')
    parser.add_argument('--model', type=str, default='EEGNet', choices=['EEGNet', 'ShallowConv',
                                                                        'DeepConv', 'EEGConformer'])
    parser.add_argument('--strategy', type=str, default='within-subject',
                        choices=['cross-subject', 'within-subject'])
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    config = read_yaml(f"{os.getcwd()}\\config\\{args.config}")
    save_dir = f"{os.getcwd()}\\save\\{args.dataset}\\{int(time.time())}\\"
    print(config)
    if args.save:
        save_config(config, save_dir)
    args.save_dir = save_dir
    if args.dataset == 'bci2a':
        exp = BCI2aExperiment(args=args, config=config)
        exp.run()
    elif args.dataset == 'physionet':
        physionet(args, config)
