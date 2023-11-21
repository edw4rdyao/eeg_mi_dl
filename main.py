import argparse
import os
import time

from experiments.bci2a import BCI2aExperiment
from experiments.physionet import physionet
from utils import read_yaml, save_json2file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bci2a', choices=['bci2a', 'physionet'])
    parser.add_argument('--model', type=str, default='EEGNet', choices=['EEGNet', 'ShallowConv',
                                                                        'DeepConv', 'EEGConformer',
                                                                        'ATCNet', 'EEGInception',
                                                                        'EEGITNet'])
    parser.add_argument('--config', type=str, default='default')
    parser.add_argument('--strategy', type=str, default='within-subject',
                        choices=['cross-subject', 'within-subject'])
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    # suit default config for specific dataset and model
    if args.config == 'default':
        args.config = f'{args.dataset}_{args.model}_default.yaml'
    # read config from json file
    config = read_yaml(f"{os.getcwd()}\\config\\{args.config}")
    # result save directory
    save_dir = f"{os.getcwd()}\\save\\{args.dataset}\\{int(time.time())}_{args.dataset}_{args.model}\\"
    args.save_dir = save_dir
    print(config)
    if args.save:
        save_json2file(config, save_dir, f'{args.dataset}_{args.model}_config.json')
    # for every dataset
    if args.dataset == 'bci2a':
        exp = BCI2aExperiment(args=args, config=config)
        exp.run()
    elif args.dataset == 'physionet':
        physionet(args, config)
