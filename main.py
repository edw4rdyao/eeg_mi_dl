import argparse
import os
import time

from experiments.bci2a import BCI2aExperiment
from experiments.physionet import physionet
from utils import read_yaml, save_json2file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bci2a', choices=['bci2a', 'physionet'],
                        help='data set used of the experiments')
    parser.add_argument('--model', type=str, default='EEGNet',
                        choices=['EEGNet', 'EEGConformer', 'ATCNet', 'EEGInception', 'EEGITNet'],
                        help='model used of the experiments')
    parser.add_argument('--config', type=str, default='default', help='config file name(.yaml format)')
    parser.add_argument('--strategy', type=str, default='within-subject', choices=['cross-subject', 'within-subject'],
                        help='experiments strategy on subjects')
    parser.add_argument('--save', action='store_true', help='save the pytorch model and history')
    args = parser.parse_args()
    # suit default config for specific dataset and model
    if args.config == 'default':
        args.config = f'{args.dataset}_{args.model}_{args.config}.yaml'
    # read config from yaml file
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
        raise Warning('physionet experiments are developing.')
        # physionet(args, config)
