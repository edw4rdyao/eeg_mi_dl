import moabb
import torch
from braindecode import EEGClassifier
from braindecode.models import EEGNetv4, ShallowFBCSPNet, Deep4Net
from braindecode.util import set_random_seeds
from skorch.callbacks import LRScheduler, Checkpoint
from skorch.helper import predefined_split
from torch.utils.data import ConcatDataset
from torchinfo import summary

import dataset_loader
import nn_models
import utils
from nn_models import cuda

moabb.set_log_level("info")


def __get_subject_split():
    all_valid_subjects = []
    train_subjects = []
    test_subjects = []
    for i in range(1, 110):
        if i not in [88, 89, 92, 100]:
            all_valid_subjects.append(i)
            if i <= 84:
                train_subjects.append(i)
            else:
                test_subjects.append(i)
    return all_valid_subjects, train_subjects, test_subjects


def __get_subjects_datasets(dataset_split_by_subject, split_subject, n_classes):
    if n_classes == 2:
        valid_dataset = []
        for i in split_subject:
            for ds in dataset_split_by_subject[str(i)].datasets:
                if 'left_hand' in ds.windows.event_id or 'right_hand' in ds.windows.event_id:
                    valid_dataset.append(ds)
        split_datasets = ConcatDataset(valid_dataset)
    else:
        split_datasets = ConcatDataset([dataset_split_by_subject[str(i)] for i in split_subject])
    return split_datasets


def physionet(args, config):
    set_random_seeds(seed=config['fit']['seed'], cuda=cuda)
    all_valid_subjects, _, _ = __get_subject_split()
    ds = dataset_loader.DatasetFromBraindecode('physionet', subject_ids=all_valid_subjects)
    ds.uniform_duration(4.0)
    ds.drop_last_annotation()
    ds.preprocess(resample_freq=config['dataset']['resample'], high_freq=config['dataset']['high_freq'],
                  low_freq=config['dataset']['low_freq'], picked_channels=config['dataset']['channels'])
    channels_name = ds.get_channels_name()
    print(channels_name)
    n_classes = config['dataset']['n_classes']
    if n_classes == 3:
        events_mapping = {
            'left_hand': 0,
            'right_hand': 1,
            'feet': 2
        }
    else:
        events_mapping = {
            'left_hand': 0,
            'right_hand': 1,
            'feet': 2,
            'hands': 3
        }
    windows_dataset = ds.create_windows_dataset(trial_start_offset_seconds=-1, trial_stop_offset_seconds=1,
                                                mapping=events_mapping)
    n_channels = ds.get_channel_num()
    input_window_samples = ds.get_input_window_sample()
    if args.model == 'EEGNet':
        model = EEGNetv4(in_chans=n_channels, n_classes=n_classes,
                         input_window_samples=input_window_samples, kernel_length=32, drop_prob=0.5)
    elif args.model == 'ShallowConv':
        model = ShallowFBCSPNet(in_chans=n_channels, n_classes=n_classes,
                                input_window_samples=input_window_samples, final_conv_length='auto')
    elif args.model == 'DeepConv':
        model = Deep4Net(in_chans=n_channels, n_classes=n_classes,
                         input_window_samples=input_window_samples, final_conv_length='auto')
    elif args.model == 'ASTGCN':
        model = nn_models.ASTGCN(n_channels=n_channels, n_classes=4, input_window_size=input_window_samples,
                                 kernel_length=32)
    elif args.model == 'BaseCNN':
        model = nn_models.BaseCNN(n_channels=n_channels, n_classes=n_classes, input_window_size=input_window_samples)
    else:
        raise ValueError(f"model {args.model} is not supported on this dataset.")

    if cuda:
        model.cuda()
    summary(model, (1, n_channels, input_window_samples, 1))

    n_epochs = config['fit']['epochs']
    lr = config['fit']['lr']
    batch_size = config['fit']['batch_size']
    callbacks = [("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1))]
    if args.save:
        callbacks.append(Checkpoint(monitor='valid_acc_best', dirname=args.save_dir,
                                    f_params='{last_epoch[valid_accuracy]}.pt'))
    if args.selection:
        callbacks.append(("get_electrode_importance", utils.GetElectrodeImportance()))
    clf = EEGClassifier(module=model,
                        iterator_train__shuffle=True,
                        criterion=torch.nn.CrossEntropyLoss,
                        optimizer=torch.optim.Adam,
                        train_split=None,
                        optimizer__lr=lr,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        device='cuda' if cuda else 'cpu'
                        )
    dataset_split_by_subject = windows_dataset.split('subject')
    _, train_subjects, test_subjects = __get_subject_split()
    train_set = __get_subjects_datasets(dataset_split_by_subject, train_subjects, n_classes)
    test_set = __get_subjects_datasets(dataset_split_by_subject, test_subjects, n_classes)
    clf.train_split = predefined_split(test_set)
    clf.fit(X=train_set, y=None, epochs=n_epochs)
