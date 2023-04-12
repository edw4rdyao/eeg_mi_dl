import dataset_loader
from braindecode import EEGClassifier
from braindecode.util import set_random_seeds
import torch
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler, Checkpoint
from sklearn.model_selection import KFold, cross_val_score
import moabb
import nn_models
from nn_models import cuda
from torchinfo import summary
from torch.utils.data import ConcatDataset, DataLoader
moabb.set_log_level("info")


def _get_subject_split():
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


def _cross_subject_experiment(windows_dataset, clf, n_epochs):
    _, train_subjects, test_subjects = _get_subject_split()
    split_by_subject = windows_dataset.split('subject')
    train_set = ConcatDataset([split_by_subject[str(i)] for i in train_subjects])
    test_set = ConcatDataset([split_by_subject[str(i)] for i in test_subjects])
    clf.train_split = predefined_split(test_set)
    clf.fit(X=train_set, y=None, epochs=n_epochs)


def physionet(args, config):
    set_random_seeds(seed=20233202, cuda=cuda)
    all_valid_subjects, _, _ = _get_subject_split()
    ds = dataset_loader.DatasetFromBraindecode('physionet', subject_ids=all_valid_subjects)
    ds.uniform_duration(4.0)
    ds.drop_last_annotation()
    ds.preprocess_dataset(resample_freq=config['dataset']['resample'], high_freq=config['dataset']['high_freq'],
                          low_freq=config['dataset']['low_freq'])
    windows_dataset = ds.create_windows_dataset(trial_start_offset_seconds=-1,
                                                trial_stop_offset_seconds=1,
                                                mapping={
                                                    'left_hand': 0,
                                                    'right_hand': 1,
                                                    'hands': 2,
                                                    'feet': 3
                                                })
    n_channels = ds.get_channel_num()
    input_window_samples = ds.get_input_window_sample()
    n_classes = config['dataset']['n_classes']
    if args.model == 'EEGNet':
        model = nn_models.EEGNetv4(in_chans=n_channels, n_classes=n_classes,
                                   input_window_samples=input_window_samples, kernel_length=64, drop_prob=0.5)
    elif args.model == 'ST_GCN':
        model = nn_models.ST_GCN(n_channels=n_channels, n_classes=n_classes, input_window_size=input_window_samples,
                                 kernel_length=32, drop_prob=0.5)
    elif args.model == 'ASTGCN':
        model = nn_models.ASTGCN(n_channels=n_channels, n_classes=4, input_window_size=input_window_samples,
                                 kernel_length=32)
    else:
        raise ValueError(f"model {args.model} is not supported on this dataset.")

    if cuda:
        model.cuda()
    summary(model, (1, n_channels, input_window_samples, 1))

    n_epochs = config['fit']['epochs']
    lr = config['fit']['lr']
    batch_size = config['fit']['batch_size']
    callbacks = ["accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1))]
    if args.save:
        callbacks.append(Checkpoint(dirname=args.save_dir))
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
    _cross_subject_experiment(windows_dataset=windows_dataset, clf=clf, n_epochs=n_epochs)
