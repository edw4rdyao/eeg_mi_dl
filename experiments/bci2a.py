import moabb
import torch
from braindecode import EEGClassifier
from braindecode.augmentation import AugmentedDataLoader
from braindecode.models import EEGNetv4, ShallowFBCSPNet
from braindecode.models import to_dense_prediction_model, get_output_shape
from braindecode.training import CroppedLoss
from braindecode.util import set_random_seeds
from skorch.callbacks import LRScheduler, Checkpoint
from skorch.helper import predefined_split
from torch.nn import functional
from torch.utils.data import ConcatDataset
from torchinfo import summary

import dataset_loader
from nn_models import cuda
from utils import get_augmentation_transform

moabb.set_log_level("info")


def bci2a(args, config):
    set_random_seeds(seed=config['fit']['seed'], cuda=cuda)
    ds = dataset_loader.DatasetFromBraindecode('bci2a', subject_ids=None)
    ds.preprocess_dataset(resample_freq=config['dataset']['resample'], high_freq=config['dataset']['high_freq'],
                          low_freq=config['dataset']['low_freq'])
    windows_dataset = ds.create_windows_dataset(trial_start_offset_seconds=-0.5)
    n_channels = ds.get_channel_num()
    input_window_samples = ds.get_input_window_sample()
    n_classes = config['dataset']['n_classes']
    if args.model == 'EEGNet':
        model = EEGNetv4(in_chans=n_channels, n_classes=n_classes,
                         input_window_samples=input_window_samples, kernel_length=32, drop_prob=0.5)
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
        callbacks.append(Checkpoint(monitor="valid_accuracy_best", dirname=args.save_dir))

    clf = EEGClassifier(module=model,
                        criterion=torch.nn.CrossEntropyLoss,
                        optimizer=torch.optim.Adam,
                        train_split=None,
                        optimizer__lr=lr,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        device='cuda' if cuda else 'cpu'
                        )
    if args.strategy == 'within-subject':
        __within_subject_experiment(windows_dataset=windows_dataset, clf=clf, n_epochs=n_epochs)
    elif args.strategy == 'cross-subject':
        __cross_subject_experiment(windows_dataset=windows_dataset, clf=clf, n_epochs=n_epochs)
    else:
        raise ValueError(f"strategy {args.model} is not supported on this dataset.")


def __within_subject_experiment(windows_dataset, clf, n_epochs):
    subjects_windows_dataset = windows_dataset.split('subject')
    for subject, windows_dataset in subjects_windows_dataset.items():
        split_by_session = windows_dataset.split('session')
        train_set = split_by_session['session_T']
        test_set = split_by_session['session_E']
        clf.train_split = predefined_split(test_set)
        clf.fit(train_set, y=None, epochs=n_epochs)


def __cross_subject_experiment(windows_dataset, clf, n_epochs):
    split_by_subject = windows_dataset.split('subject')
    train_subjects = ['1', '2', '3', '4', '5', '6', '7', '8']
    test_subjects = ['9']
    train_set = ConcatDataset([split_by_subject[i] for i in train_subjects])
    test_set = ConcatDataset([split_by_subject[i] for i in test_subjects])
    clf.train_split = predefined_split(test_set)
    clf.fit(train_set, y=None, epochs=n_epochs)


def bci2a_shallow_conv_net():
    set_random_seeds(seed=20233202, cuda=cuda)
    ds = dataset_loader.DatasetFromBraindecode('bci2a', subject_ids=None)
    ds.preprocess_dataset(low_freq=4, high_freq=38)
    n_channels = ds.get_channel_num()
    input_window_samples = 1000
    model = ShallowFBCSPNet(in_chans=n_channels, n_classes=4, input_window_samples=input_window_samples,
                            final_conv_length=30, drop_prob=0.25)
    if cuda:
        model.cuda()
    summary(model, (1, n_channels, input_window_samples, 1))
    # for cropped training
    to_dense_prediction_model(model)
    n_preds_per_input = get_output_shape(model, n_channels, input_window_samples)[2]
    windows_dataset = ds.create_windows_dataset(trial_start_offset_seconds=-0.5,
                                                window_size_samples=input_window_samples,
                                                window_stride_samples=n_preds_per_input)
    transforms = get_augmentation_transform(sample_freq=ds.get_sample_freq())
    n_epochs = 300
    lr = 0.000625
    weight_decay = 0
    batch_size = 64
    clf = EEGClassifier(module=model, iterator_train=AugmentedDataLoader, iterator_train__transforms=transforms,
                        train_split=None, criterion=CroppedLoss, criterion__loss_function=functional.nll_loss,
                        optimizer=torch.optim.AdamW, optimizer__lr=lr,
                        optimizer__weight_decay=weight_decay, batch_size=batch_size,
                        callbacks=["accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1))],
                        cropped=True, device='cuda' if cuda else 'cpu'
                        )
    __within_subject_experiment(windows_dataset=windows_dataset, clf=clf, n_epochs=n_epochs)
