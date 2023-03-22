import dataset_loader
from braindecode import EEGClassifier
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet, EEGNetv4
import torch
from braindecode.augmentation import AugmentedDataLoader, SignFlip, FrequencyShift
from skorch.helper import predefined_split, SliceDataset
from skorch.callbacks import LRScheduler
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import moabb
from moabb.evaluations import CrossSessionEvaluation, WithinSessionEvaluation, CrossSubjectEvaluation
from moabb.paradigms import MotorImagery
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from utils import get_augmentation_transform
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from moabb.datasets import PhysionetMI
moabb.set_log_level("info")


def _load_dataset():
    ds = dataset_loader.DatasetFromBraindecode('physionet', subject_ids=list(range(1, 21)))
    dataset = ds.raw_dataset
    windows_dataset = ds.create_windows_dataset(trial_start_offset_seconds=0,
                                                trial_stop_offset_seconds=-1,
                                                mapping={
                                                    'left_hand': 0,
                                                    'right_hand': 1,
                                                    'hands': 2,
                                                    'feet': 3
                                                })
    return dataset, windows_dataset


def physionet_eeg_net():
    dataset, windows_dataset = _load_dataset()
    cuda = torch.cuda.is_available()
    set_random_seeds(seed=14381438, cuda=cuda)
    device = 'cuda' if cuda else 'cpu'
    n_channels = windows_dataset[0][0].shape[0]
    input_window_samples = windows_dataset[0][0].shape[1]
    model = EEGNetv4(in_chans=n_channels, n_classes=4, input_window_samples=input_window_samples,
                     final_conv_length='auto')
    if cuda:
        model.cuda()
    sfreq = dataset.datasets[0].raw.info['sfreq']
    transforms = get_augmentation_transform(sample_freq=sfreq)
    lr = 0.000625
    weight_decay = 0
    batch_size = 16
    n_epochs = 250
    clf = EEGClassifier(
        model,
        iterator_train=AugmentedDataLoader,
        iterator_train__transforms=transforms,
        criterion=torch.nn.NLLLoss,
        optimizer=torch.optim.AdamW,
        train_split=None,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        callbacks=[
            "accuracy",
            ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
        ],
        device=device,
    )
    subjects_windows_dataset = windows_dataset.split('subject')
    train_val_split = KFold(n_splits=5, shuffle=True)
    acc = []
    for sbj, sbj_dataset in subjects_windows_dataset.items():
        fit_params = {'epochs': n_epochs}
        X_train = SliceDataset(sbj_dataset, idx=0)
        y_train = np.array([y for y in SliceDataset(sbj_dataset, idx=1)])
        result = cross_val_score(clf, X=X_train, y=y_train, cv=train_val_split, fit_params=fit_params)
        print(f"Subject {sbj}", f"average validation accuracy: {np.mean(result):.5f}")
        acc.append(np.mean(result))
    print(acc)


def physionet_shallow_conv_net():
    dataset, windows_dataset = _load_dataset()
    cuda = torch.cuda.is_available()
    seed = 20202020
    set_random_seeds(seed=seed, cuda=cuda)
    device = 'cuda' if cuda else 'cpu'
    n_channels = windows_dataset[0][0].shape[0]
    input_window_samples = windows_dataset[0][0].shape[1]
    model = ShallowFBCSPNet(n_channels, n_classes=4, input_window_samples=input_window_samples,
                            final_conv_length='auto')
    if cuda:
        model.cuda()
    sfreq = dataset.datasets[0].raw.info['sfreq']
    transforms = get_augmentation_transform(sample_freq=sfreq)
    lr = 0.000625
    weight_decay = 0
    batch_size = 64
    n_epochs = 250
    clf = EEGClassifier(
        model,
        iterator_train=AugmentedDataLoader,
        iterator_train__transforms=transforms,
        criterion=torch.nn.NLLLoss,
        optimizer=torch.optim.AdamW,
        train_split=None,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        callbacks=[
            "accuracy",
            ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
        ],
        device=device,
    )
    subjects_windows_dataset = windows_dataset.split('subject')
    train_val_split = KFold(n_splits=5, shuffle=True)
    acc = []
    for sbj, sbj_dataset in subjects_windows_dataset.items():
        fit_params = {'epochs': n_epochs}
        X_train = SliceDataset(sbj_dataset, idx=0)
        y_train = np.array([y for y in SliceDataset(sbj_dataset, idx=1)])
        result = cross_val_score(clf, X=X_train, y=y_train, cv=train_val_split, fit_params=fit_params)
        print(f"Subject {sbj}", f"average validation accuracy: {np.mean(result):.5f}")
        acc.append(np.mean(result))
    print(acc)
