import dataset_loader
from braindecode import EEGClassifier
from braindecode.util import set_random_seeds
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
import nn_models
from nn_models import cuda
import time
from datetime import datetime
from braindecode.models import to_dense_prediction_model, get_output_shape
from braindecode.training import CroppedLoss
from torchinfo import summary
from torch.utils.data import ConcatDataset

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


def _cross_subject_experiment(model_name, windows_dataset, clf, n_epochs):
    f = open(f"./log/{model_name}-{time.time()}.txt", "w")
    f.write("Model: " + model_name + "\nTime: " + str(datetime.now()) + "\n")
    # for every subject in dataset, fit classifier and test
    _, train_subjects, test_subjects = _get_subject_split()
    split_by_subject = windows_dataset.split('subject')
    train_set = ConcatDataset([split_by_subject[str(i)] for i in train_subjects])
    test_set = ConcatDataset([split_by_subject[str(i)] for i in test_subjects])
    clf.train_split = predefined_split(test_set)
    clf.fit(train_set, y=None, epochs=n_epochs)
    y_test = test_set.get_metadata().target
    test_accuracy = clf.score(test_set, y=y_test)
    out = f"Test accuracy: " + str(round(test_accuracy, 5)) + "\n"
    print(out)
    f.write(out)
    f.close()


def physionet_eeg_net():
    set_random_seeds(seed=14388341, cuda=cuda)
    all_valid_subjects, _, _ = _get_subject_split()
    ds = dataset_loader.DatasetFromBraindecode('physionet', subject_ids=[1, 24, 32, 108])
    ds.uniform_duration(4.0)
    ds.preprocess_dataset()
    windows_dataset = ds.create_windows_dataset(trial_start_offset_seconds=0,
                                                trial_stop_offset_seconds=-1,
                                                mapping={
                                                    'left_hand': 0,
                                                    'right_hand': 1,
                                                    'hands': 2,
                                                    'feet': 3
                                                })
    n_channels = ds.get_channel_num()
    input_window_samples = ds.get_input_window_sample()
    # model = nn_models.EEGNetv4(in_chans=n_channels, n_classes=4, input_window_samples=input_window_samples,
    #                            kernel_length=64, drop_prob=0.5)
    # model = nn_models.EEGNetGCN(n_channels=n_channels, n_classes=4, input_window_size=input_window_samples,
    #                             kernel_length=64)
    # model = nn_models.ST_GCN(n_channels=n_channels, n_classes=4, input_window_size=input_window_samples,
    #                          kernel_length=64)
    model = nn_models.ASTGCN(n_channels=n_channels, n_classes=4, input_window_size=input_window_samples,
                             kernel_length=64)
    if cuda:
        model.cuda()
    summary(model, (1, n_channels, input_window_samples, 1))
    n_epochs = 750
    lr = 0.001
    weight_decay = 1e-8
    batch_size = 32
    clf = EEGClassifier(module=model,
                        criterion=torch.nn.CrossEntropyLoss, optimizer=torch.optim.AdamW, train_split=None,
                        optimizer__lr=lr, optimizer__weight_decay=weight_decay, batch_size=batch_size,
                        callbacks=["accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1))],
                        device='cuda' if cuda else 'cpu'
                        )
    _cross_subject_experiment(model_name='EEGNet', windows_dataset=windows_dataset, clf=clf, n_epochs=n_epochs)
