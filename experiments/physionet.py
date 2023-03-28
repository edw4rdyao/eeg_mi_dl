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

moabb.set_log_level("info")


def _within_subject_experiment(model_name, windows_dataset, clf, n_epochs):
    f = open(f"./log/{model_name}-{time.time()}.txt", "w")
    f.write("Model: " + model_name + "\nTime: " + str(datetime.now()) + "\n")
    # for every subject in dataset, fit classifier and test
    subjects_windows_dataset = windows_dataset.split('subject')
    subjects_accuracy = []
    for subject, windows_dataset in subjects_windows_dataset.items():
        split_by_session = windows_dataset.split('session')
        train_set = split_by_session['session_T']
        test_set = split_by_session['session_E']
        clf.train_split = predefined_split(test_set)
        clf.fit(train_set, y=None, epochs=n_epochs)
        y_test = test_set.get_metadata().target
        test_accuracy = clf.score(test_set, y=y_test)
        out = f"Subject{subject} test accuracy: " + str(round(test_accuracy, 5)) + "\n"
        print(out)
        f.write(out)
        subjects_accuracy.append(round(test_accuracy, 5))
    result = model_name + " within-subject accuracy: " + str(subjects_accuracy) + "\n"
    result += f"Mean accuracy: {np.mean(subjects_accuracy) * 100:.2f}%\n"
    f.write(result)
    f.close()


def _cross_subject_experiment(model_name, windows_dataset, clf, n_epochs):
    f = open(f"./log/{model_name}-{time.time()}.txt", "w")
    f.write("Model: " + model_name + "\nTime: " + str(datetime.now()) + "\n")
    # for every subject in dataset, fit classifier and test
    split_by_session = windows_dataset.split('session')
    train_set = split_by_session['session_T']
    test_set = split_by_session['session_E']
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
    ds = dataset_loader.DatasetFromBraindecode('physionet', subject_ids=[1,2,3])
    ds.preprocess_dataset(low_freq=4, high_freq=38)
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
    model = nn_models.EEGNetv4(in_chans=n_channels, n_classes=2, input_window_samples=input_window_samples,
                               kernel_length=32, drop_prob=0.5)
    if cuda:
        model.cuda()
    summary(model, (1, n_channels, input_window_samples, 1))
    n_epochs = 750
    lr = 0.001
    weight_decay = 0
    batch_size = 64
    clf = EEGClassifier(module=model,
                        criterion=torch.nn.CrossEntropyLoss, optimizer=torch.optim.AdamW, train_split=None,
                        optimizer__lr=lr, optimizer__weight_decay=weight_decay, batch_size=batch_size,
                        callbacks=["accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1))],
                        device='cuda' if cuda else 'cpu'
                        )
    # _cross_subject_experiment(model_name='EEGNet', windows_dataset=windows_dataset, clf=clf, n_epochs=n_epochs)
    _within_subject_experiment(model_name='EEGNet', windows_dataset=windows_dataset, clf=clf, n_epochs=n_epochs)
