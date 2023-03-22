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
from nn_models import cuda, get_eeg_net, get_shallow_conv_net
import pandas as pd
from sklearn.model_selection import GridSearchCV
from braindecode.preprocessing import exponential_moving_standardize, preprocess, Preprocessor
from numpy import multiply
import time
from datetime import datetime
from braindecode.models import to_dense_prediction_model, get_output_shape
from braindecode.training import CroppedLoss
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


def bci2a_shallow_conv_net():
    set_random_seeds(seed=20233202, cuda=cuda)
    ds = dataset_loader.DatasetFromBraindecode('bci2a', subject_ids=None)
    preprocess(ds.raw_dataset, [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),
        Preprocessor(lambda data: multiply(data, 1e6)),
        Preprocessor('filter', l_freq=4, h_freq=38),
        Preprocessor(exponential_moving_standardize,
                     factor_new=1e-3, init_block_size=1000)
    ])
    n_channels = ds.get_channel_num()
    input_window_samples = 1000
    model = get_shallow_conv_net(n_channels=n_channels, n_classes=4, input_window_samples=input_window_samples,
                                 final_conv_length=30, drop_prob=0.25)
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
                        train_split=None, criterion=CroppedLoss, criterion__loss_function=torch.nn.functional.nll_loss,
                        optimizer=torch.optim.AdamW, optimizer__lr=lr,
                        optimizer__weight_decay=weight_decay, batch_size=batch_size,
                        callbacks=["accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1))],
                        cropped=True, device='cuda' if cuda else 'cpu'
                        )
    _within_subject_experiment(model_name='ShallowConvNet', windows_dataset=windows_dataset, clf=clf, n_epochs=n_epochs)


def bci2a_eeg_net():
    set_random_seeds(seed=14388341, cuda=cuda)
    ds = dataset_loader.DatasetFromBraindecode('bci2a', subject_ids=None)
    preprocess(ds.raw_dataset, [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),
        Preprocessor(lambda data: multiply(data, 1e6)),
        # Preprocessor('resample', sfreq=128),
        Preprocessor('filter', l_freq=4, h_freq=40),
        Preprocessor(exponential_moving_standardize,
                     factor_new=1e-3, init_block_size=1000)
    ])
    windows_dataset = ds.create_windows_dataset(trial_start_offset_seconds=0.5, trial_stop_offset_seconds=-1.5)
    n_channels = ds.get_channel_num()
    input_window_samples = windows_dataset[0][0].shape[1]
    model = get_eeg_net(n_channels=n_channels, n_classes=4, input_window_samples=input_window_samples,
                        kernel_length=64, drop_prob=0.5)
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
    _within_subject_experiment(model_name='EEGNet', windows_dataset=windows_dataset, clf=clf, n_epochs=n_epochs)


def bci2a_csp_lda():
    ds = dataset_loader.DatasetFromMoabb('bci2a')
    datasets = [ds.dataset_instance]
    pipelines = {"CSP+LDA": make_pipeline(CSP(n_components=8), LDA())}
    paradigm = MotorImagery(n_classes=4)
    overwrite = False
    evaluation = CrossSessionEvaluation(
        paradigm=paradigm, datasets=datasets, suffix="examples", overwrite=overwrite
    )
    results = evaluation.process(pipelines)
    print(results)
