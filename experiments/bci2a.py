import dataset_process
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
from classify_models import cuda, get_eeg_net, get_shallow_conv_net
import pandas as pd
from sklearn.model_selection import GridSearchCV
import time
from datetime import datetime
moabb.set_log_level("info")


def _load_dataset():
    dataset_bd = dataset_process.DatasetFromBraindecode('bci2a', subject_ids=None)
    dataset = dataset_bd.dataset_instance
    dataset_bd.preprocess_dataset()
    windows_dataset = dataset_bd.create_windows_dataset(trial_start_offset_seconds=-0.5)
    return dataset, windows_dataset


def _base_experiment(name, windows_dataset, clf, param_grid):
    f = open(f"{time.time()}.txt", "w")
    f.write("Model: " + name + "\nTime: " + str(datetime.now()) + "\n")
    # for every subject in dataset, fit classifier and test
    subjects_windows_dataset = windows_dataset.split('subject')
    subjects_accuracy = []
    for subject, windows_dataset in subjects_windows_dataset.items():
        split_by_session = windows_dataset.split('session')
        train_set = split_by_session['session_T']
        test_set = split_by_session['session_E']
        X_train = SliceDataset(train_set, idx=0)
        y_train = np.array([y for y in SliceDataset(train_set, idx=1)])
        # use 5-fold cross-validation for training
        train_val_split = KFold(n_splits=5, shuffle=False)
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=train_val_split, return_train_score=True,
                                   scoring='accuracy', refit=True, verbose=2, error_score='raise')
        grid_search.fit(X=X_train, y=y_train)
        grid_search_result = pd.DataFrame(grid_search.cv_results_)
        best_result = grid_search_result[grid_search_result["rank_test_score"] == 1].squeeze()
        y_test = test_set.get_metadata().target
        test_accuracy = grid_search.best_estimator_.score(X=test_set, y=y_test)
        result = f"Subject {subject}:\n"\
                 f"Best hyper-parameters: {best_result['params']}\n" \
                 f"Mean validation accuracy: {best_result['mean_test_score'] * 100:.2f}%\n"\
                 f"Test accuracy: {(test_accuracy * 100):.2f}%\n"
        subjects_accuracy.append(round(test_accuracy, 5))
        f.write(result)
    result = name + " within-subject accuracy: " + str(subjects_accuracy) + "\n"
    result += f"Mean accuracy: {np.mean(subjects_accuracy) * 100:.2f}%\n"
    f.write(result)
    f.close()


# BCI2a dataset using ShallowFBCSPNet model
# reference example: https://braindecode.org/stable/auto_examples/plot_data_augmentation.html#
def bci2a_shallow_conv_net():
    set_random_seeds(seed=20233202, cuda=cuda)
    dataset, windows_dataset = _load_dataset()
    n_channels = windows_dataset[0][0].shape[0]
    input_window_samples = windows_dataset[0][0].shape[1]
    sfreq = dataset.datasets[0].raw.info['sfreq']
    transforms = get_augmentation_transform(sample_freq=sfreq)
    n_epochs = 300
    param_grid = {
        "optimizer__lr": [0.006, 0.0006],
        "batch_size": [16, 64],
        "optimizer__weight_decay": [0],
        "max_epochs": [n_epochs]
    }
    model = get_shallow_conv_net(n_channels=n_channels, n_classes=4, input_window_samples=input_window_samples)
    clf = EEGClassifier(module=model, iterator_train=AugmentedDataLoader, iterator_train__transforms=transforms,
                        criterion=torch.nn.NLLLoss, optimizer=torch.optim.AdamW, train_split=None,
                        callbacks=["accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1))],
                        device='cuda' if cuda else 'cpu'
                        )
    _base_experiment(name='ShallowConvNet', windows_dataset=windows_dataset, clf=clf, param_grid=param_grid)


def bci2a_eeg_net():
    set_random_seeds(seed=14388341, cuda=cuda)
    dataset, windows_dataset = _load_dataset()
    n_channels = windows_dataset[0][0].shape[0]
    input_window_samples = windows_dataset[0][0].shape[1]
    sfreq = dataset.datasets[0].raw.info['sfreq']
    transforms = get_augmentation_transform(sample_freq=sfreq)
    model = get_eeg_net(n_channels=n_channels, n_classes=4, input_window_samples=input_window_samples)
    n_epochs = 300
    param_grid = {
        "optimizer__lr": [0.001, 0.0005],
        "batch_size": [16, 64],
        "optimizer__weight_decay": [0],
        "max_epochs": [n_epochs]
    }
    clf = EEGClassifier(module=model, iterator_train=AugmentedDataLoader, iterator_train__transforms=transforms,
                        criterion=torch.nn.NLLLoss, optimizer=torch.optim.AdamW, train_split=None,
                        callbacks=["accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1))],
                        device='cuda' if cuda else 'cpu'
                        )
    _base_experiment(name='EEGNet', windows_dataset=windows_dataset, clf=clf, param_grid=param_grid)


def bci2a_csp_lda():
    # get dataset
    dataset_mb = dataset_process.DatasetFromMoabb('bci2a')
    datasets = [dataset_mb.dataset_instance]
    # make pipelines (feature extract & estimator)
    pipelines = {"CSP+LDA": make_pipeline(CSP(n_components=8), LDA())}
    # set paradigm
    paradigm = MotorImagery(n_classes=4)
    overwrite = False
    # evaluate and get result
    evaluation = CrossSessionEvaluation(
        paradigm=paradigm, datasets=datasets, suffix="examples", overwrite=overwrite
    )
    results = evaluation.process(pipelines)
    print(results)
