import dataset_process
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
moabb.set_log_level("info")


# bci 2a dataset using ShallowFBCSPNet model
# reference example: https://braindecode.org/stable/auto_examples/plot_data_augmentation.html#
def bci2a_shallow_conv_net():
    # get dataset
    dataset_bd = dataset_process.DatasetFromBraindecode('bci2a', subject_ids=None)
    dataset = dataset_bd.dataset_instance
    # preprocess the data
    dataset_bd.preprocess_data()
    # get windows data(segmenting Epochs)
    windows_dataset = dataset_bd.create_windows_dataset(trial_start_offset_seconds=-0.5)
    # cuda setting
    cuda = torch.cuda.is_available()
    # Set random seed to be able to roughly reproduce results
    seed = 20200220
    set_random_seeds(seed=seed, cuda=cuda)
    device = 'cuda' if cuda else 'cpu'
    # initial nn model
    # extract number of chans and time steps from dataset
    n_channels = windows_dataset[0][0].shape[0]
    input_window_samples = windows_dataset[0][0].shape[1]
    n_classes = 4
    model = ShallowFBCSPNet(n_channels, n_classes, input_window_samples=input_window_samples, final_conv_length='auto')
    if cuda:
        model.cuda()
    # transforms of data augmentation
    sfreq = dataset.datasets[0].raw.info['sfreq']
    transforms = get_augmentation_transform(sample_freq=sfreq)
    # EEG classifier
    lr = 0.0625 * 0.01
    weight_decay = 0
    batch_size = 64
    n_epochs = 500
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
        device=device
    )
    # for every subject in dataset, fit classifier and test
    subjects_windows_dataset = windows_dataset.split('subject')
    acc = []
    for subject, windows_dataset in subjects_windows_dataset.items():
        # split data for test, valid and test
        splitted = windows_dataset.split('session')
        train_set = splitted['session_T']
        test_set = splitted['session_E']
        # train-test split
        clf.fit(train_set, y=None, epochs=n_epochs)
        # test model
        y_test = test_set.get_metadata().target
        test_acc = clf.score(test_set, y=y_test)
        print(f"Subject{subject} Test acc: {(test_acc * 100):.2f}%")
        acc.append(round(test_acc, 3))
    print("ShallowConvNet within subject acc:", acc)
    print(f"Average acc:{np.mean(acc):.3f}")
    # KFold cross validation to adjust hyperparameters
    # train_val_split = KFold(n_splits=5, shuffle=False)
    # X_train = SliceDataset(train_set, idx=0)
    # y_train = np.array([y for y in SliceDataset(train_set, idx=1)])
    # fit_params = {"epochs": n_epochs}
    # cv_results = cross_val_score(
    #     clf, X_train, y_train, scoring="accuracy", cv=train_val_split, fit_params=fit_params
    # )
    # print(
    #     f"Validation accuracy: {np.mean(cv_results * 100):.2f}"
    #     f"+-{np.std(cv_results * 100):.2f}%"
    # )


def bci2a_eeg_net():
    dataset_bd = dataset_process.DatasetFromBraindecode('bci2a', subject_ids=None)
    dataset = dataset_bd.dataset_instance
    dataset_bd.preprocess_data()
    windows_dataset = dataset_bd.create_windows_dataset(trial_start_offset_seconds=-0.5)
    cuda = torch.cuda.is_available()
    seed = 20200220
    set_random_seeds(seed=seed, cuda=cuda)
    device = 'cuda' if cuda else 'cpu'
    n_channels = windows_dataset[0][0].shape[0]
    input_window_samples = windows_dataset[0][0].shape[1]
    model = EEGNetv4(in_chans=n_channels, n_classes=4, input_window_samples=input_window_samples,
                     final_conv_length='auto')
    if cuda:
        model.cuda()
    sfreq = dataset.datasets[0].raw.info['sfreq']
    transforms = get_augmentation_transform(sample_freq=sfreq)
    # EEG classifier
    lr = 0.0625 * 0.01
    weight_decay = 0
    batch_size = 64
    n_epochs = 500
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
    # for every subject in dataset, fit classifier and test
    subjects_windows_dataset = windows_dataset.split('subject')
    acc = []
    for subject, windows_dataset in subjects_windows_dataset.items():
        # split data for test, valid and test
        splitted = windows_dataset.split('session')
        train_set = splitted['session_T']
        test_set = splitted['session_E']
        # train-test split
        clf.fit(train_set, y=None, epochs=n_epochs)
        # test model
        y_test = test_set.get_metadata().target
        test_acc = clf.score(test_set, y=y_test)
        print(f"Subject{subject} Test acc: {(test_acc * 100):.2f}%")
        acc.append(round(test_acc, 3))
    print("EEGNet within subject acc:", acc)
    print(f"Average acc:{np.mean(acc):.3f}")


def bci2a_csp_lda():
    # get dataset
    dataset_mb = dataset_process.DatasetFromMoabb('bci2a')
    dataset = dataset_mb.dataset_instance
    datasets = [dataset]
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
