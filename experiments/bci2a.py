import dataset_process
from braindecode import EEGClassifier
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet
import torch
from braindecode.augmentation import FrequencyShift
from braindecode.augmentation import AugmentedDataLoader, SignFlip
from skorch.helper import predefined_split, SliceDataset
from skorch.callbacks import LRScheduler
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

# bci 2a dataset using ShallowFBCSPNet model
# ref: https://braindecode.org/stable/auto_examples/plot_data_augmentation.html#
def bci2a_shallow_fbcspnet():
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
    model = ShallowFBCSPNet(
        n_channels,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length='auto',
    )
    if cuda:
        model.cuda()

    # transforms of data augmentation
    sfreq = dataset.datasets[0].raw.info['sfreq']
    freq_shift = FrequencyShift(
        probability=.5,
        sfreq=sfreq,
        max_delta_freq=2.
    )
    sign_flip = SignFlip(probability=.1)
    transforms = [
        freq_shift,
        sign_flip
    ]

    # EEG classifier
    lr = 0.0625 * 0.01
    weight_decay = 0
    batch_size = 64
    n_epochs = 200
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

    # for every subject in dataset
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
        acc.append(test_acc)
    # [0.6458333333333334, 0.3854166666666667, 0.7430555555555556, 0.5798611111111112, 0.375, 0.5069444444444444,
    #  0.7465277777777778, 0.7083333333333334, 0.7361111111111112]

    # KFold cross validation
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

