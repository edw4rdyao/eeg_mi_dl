from braindecode.augmentation import SignFlip, FrequencyShift
import torch


def get_augmentation_transform(sample_freq):
    freq_shift = FrequencyShift(
        probability=.5,
        sfreq=sample_freq,
        max_delta_freq=2.
    )
    sign_flip = SignFlip(probability=.1)
    transforms = [
        freq_shift,
        sign_flip
    ]
    return transforms


def get_adjacency_matrix(n_electrodes, mode):
    adj = torch.zeros(n_electrodes, n_electrodes)
    if mode == 'full':
        adj = torch.ones(n_electrodes, n_electrodes)
    return adj
