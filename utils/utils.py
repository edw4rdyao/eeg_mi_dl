from braindecode.augmentation import SignFlip, FrequencyShift


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
