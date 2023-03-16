"""
Dataset from braindecode lib. https://braindecode.org/stable/index.html
Encapsulate the operations
"""
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import exponential_moving_standardize, preprocess, Preprocessor
from numpy import multiply
from braindecode.preprocessing import create_windows_from_events


def _load_bci2a_braindecode(subject_ids):
    dataset_description = "Dataset IIa from BCI Competition 4"
    dataset_instance = MOABBDataset(dataset_name="BNCI2014001", subject_ids=subject_ids)
    return dataset_instance, dataset_description


def _load_physionet_braindecode(subject_ids):
    dataset_description = "Physionet MI dataset: https://physionet.org/pn4/eegmmidb/"
    dataset_instance = MOABBDataset(dataset_name="PhysionetMI", subject_ids=subject_ids)
    return dataset_instance, dataset_description


class DatasetFromBraindecode:
    def __init__(self, dataset_name, subject_ids):
        if dataset_name == 'bci2a':
            self.dataset_instance, self.dataset_description = _load_bci2a_braindecode(subject_ids)
        elif dataset_name == 'physionet':
            self.dataset_instance, self.dataset_description = _load_physionet_braindecode(subject_ids)
        else:
            raise ValueError(
                "dataset:%s is not supported" % dataset_name
            )

    def preprocess_dataset(self, low_flt_freq=4, high_flt_freq=38, factor_new=1e-3, init_block_size=1000, factor=1e6):
        preprocessors = [
            Preprocessor('pick_types', eeg=True, meg=False, stim=False),
            Preprocessor(lambda data: multiply(data, factor)),
            Preprocessor('filter', l_freq=low_flt_freq, h_freq=high_flt_freq),
            Preprocessor(exponential_moving_standardize,
                         factor_new=factor_new, init_block_size=init_block_size)
        ]
        preprocess(self.dataset_instance, preprocessors)

    def create_windows_dataset(self, trial_start_offset_seconds=0, trial_stop_offset_seconds=0, mapping=None):
        # Extract sampling frequency, check that they are same in all datasets
        sfreq = self.dataset_instance.datasets[0].raw.info['sfreq']
        assert all([ds.raw.info['sfreq'] == sfreq for ds in self.dataset_instance.datasets])
        # Calculate the trial start offset in samples.
        trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)
        trial_stop_offset_samples = int(trial_stop_offset_seconds * sfreq)
        # Create windows using braindecode function for this. It needs parameters to
        # define how trials should be used.
        windows_dataset = create_windows_from_events(
            self.dataset_instance,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=trial_stop_offset_samples,
            preload=True,
            mapping=mapping,
        )
        return windows_dataset
