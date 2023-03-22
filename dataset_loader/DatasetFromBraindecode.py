"""
Dataset based on braindecode lib. https://braindecode.org/stable/index.html
Encapsulate the operations
"""
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events
from braindecode.preprocessing import exponential_moving_standardize, preprocess, Preprocessor
from numpy import multiply


def _load_bci2a_braindecode(subject_ids):
    dataset_description = "Dataset IIa from BCI Competition IV: https://www.bbci.de/competition/iv/"
    dataset_instance = MOABBDataset(dataset_name="BNCI2014001", subject_ids=subject_ids)
    return dataset_instance, dataset_description


def _load_physionet_braindecode(subject_ids):
    dataset_description = "Physionet MI dataset: https://physionet.org/pn4/eegmmidb/"
    dataset_instance = MOABBDataset(dataset_name="PhysionetMI", subject_ids=subject_ids)
    return dataset_instance, dataset_description


class DatasetFromBraindecode:
    """

    """
    def __init__(self, dataset_name, subject_ids):
        """get dataset instance from braindecode lib

        Parameters
        ----------
        dataset_name : str
            dataset name
        subject_ids : list | None
            dataset subjects list
        """
        self.windows_dataset = None
        self.dataset_name = dataset_name
        if dataset_name == 'bci2a':
            self.raw_dataset, self.description = _load_bci2a_braindecode(subject_ids)
        elif dataset_name == 'physionet':
            self.raw_dataset, self.description = _load_physionet_braindecode(subject_ids)
        else:
            raise ValueError(
                "dataset:%s is not supported" % dataset_name
            )

    def get_sample_freq(self):
        """get the sample frequency of dataset

        Returns
        -------
        sfreq: int
            the sample frequency of raw dataset, which is got from
            mne.io.Raw object of dataset
        """
        return self.raw_dataset.datasets[0].raw.info['sfreq']

    def get_channel_num(self):
        """get channel nums of dataset

        Returns
        -------
        n_channel: int
            the channel num of dataset from raw dataset
        """
        return self.raw_dataset[0][0].shape[0]

    def get_input_window_sample(self):
        """

        Returns
        -------
        input_window_sample: int

        """
        if not self.windows_dataset:
            raise ValueError(
                "dataset has not created windows dataset"
            )
        return self.windows_dataset[0][0].shape[1]

    def preprocess_dataset(self, pick_eeg=True, resample_freq=None, low_freq=None, high_freq=None):
        """

        Parameters
        ----------
        pick_eeg
        resample_freq
        low_freq
        high_freq

        Returns
        -------

        """
        preprocessors = []
        if pick_eeg:
            preprocessors.append(Preprocessor('pick_types', eeg=True, meg=False, stim=False))
        preprocessors.append(Preprocessor(lambda data: multiply(data, 1e6)))
        if low_freq or high_freq:
            preprocessors.append(Preprocessor('filter', l_freq=low_freq, h_freq=high_freq))
        if resample_freq:
            preprocessors.append(Preprocessor('resample', sfreq=resample_freq))
        preprocessors.append(Preprocessor(exponential_moving_standardize, factor_new=1e-3, init_block_size=1000))
        preprocess(self.raw_dataset, preprocessors)

    def create_windows_dataset(self, trial_start_offset_seconds=0, trial_stop_offset_seconds=0, mapping=None,
                               window_size_samples=None, window_stride_samples=None):
        """

        Parameters
        ----------
        trial_start_offset_seconds
        trial_stop_offset_seconds
        mapping
        window_size_samples
        window_stride_samples

        Returns
        -------

        """
        sfreq = self.get_sample_freq()
        assert all([ds.raw.info['sfreq'] == sfreq for ds in self.raw_dataset.datasets])
        trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)
        trial_stop_offset_samples = int(trial_stop_offset_seconds * sfreq)
        self.windows_dataset = create_windows_from_events(
            self.raw_dataset,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=trial_stop_offset_samples,
            window_size_samples=window_size_samples,
            window_stride_samples=window_stride_samples,
            preload=True,
            drop_last_window=False,
            mapping=mapping
        )
        return self.windows_dataset
