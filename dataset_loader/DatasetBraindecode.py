from braindecode.preprocessing import create_windows_from_events
from braindecode.preprocessing import exponential_moving_standardize, preprocess, Preprocessor

from .DatasetEncap import DatasetEncap


class DatasetFromBraindecode(DatasetEncap):
    def __init__(self, dataset_name, subject_ids):
        super().__init__(dataset_name, subject_ids)

    def get_sample_freq(self):
        return self.raw_dataset.datasets[0].raw.info['sfreq']

    def get_channel_num(self):
        return self.raw_dataset[0][0].shape[0]

    def get_channels_name(self):
        return self.raw_dataset.datasets[0].raw.info['ch_names']

    def get_input_window_sample(self):
        if not self.windows_dataset:
            raise ValueError(
                "dataset has not created windows dataset"
            )
        return self.windows_dataset[0][0].shape[1]

    def preprocess(self, resample_freq=None, low_freq=None, high_freq=None, pick_channels=None):
        # preprocess data using "braindecode.preprocessor"
        # only use eeg(stim channels must be removed)
        preprocessors = [Preprocessor('pick', picks=['eeg'])]
        if pick_channels:
            preprocessors.append(Preprocessor('pick', picks=pick_channels))
        # preprocessors.append(Preprocessor(lambda data: multiply(data, 1e6)))
        if low_freq or high_freq:
            preprocessors.append(Preprocessor('filter', l_freq=low_freq, h_freq=high_freq))
        if resample_freq:
            preprocessors.append(Preprocessor('resample', sfreq=resample_freq))
        preprocessors.append(Preprocessor(exponential_moving_standardize, factor_new=1e-3))
        preprocess(self.raw_dataset, preprocessors)

    def uniform_duration(self, mapping):
        for ds in self.raw_dataset.datasets:
            if hasattr(ds, 'raw'):
                ds.raw.annotations.set_durations(mapping)
            else:
                raise ValueError('this operation is for mne.io.Raw')

    def drop_last_annotation(self):
        for ds in self.raw_dataset.datasets:
            if hasattr(ds, 'raw'):
                ds.raw.annotations.delete(len(ds.raw.annotations) - 1)
            else:
                raise ValueError('this operation is for mne.io.Raw')

    def create_windows_dataset(self, trial_start_offset_seconds=0, trial_stop_offset_seconds=0, mapping=None,
                               window_size_samples=None, window_stride_samples=None):
        # create trial data for training and test using "create_windows_from_events"
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
            mapping=mapping
        )
        return self.windows_dataset
