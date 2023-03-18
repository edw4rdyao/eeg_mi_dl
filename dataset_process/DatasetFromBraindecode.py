"""
Dataset from braindecode lib. https://braindecode.org/stable/index.html
Encapsulate the operations
"""
from braindecode.datasets import MOABBDataset
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

    def get_sfreq(self):
        return self.dataset_instance.datasets[0].raw.info['sfreq']

    def create_windows_dataset(self, trial_start_offset_seconds=0, trial_stop_offset_seconds=0, mapping=None,
                               window_size_samples=None, window_stride_samples=None):
        sfreq = self.get_sfreq()
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
            window_size_samples=window_size_samples,
            window_stride_samples=window_stride_samples,
            preload=True,
            drop_last_window=False,
            mapping=mapping
        )
        return windows_dataset
