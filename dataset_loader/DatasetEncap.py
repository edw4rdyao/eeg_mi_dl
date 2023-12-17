"""
Dataset based on braindecode lib. https://braindecode.org/stable/index.html
Encapsulate the operations
"""
from braindecode.datasets import MOABBDataset


class DatasetEncap:
    def __init__(self, dataset_name, subject_ids):
        self.dataset_name = dataset_name
        self.subject_ids = subject_ids
        self.windows_dataset = None
        if dataset_name == 'bci2a':
            self.raw_dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=None)
        elif dataset_name == 'physionet':
            self.raw_dataset = MOABBDataset(dataset_name="PhysionetMI", subject_ids=subject_ids)
        else:
            raise ValueError(
                "dataset:%s is not supported" % dataset_name
            )

    def preprocess(self, resample_freq=None, low_freq=None, high_freq=None, pick_channels=None):
        pass

    def create_windows_dataset(self, trial_start_offset_seconds=0, trial_stop_offset_seconds=0, mapping=None,
                               window_size_samples=None, window_stride_samples=None):
        pass
