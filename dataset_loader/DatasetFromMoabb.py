"""
Dataset from moabb lib. https://moabb.neurotechx.com/docs/index.html
Encapsulate the operations
"""
from moabb.datasets import BNCI2014001


def _load_bci2a_moabb():
    dataset_description = "Dataset IIa from BCI Competition 4"
    return BNCI2014001(), dataset_description


class DatasetFromMoabb:
    def __init__(self, dataset_name):
        if dataset_name == 'bci2a':
            self.dataset_instance, self.dataset_description = _load_bci2a_moabb()
        else:
            raise ValueError(
                "dataset:%s is not supported" % dataset_name
            )
