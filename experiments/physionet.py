import moabb
import torch
from braindecode.models import EEGNetv4
from braindecode.util import set_random_seeds
from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler, TrainEndCheckpoint
from skorch.helper import SliceDataset
from torch.utils.data import ConcatDataset

import dataset_loader
from nn_models import cuda

moabb.set_log_level("info")


class PhysionetExperiment:
    def __init__(self, args, config, logger):
        set_random_seeds(seed=config['fit']['seed'], cuda=cuda)
        self.all_valid_subjects, self.train_subjects, self.test_subjects = get_subject_split()
        self.ds = dataset_loader.DatasetFromBraindecode('physionet', subject_ids=self.all_valid_subjects)
        self.ds.uniform_duration(4.0)
        self.ds.drop_last_annotation()
        self.ds.preprocess(resample_freq=config['dataset']['resample'],
                           high_freq=config['dataset']['high_freq'],
                           low_freq=config['dataset']['low_freq'],
                           pick_channels=config['dataset']['channels'])
        self.n_classes = config['dataset']['n_classes']
        if self.n_classes == 3:
            events_mapping = {
                'left_hand': 0,
                'right_hand': 1,
                'feet': 2
            }
        else:
            events_mapping = {
                'left_hand': 0,
                'right_hand': 1,
                'feet': 2,
                'hands': 3
            }
        self.windows_dataset = self.ds.create_windows_dataset(
            trial_start_offset_seconds=config['dataset']['start_offset'],
            trial_stop_offset_seconds=config['dataset']['stop_offset'],
            mapping=events_mapping
        )
        self.n_channels = self.ds.get_channel_num()
        self.n_times = self.ds.get_input_window_sample()

        self.n_epochs = config['fit']['epochs']
        self.lr = config['fit']['lr']
        self.batch_size = config['fit']['batch_size']

        self.save = args.save
        # self.selection = args.selection
        self.save_dir = args.save_dir
        self.strategy = args.strategy
        self.model_name = args.model
        self.verbose = config['fit']['verbose']

        self.logger = logger
        self.logger.info("channels_name:", self.ds.get_channels_name())

    def __get_subjects_datasets(self, split_subjects):
        dataset_split_by_subject = self.windows_dataset.split('subject')
        if self.n_classes == 2:
            valid_dataset_for_2cls = []
            for i in split_subjects:
                for ds in dataset_split_by_subject[str(i)].datasets:
                    if 'left_hand' in ds.windows.event_id or 'right_hand' in ds.windows.event_id:
                        valid_dataset_for_2cls.append(ds)
            split_datasets = ConcatDataset(valid_dataset_for_2cls)
        else:
            split_datasets = ConcatDataset([dataset_split_by_subject[str(i)] for i in split_subjects])
        return split_datasets

    def __get_classifier(self):
        # for different models, suit the training routines or other params in the [origin paper or code] for classifier
        # NeuralNetClassifier is following skorch https://skorch.readthedocs.io/en/stable/user/neuralnet.html
        callbacks = [("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=self.n_epochs - 1))]
        if self.model_name == 'EEGNet':
            return NeuralNetClassifier(
                module=EEGNetv4,
                module__n_chans=self.n_channels,
                module__n_outputs=self.n_classes,
                module__n_times=self.n_times,
                module__kernel_length=32,
                module__drop_prob=0.5,
                criterion=torch.nn.CrossEntropyLoss,
                optimizer=torch.optim.Adam,
                optimizer__lr=self.lr,
                train_split=None,
                iterator_train__shuffle=True,
                batch_size=self.batch_size,
                callbacks=callbacks,
                device='cuda' if cuda else 'cpu',
                verbose=self.verbose
            )
        else:
            raise ValueError(f"model {self.model_name} is not supported on this dataset.")

    def __cross_subject_experiment(self):
        train_dataset = self.__get_subjects_datasets(self.train_subjects)
        test_dataset = self.__get_subjects_datasets(self.test_subjects)
        train_X = SliceDataset(train_dataset, idx=0)
        train_y = SliceDataset(train_dataset, idx=1)
        self.logger.info("train shape {}".format(train_X.shape))
        test_X = SliceDataset(test_dataset, idx=0)
        self.logger.info("test shape {}".format(test_X.shape))
        test_y = SliceDataset(test_dataset, idx=1)
        clf = self.__get_classifier()
        if self.save:
            clf.callbacks.append(TrainEndCheckpoint(dirname=self.save_dir))
        # if self.selection:
        #     clf.callbacks.append(("get_electrode_importance", utils.GetElectrodeImportance()))
        clf.fit(train_X, y=train_y, epochs=self.n_epochs)
        # calculate test accuracy for test subjects
        test_accuracy = clf.score(test_X, y=test_y)
        self.logger.info(f"{self.n_classes} classes test subjects accuracy: {(test_accuracy * 100):.4f}%")

    def __within_subject_experiment(self):
        pass

    def run(self):
        if self.strategy == 'within-subject':
            self.__within_subject_experiment()
        elif self.strategy == 'cross-subject':
            self.__cross_subject_experiment()


def get_subject_split():
    all_valid_subjects = []
    train_subjects = []
    test_subjects = []
    for i in range(1, 110):
        if i not in [88, 89, 92, 100]:
            all_valid_subjects.append(i)
            if i <= 84:
                train_subjects.append(i)
            else:
                test_subjects.append(i)
    return all_valid_subjects, train_subjects, test_subjects
