import torch
from braindecode.models import EEGNetv4, EEGConformer, ATCNet, EEGITNet, EEGInception
from braindecode.util import set_random_seeds
from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler, TrainEndCheckpoint
from skorch.helper import SliceDataset

import dataset_loader
from nn_models import cuda


class BCI2aExperiment:
    def __init__(self, args, config, logger):
        set_random_seeds(seed=config['fit']['seed'], cuda=cuda)
        # load and preprocess data using braindecode
        self.ds = dataset_loader.DatasetFromBraindecode('bci2a', subject_ids=None)
        self.ds.preprocess(resample_freq=config['dataset']['resample'],
                           high_freq=config['dataset']['high_freq'],
                           low_freq=config['dataset']['low_freq'])
        # clip to create window dataset
        self.windows_dataset = self.ds.create_windows_dataset(
            trial_start_offset_seconds=config['dataset']['start_offset'],
            trial_stop_offset_seconds=config['dataset']['stop_offset']
        )

        self.n_channels = self.ds.get_channel_num()
        self.n_times = self.ds.get_input_window_sample()
        self.n_classes = config['dataset']['n_classes']
        # training routine
        self.n_epochs = config['fit']['epochs']
        self.lr = config['fit']['lr']
        self.batch_size = config['fit']['batch_size']

        # user options
        self.save = args.save
        self.save_dir = args.save_dir
        self.strategy = args.strategy
        self.model_name = args.model
        self.verbose = config['fit']['verbose']

        self.logger = logger

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
        elif self.model_name == 'EEGConformer':
            return NeuralNetClassifier(
                module=EEGConformer,
                module__n_chans=self.n_channels,
                module__n_outputs=self.n_classes,
                module__n_times=self.n_times,
                module__final_fc_length='auto',
                module__add_log_softmax=False,
                criterion=torch.nn.CrossEntropyLoss,
                optimizer=torch.optim.Adam,
                optimizer__betas=(0.5, 0.999),
                optimizer__lr=self.lr,
                train_split=None,
                iterator_train__shuffle=True,
                batch_size=self.batch_size,
                callbacks=callbacks,
                device='cuda' if cuda else 'cpu',
                verbose=self.verbose
            )
        elif self.model_name == 'ATCNet':
            return NeuralNetClassifier(
                module=ATCNet,
                module__n_chans=self.n_channels,
                module__n_outputs=self.n_classes,
                module__n_times=self.n_times,
                module__add_log_softmax=False,
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
        elif self.model_name == 'EEGITNet':
            return NeuralNetClassifier(
                module=EEGITNet,
                module__n_chans=self.n_channels,
                module__n_outputs=self.n_classes,
                module__n_times=self.n_times,
                module__add_log_softmax=False,
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
        elif self.model_name == 'EEGInception':
            return NeuralNetClassifier(
                module=EEGInception,
                module__n_chans=self.n_channels,
                module__n_outputs=self.n_classes,
                module__n_times=self.n_times,
                module__add_log_softmax=False,
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

    def __within_subject_experiment(self):
        #  split dataset for single subject
        subjects_windows_dataset = self.windows_dataset.split('subject')
        n_subjects = len(subjects_windows_dataset.items())
        avg_accuracy = 0
        for subject, windows_dataset in subjects_windows_dataset.items():
            # evaluate the model by test accuracy for "Hold-Out" strategy
            train_dataset = windows_dataset.split('session')['0train']
            test_dataset = windows_dataset.split('session')['1test']
            train_X = SliceDataset(train_dataset, idx=0)
            train_y = SliceDataset(train_dataset, idx=1)
            test_X = SliceDataset(test_dataset, idx=0)
            test_y = SliceDataset(test_dataset, idx=1)
            clf = self.__get_classifier()
            # save the last epoch model for test
            if self.save:
                clf.callbacks.append(TrainEndCheckpoint(dirname=self.save_dir + f'/S{subject}'))
            clf.fit(train_X, y=train_y, epochs=self.n_epochs)
            # calculate test accuracy for subject
            test_accuracy = clf.score(test_X, y=test_y)
            avg_accuracy += test_accuracy
            self.logger.info(f"Subject{subject} test accuracy: {(test_accuracy * 100):.4f}%")
        self.logger.info(f"Average test accuracy: {(avg_accuracy / n_subjects * 100):.4f}%")

    def __cross_subject_experiment(self):
        pass

    def run(self):
        if self.strategy == 'within-subject':
            self.__within_subject_experiment()
        elif self.strategy == 'cross-subject':
            self.__cross_subject_experiment()
