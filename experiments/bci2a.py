import moabb
import torch
from braindecode import EEGClassifier
from braindecode.augmentation import AugmentedDataLoader
from braindecode.models import EEGNetv4, ShallowFBCSPNet, EEGConformer
from braindecode.models import to_dense_prediction_model, get_output_shape
from braindecode.training import CroppedLoss
from braindecode.util import set_random_seeds
from skorch.callbacks import LRScheduler, TrainEndCheckpoint
from torch.nn import functional
from torchinfo import summary

import dataset_loader
from nn_models import cuda
from utils import get_augmentation_transform, save_str2file

moabb.set_log_level("info")


class BCI2aExperiment:
    def __init__(self, args, config):
        # load and preprocess data using braindecode
        self.ds = dataset_loader.DatasetFromBraindecode('bci2a', subject_ids=None)
        self.ds.preprocess_dataset(resample_freq=config['dataset']['resample'],
                                   high_freq=config['dataset']['high_freq'],
                                   low_freq=config['dataset']['low_freq'])
        # clip to create window dataset
        # self.windows_dataset = self.ds.create_windows_dataset(trial_start_offset_seconds=-0.5)
        self.windows_dataset = self.ds.create_windows_dataset()
        self.n_channels = self.ds.get_channel_num()
        self.input_window_samples = self.ds.get_input_window_sample()
        self.n_classes = config['dataset']['n_classes']
        # training routine
        self.n_epochs = config['fit']['epochs']
        self.lr = config['fit']['lr']
        self.batch_size = config['fit']['batch_size']

        self.save = args.save
        self.save_dir = args.save_dir
        self.strategy = args.strategy
        self.model_name = args.model

        # load deep leaning model from braindecode(reproduce)
        if args.model == 'EEGNet':
            self.model = EEGNetv4(n_chans=self.n_channels, n_outputs=self.n_classes,
                                  n_times=self.input_window_samples, kernel_length=32, drop_prob=0.5)
            summary(self.model, (1, self.n_channels, self.input_window_samples, 1))
        elif args.model == 'EEGConformer':
            self.model = EEGConformer(n_outputs=self.n_classes, n_chans=self.n_channels,
                                      n_times=self.input_window_samples,
                                      final_fc_length='auto', add_log_softmax=False)
            summary(self.model, (1, self.n_channels, self.input_window_samples))
        else:
            raise ValueError(f"model {args.model} is not supported on this dataset.")
        if cuda:
            self.model.cuda()

    def __get_classifier(self):
        # for different models, suit the training routines or other params in the origin paper or code for classifier
        if self.model_name == 'EEGNet':
            callbacks = [("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=self.n_epochs - 1))]
            return EEGClassifier(module=self.model,
                                 criterion=torch.nn.CrossEntropyLoss,
                                 optimizer=torch.optim.Adam,
                                 optimizer__lr=self.lr,
                                 train_split=None,
                                 batch_size=self.batch_size,
                                 callbacks=callbacks,
                                 device='cuda' if cuda else 'cpu',
                                 verbose=0
                                 )
        elif self.model_name == 'EEGConformer':
            callbacks = []
            return EEGClassifier(module=self.model,
                                 criterion=torch.nn.CrossEntropyLoss,
                                 optimizer=torch.optim.Adam,
                                 optimizer__betas=(0.5, 0.999),
                                 optimizer__lr=self.lr,
                                 train_split=None,
                                 batch_size=self.batch_size,
                                 callbacks=callbacks,
                                 device='cuda' if cuda else 'cpu'
                                 )

    def __within_subject_experiment(self):
        #  split dataset for single subject
        subjects_windows_dataset = self.windows_dataset.split('subject')
        n_subjects = len(subjects_windows_dataset.items())
        avg_accuracy = 0
        result = ''
        for subject, windows_dataset in subjects_windows_dataset.items():
            # evaluate the model by test accuracy for "Hold-Out" strategy
            split_by_session = windows_dataset.split('session')
            train_set = split_by_session['session_T']
            test_set = split_by_session['session_E']
            clf = self.__get_classifier()
            # save the last epoch model for test
            if self.save:
                clf.callbacks.append(TrainEndCheckpoint(dirname=self.save_dir+f'\\S{subject}'))
            clf.fit(train_set, y=None, epochs=self.n_epochs)
            # calculate test accuracy for subject
            test_accuracy = clf.score(test_set, y=test_set.get_metadata().target)
            avg_accuracy += test_accuracy
            print(f"Subject{subject} test accuracy: {(test_accuracy * 100):.4f}%")
            result += f"Subject{subject} test accuracy: {(test_accuracy * 100):.4f}%\n"
        print(f"Average test accuracy: {(avg_accuracy / n_subjects * 100):.4f}%")
        # save the result
        result += f"Average test accuracy: {(avg_accuracy / n_subjects * 100):.4f}%"
        save_str2file(result, self.save_dir, 'result.txt')

    def __cross_subject_experiment(self):
        pass

    def run(self):
        if self.strategy == 'within-subject':
            self.__within_subject_experiment()
        elif self.strategy == 'cross-subject':
            self.__cross_subject_experiment()


def bci2a_shallow_conv_net():
    set_random_seeds(seed=20233202, cuda=cuda)
    ds = dataset_loader.DatasetFromBraindecode('bci2a', subject_ids=None)
    ds.preprocess_dataset(low_freq=4, high_freq=38)
    n_channels = ds.get_channel_num()
    input_window_samples = 1000
    model = ShallowFBCSPNet(in_chans=n_channels, n_classes=4, input_window_samples=input_window_samples,
                            final_conv_length=30, drop_prob=0.25)
    if cuda:
        model.cuda()
    summary(model, (1, n_channels, input_window_samples, 1))
    # for cropped training
    to_dense_prediction_model(model)
    n_preds_per_input = get_output_shape(model, n_channels, input_window_samples)[2]
    windows_dataset = ds.create_windows_dataset(trial_start_offset_seconds=-0.5,
                                                window_size_samples=input_window_samples,
                                                window_stride_samples=n_preds_per_input)
    transforms = get_augmentation_transform(sample_freq=ds.get_sample_freq())
    n_epochs = 300
    lr = 0.000625
    weight_decay = 0
    batch_size = 64
    clf = EEGClassifier(module=model, iterator_train=AugmentedDataLoader, iterator_train__transforms=transforms,
                        train_split=None, criterion=CroppedLoss, criterion__loss_function=functional.nll_loss,
                        optimizer=torch.optim.AdamW, optimizer__lr=lr,
                        optimizer__weight_decay=weight_decay, batch_size=batch_size,
                        callbacks=[("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1))],
                        cropped=True, device='cuda' if cuda else 'cpu'
                        )
