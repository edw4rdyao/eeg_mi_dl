from braindecode.models import ShallowFBCSPNet, EEGNetv4
import torch
cuda = torch.cuda.is_available()


def get_shallow_conv_net(n_channels, n_classes, input_window_samples, final_conv_length='auto'):
    model = ShallowFBCSPNet(in_chans=n_channels, n_classes=n_classes, input_window_samples=input_window_samples,
                            final_conv_length=final_conv_length)
    if cuda:
        model.cuda()
    return model


def get_eeg_net(n_channels, n_classes, input_window_samples,
                final_conv_length='auto', kernel_length=64, drop_prob=0.25):
    model = EEGNetv4(in_chans=n_channels, n_classes=n_classes, input_window_samples=input_window_samples,
                     final_conv_length=final_conv_length, kernel_length=kernel_length, drop_prob=drop_prob)
    if cuda:
        model.cuda()
    return model
