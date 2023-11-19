# EEG Motor Imagery Deep Learning

[English](README.md) [中文](README.zh_CN.md)

A research repository of deep learning on electroencephalographic (EEG) for Motor imagery(MI), including eeg data
processing(visualization & analysis), papers(research and summary), deep learning models(reproduction and experiments).

<details>
<summary>
The experiments in this repository are based on <a href="https://github.com/mne-tools/mne-python">MNE-Python</a>, 
<a href="https://github.com/NeuroTechX/moabb">Moabb</a>,
<a href="https://github.com/braindecode/braindecode">Braindecode</a>.
</summary>

- [MNE-Python](https://github.com/mne-tools/mne-python)

MNE-Python is an open-source Python package for exploring, visualizing, and analyzing human neurophysiological data
such as MEG, EEG, sEEG, ECoG, and more. It includes modules for data input/output, preprocessing, visualization, source
estimation, time-frequency analysis, connectivity analysis, machine learning, and statistics.

- [Moabb](https://github.com/NeuroTechX/moabb)

Moabb: Build a comprehensive benchmark of popular Brain-Computer Interface (BCI) algorithms applied on an extensive
list of freely available EEG datasets.

- [Braindecode](https://github.com/braindecode/braindecode)

Braindecode is an open-source Python toolbox for decoding raw electrophysiology brain data with deep learning models.
It includes dataset fetchers, data preprocessing and visualization tools, as well as implementations of several deep
learning architectures and data augmentations for analysis of EEG, ECoG and MEG.

</details>

You can find more contents of this repository through the following sections:

- [Paper Researching](#field-researching)
- [EEG Data Analysis and Processing](#eeg-data-analysis-and-processing)
- [Experiments](#experiments)

## Field Researching

Field Researching currently includes the following papers and datasets:

- [Awesome Papers](#awesome-papers)
- [Public Datasets](#public-datasets)

### Awesome Papers

<details>
<summary>
2017 Schirrmeister et al.
<u><i>Deep learning with convolutional neural networks for EEG decoding and visualization</i></u>
[<a href="https://onlinelibrary.wiley.com/doi/10.1002/hbm.23730"><b>paper link</b></a>]
[<a href="https://github.com/robintibor/braindecode"><b>source code</b></a>]
[<a href="https://github.com/robintibor/braindecode"><b>reproduce1</b></a>]
</summary>
</details>

<details>
<summary>
2018 Lawhern et al.
<u><i>EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces</i></u>
[<a href="https://iopscience.iop.org/article/10.1088/1741-2552/aace8c"><b>paper link</b></a>]
[<a href="https://github.com/vlawhern/arl-eegmodels"><b>source code</b></a>]
[<a href="https://github.com/braindecode/braindecode/tree/master/braindecode/models"><b>reproduce1</b></a>]
[<a href="https://colab.research.google.com/drive/1ANF8PwvtUPawTeQt4Uu4iwscpyhHBgvM"><b>reproduce2</b></a>]
</summary>
</details>

<details>
<summary>
2018 Sakhavi et al.
<u><i>Learning Temporal Information for Brain-Computer Interface Using Convolutional Neural Networks</i></u>
[<a href="https://ieeexplore.ieee.org/document/8310961"><b>paper link</b></a>]
</summary>
</details>

<details>
<summary>
2019 Dose et al.
<u><i>An end-to-end deep learning approach to MI-EEG signal classification for BCIs</i></u>
[<a href="https://www.sciencedirect.com/science/article/abs/pii/S0957417418305359"><b>paper link</b></a>]
[<a href="https://github.com/hauke-d/cnn-eeg"><b>source code</b></a>]
</summary>
</details>

<details>
<summary>
2020 Wang et al.
<u><i>An Accurate EEGNet-based Motor-Imagery Brain Computer Interface for Low-Power Edge Computing</i></u>
[<a href="https://ieeexplore.ieee.org/document/9137134"><b>paper link</b></a>]
[<a href="https://github.com/MHersche/eegnet-based-embedded-bci"><b>source code</b></a>]
</summary>
</details>

<details>
<summary>
2020 Ingolfsson et al.
<u><i>EEG-TCNet: An Accurate Temporal Convolutional Network for Embedded Motor-Imagery Brain-Machine Interfaces</i></u>
[<a href="https://ieeexplore.ieee.org/document/9283028"><b>paper link</b></a>]
[<a href="https://github.com/iis-eth-zurich/eeg-tcnet"><b>source code</b></a>]
[<a href="https://github.com/okbalefthanded/eeg-tcnet/blob/master/eeg_tcnet_colab.ipynb"><b>reproduce1</b></a>]
</summary>
</details>

<details>
<summary>
2021 Mane et al.
<u><i>A Multi-view CNN with Novel Variance Layer for Motor Imagery Brain Computer Interface</i></u>
[<a href="https://ieeexplore.ieee.org/document/9175874"><b>paper link</b></a>]
[<a href="https://github.com/ravikiran-mane/FBCNet"><b>source code</b></a>]
</summary>
</details>

### Public Datasets

List of the most frequently used public datasets in the papers

- BCI IV 2a([BCI Competition IV](https://www.bbci.de/competition/iv/))

Dataset description: [BCI Competition 2008 – Graz data set A](https://www.bbci.de/competition/iv/desc_2a.pdf)

Download link: [.gdf format](https://www.bbci.de/competition/iv/#dataset2a)
or [.mat format](http://bnci-horizon-2020.eu/database/data-sets)

![](./static/bci2a.png)

- Physionet([Physionet Dataset](https://physionet.org/content/eegmmidb/1.0.0/))

Dataset
description: [Physionet Database EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)

Download link: [.edf format](https://physionet.org/content/eegmmidb/1.0.0/)

![](./static/physionet.png) [[Image reference](https://www.sciencedirect.com/science/article/abs/pii/S0957417418305359)]

*For deep learning experiments, to easier downloading of datasets and faster data processing, it is recommended to
use [Moabb dataset](http://moabb.neurotechx.com/docs/datasets.html#module-moabb.datasets)
or [Braindecode dataset](https://braindecode.org/stable/generated/braindecode.datasets.BNCI2014001.html) to do
experiments.*

## EEG Data Analysis and Processing

- Data Load and Analysis

Using MNE-Python library with Jupyter Notebook to analyze demo EEG data of [BCI IV 2a](#public-datasets), including
loading data, plotting signal, extracting events...

For details and code, please move
to [data_load_visualization.ipynb](./data_analysis_notebook_mne/data_load_visualization.ipynb), for more examples,
to [MNE-Python tutorials](https://mne.tools/stable/auto_tutorials/index.html).

- Data Processing

Using MNE-Python library with Jupyter Notebook to process demo EEG data of [BCI IV 2a](#public-datasets), including
filtering, resampling, segmenting data ...

For details and code, please move to [data_processing.ipynb](./data_analysis_notebook_mne/data_processing.ipynb), for
more examples, to [MNE-Python tutorials](https://mne.tools/stable/auto_tutorials/index.html).

## Experiments
