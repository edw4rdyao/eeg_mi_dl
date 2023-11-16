# EEG Motor Imagery Deep Learning

[English](README.md) [中文](README.zh_CN.md)

关于深度学习方法在**脑机接口中运动想象脑电数据分类**的应用，包括脑电信号处理（可视化和分析）、领域论文整理和总结、深度学习模型复现和实验。

本仓库的实验基于以下几个开源库及工具箱：

- [MNE-Python](https://github.com/mne-tools/mne-python)
  
MNE-Python 是一个开源 Python 包，用于探索、可视化和分析人类神经生理学数据，例如 MEG、EEG、sEEG、ECoG 等。它包括数据输入/输出、预处理
、可视化、源估计、时频分析、连通性分析、机器学习和统计等模块。

- [Moabb](https://github.com/NeuroTechX/moabb)

Moabb：建立流行脑机接口（BCI）算法的综合基准，应用于大量免费可用的脑电图数据集。

- [Braindecode](https://github.com/braindecode/braindecode)

Braindecode 是一个开源 Python 工具箱，用于使用深度学习模型解码原始大脑数据。它包括数据集获取、数据预处理和可视化工具，以及用于
EEG、ECoG 和 MEG 分析的多种深度学习模型和数据增强的实现。

您可以通过以下部分找到此仓库的更多内容：

- [领域调研](#领域调研)
- [脑电数据分析处理](#脑电数据分析处理)
- [实验](#实验)

## 领域调研

领域调研目前包括论文以及公开数据集:

- [论文整理](#论文整理)
- [公开数据集](#公开数据集)

### 论文整理

<details>
<summary>
2017 Schirrmeister et al.
<u><i>Deep learning with convolutional neural networks for EEG decoding and visualization</i></u>
[<a href="https://onlinelibrary.wiley.com/doi/10.1002/hbm.23730"><b>论文链接</b></a>]
[<a href="https://github.com/robintibor/braindecode"><b>开源代码</b></a>]
[<a href="https://github.com/robintibor/braindecode"><b>复现代码</b></a>]
</summary>
</details>

<details>
<summary>
2018 Lawhern et al.
<u><i>EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces</i></u>
[<a href="https://iopscience.iop.org/article/10.1088/1741-2552/aace8c"><b>论文链接</b></a>]
[<a href="https://github.com/vlawhern/arl-eegmodels"><b>开源代码</b></a>]
[<a href="https://github.com/braindecode/braindecode/tree/master/braindecode/models"><b>复现代码1</b></a>]
[<a href="https://colab.research.google.com/drive/1ANF8PwvtUPawTeQt4Uu4iwscpyhHBgvM"><b>复现代码2</b></a>]
</summary>
</details>

<details>
<summary>
2018 Sakhavi et al.
<u><i>Learning Temporal Information for Brain-Computer Interface Using Convolutional Neural Networks</i></u>
[<a href="https://ieeexplore.ieee.org/document/8310961"><b>论文链接</b></a>]
</summary>
</details>

<details>
<summary>
2019 Dose et al.
<u><i>An end-to-end deep learning approach to MI-EEG signal classification for BCIs</i></u>
[<a href="https://www.sciencedirect.com/science/article/abs/pii/S0957417418305359"><b>论文链接</b></a>]
[<a href="https://github.com/hauke-d/cnn-eeg"><b>开源代码</b></a>]
</summary>
</details>

<details>
<summary>
2020 Wang et al.
<u><i>An Accurate EEGNet-based Motor-Imagery Brain Computer Interface for Low-Power Edge Computing</i></u>
[<a href="https://ieeexplore.ieee.org/document/9137134"><b>论文链接</b></a>]
[<a href="https://github.com/MHersche/eegnet-based-embedded-bci"><b>开源代码</b></a>]
</summary>
</details>

<details>
<summary>
2020 Ingolfsson et al.
<u><i>EEG-TCNet: An Accurate Temporal Convolutional Network for Embedded Motor-Imagery Brain-Machine Interfaces</i></u>
[<a href="https://ieeexplore.ieee.org/document/9283028"><b>论文链接</b></a>]
[<a href="https://github.com/iis-eth-zurich/eeg-tcnet"><b>开源代码</b></a>]
[<a href="https://github.com/okbalefthanded/eeg-tcnet/blob/master/eeg_tcnet_colab.ipynb"><b>复现代码1</b></a>]
</summary>
</details>

<details>
<summary>
2021 Mane et al.
<u><i>A Multi-view CNN with Novel Variance Layer for Motor Imagery Brain Computer Interface</i></u>
[<a href="https://ieeexplore.ieee.org/document/9175874"><b>论文链接</b></a>]
[<a href="https://github.com/ravikiran-mane/FBCNet"><b>开源代码</b></a>]
</summary>
</details>


### 公开数据集

论文中最常用的脑电运动想象公开数据集：

- BCI IV 2a([BCI Competition IV](https://www.bbci.de/competition/iv/))

数据集描述: [BCI Competition 2008 – Graz data set A](https://www.bbci.de/competition/iv/desc_2a.pdf)
  
下载链接: [.gdf 格式](https://www.bbci.de/competition/iv/#dataset2a) 
或者 [.mat 格式](http://bnci-horizon-2020.eu/database/data-sets)

![](./static/bci2a.png)


- Physionet([Physionet Dataset](https://physionet.org/content/eegmmidb/1.0.0/))

数据集描述:[Physionet Database EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)
  
下载链接: [.edf 格式](https://physionet.org/content/eegmmidb/1.0.0/)

![](./static/physionet.png) [[图片参考](https://www.sciencedirect.com/science/article/abs/pii/S0957417418305359)]

*为了快速下载和预处理数据集建议使用第三方库 
[Moabb dataset](http://moabb.neurotechx.com/docs/datasets.html#module-moabb.datasets) 
或者 [Braindecode dataset](https://braindecode.org/stable/generated/braindecode.datasets.BNCI2014001.html)来做实验*

## 脑电数据分析处理

- 数据加载和分析
  
使用 MNE-Python 库和 Jupyter Notebook 分析[BCI IV 2a](#公开数据集)的演示脑电图数据，包括加载数据、绘制信号、提取事件等等。

详细代码见[data_load_visualization.ipynb](./data_analysis_notebook_mne/data_load_visualization.ipynb), 
更多示例见[MNE-Python tutorials](https://mne.tools/stable/auto_tutorials/index.html)。


- 数据处理

使用 MNE-Python 库和 Jupyter Notebook 处理 [BCI IV 2a](#公开数据集) 的演示脑电图数据，包括过滤、重采样、分割数据等等。

详细代码见[data_processing.ipynb](./data_analysis_notebook_mne/data_processing.ipynb), 
更多示例见 [MNE-Python tutorials](https://mne.tools/stable/auto_tutorials/index.html)。

## 实验
