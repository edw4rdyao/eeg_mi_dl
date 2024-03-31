# EEG Motor Imagery Deep Learning

[English](README.md) [中文](README.zh_CN.md)

关于深度学习方法在**脑机接口中运动想象脑电数据分类**的应用，包括脑电信号处理（可视化和分析）、领域论文整理和总结、深度学习模型复现和实验。


本仓库的实验基于 <a href="https://github.com/mne-tools/mne-python">MNE-Python</a>，
<a href="https://github.com/NeuroTechX/moabb">Moabb</a>，
<a href="https://github.com/braindecode/braindecode">Braindecode</a>和
<a href="https://github.com/skorch-dev/skorch">Skorch</a>。

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

这个仓库基于`Python 3.8`，在运行这个仓库的实验之前，请先安装环境：

```shell
$ pip install -r requirements.txt
```

然后你可以使用`-h`来获取使用方法：
```shell
$ python .\main.py -h
```
```
usage: main.py [-h] [--dataset {bci2a,physionet}] [--model {EEGNet,EEGConformer,ATCNet,EEGInception,EEGITNet}] [--config CONFIG] [--strategy {cross-subject,within-subject}] [--save]

optional arguments:
  -h, --help            show this help message and exit
  --dataset {bci2a,physionet}
                        data set used of the experiments
  --model {EEGNet,EEGConformer,ATCNet,EEGInception,EEGITNet}
                        model used of the experiments
  --config CONFIG       config file name(.yaml format)
  --strategy {cross-subject,within-subject}
                        experiments strategy on subjects
  --save                save the pytorch model and history (follow skorch)
```

如果你在数据集[BCI 2a](#公开数据集) 上运行`EEGNet`模型的实验，只需运行：
```shell
$ python .\main.py --dataset bci2a --model EEGNet
```

它将使用[bci2a_EEGNet_default.yaml](./config/bci2a_EEGNet_default.yaml)中的默认配置和默认的`within-subject`策略，当然你可以使用`--config`来指定。

然后你可以在`./save`文件夹中获取输出的准确率和`result.log`：
```
[13:41:23 2024] Subject1 test accuracy: 70.4861%
[13:42:06 2024] Subject2 test accuracy: 57.9861%
[13:42:50 2024] Subject3 test accuracy: 79.5139%
[13:43:33 2024] Subject4 test accuracy: 60.0694%
[13:44:16 2024] Subject5 test accuracy: 70.8333%
[13:45:00 2024] Subject6 test accuracy: 59.3750%
[13:45:46 2024] Subject7 test accuracy: 69.0972%
[13:46:30 2024] Subject8 test accuracy: 62.1528%
[13:47:13 2024] Subject9 test accuracy: 68.4028%
[13:47:13 2024] Average test accuracy: 66.4352%
```

你也可以修改配置的 yaml 文件来调整参数或制作你自己的模型来进行实验。