# Autoformer (NeurIPS 2021)

Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting

Time series forecasting is a critical demand for real applications. Enlighted by the classic time series analysis and stochastic process theory, we propose the Autoformer as a general series forecasting model [[paper](https://arxiv.org/abs/2106.13008)]. **Autoformer goes beyond the Transformer family and achieves the series-wise connection for the first time.**

In long-term forecasting, Autoformer achieves SOTA, with a **38% relative improvement** on six benchmarks, covering five practical applications: **energy, traffic, economics, weather and disease**.

:triangular_flag_on_post:**News** (2023.08) Autoformer has been included in [Hugging Face](https://huggingface.co/models?search=autoformer). See [blog](https://huggingface.co/blog/autoformer).

:triangular_flag_on_post:**News** (2023.06) The extension version of Autoformer ([Interpretable weather forecasting for worldwide stations with a unified deep model](https://www.nature.com/articles/s42256-023-00667-9)) has been published in Nature Machine Intelligence as the [Cover Article](https://www.nature.com/natmachintell/volumes/5/issues/6).

:triangular_flag_on_post:**News** (2023.02) Autoformer has been included in our [[Time-Series-Library]](https://github.com/thuml/Time-Series-Library), which covers long- and short-term forecasting, imputation, anomaly detection, and classification.

:triangular_flag_on_post:**News** (2022.02-2022.03) Autoformer has been deployed in [2022 Winter Olympics](https://en.wikipedia.org/wiki/2022_Winter_Olympics) to provide weather forecasting for competition venues, including wind speed and temperature.

## Autoformer vs. Transformers

**1. Deep decomposition architecture**

We renovate the Transformer as a deep decomposition architecture, which can progressively decompose the trend and seasonal components during the forecasting process.

<p align="center">
<img src=".\pic\Autoformer.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overall architecture of Autoformer.
</p>

**2. Series-wise Auto-Correlation mechanism**

Inspired by the stochastic process theory, we design the Auto-Correlation mechanism, which can discover period-based dependencies and aggregate the information at the series level. This empowers the model with inherent log-linear complexity. This series-wise connection contrasts clearly from the previous self-attention family.

<p align="center">
<img src=".\pic\Auto-Correlation.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 2.</b> Auto-Correlation mechansim.
</p>

## Get Started

1. Install Python 3.6, PyTorch 1.9.0.
2. Download data. You can obtain all the six benchmarks from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing). **All the datasets are well pre-processed** and can be used easily.
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results by:

```bash
bash ./scripts/ETT_script/Autoformer_ETTm1.sh
bash ./scripts/ECL_script/Autoformer.sh
bash ./scripts/Exchange_script/Autoformer.sh
bash ./scripts/Traffic_script/Autoformer.sh
bash ./scripts/Weather_script/Autoformer.sh
bash ./scripts/ILI_script/Autoformer.sh
```

4. Special-designed implementation

- **Speedup Auto-Correlation:** We built the Auto-Correlation mechanism as a batch-normalization-style block to make it more memory-access friendly. See the [paper](https://arxiv.org/abs/2106.13008) for details.

- **Without the position embedding:** Since the series-wise connection will inherently keep the sequential information, Autoformer does not need the position embedding, which is different from Transformers.

### Reproduce with Docker

To easily reproduce the results using Docker, conda and Make,  you can follow the next steps:
1. Initialize the docker image using: `make init`. 
2. Download the datasets using: `make get_dataset`.
3. Run each script in `scripts/` using `make run_module module="bash scripts/ETT_script/Autoformer_ETTm1.sh"` for each script.
4. Alternatively, run all the scripts at once:
```
for file in `ls scripts`; do make run_module module="bash scripts/$script"; done
```
### A Simple Example
See `predict.ipynb` for workflow (in Chinese).

## Main Results

We experiment on six benchmarks, covering five main-stream applications. We compare our model with ten baselines, including Informer, N-BEATS, etc. Generally, for the long-term forecasting setting, Autoformer achieves SOTA, with a **38% relative improvement** over previous baselines.

<p align="center">
<img src=".\pic\results.png" height = "550" alt="" align=center />
</p>

## Baselines

We will keep adding series forecasting models to expand this repo:

- [x] Autoformer
- [x] Informer
- [x] Transformer
- [x] Reformer
- [ ] LogTrans
- [ ] N-BEATS

## Citation

If you find this repo useful, please cite our paper. 

```
@inproceedings{wu2021autoformer,
  title={Autoformer: Decomposition Transformers with {Auto-Correlation} for Long-Term Series Forecasting},
  author={Haixu Wu and Jiehui Xu and Jianmin Wang and Mingsheng Long},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

## Contact

If you have any questions or want to use the code, please contact wuhx23@mails.tsinghua.edu.cn.

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/zhouhaoyi/Informer2020
https://github.com/zhouhaoyi/ETDataset
https://github.com/laiguokun/multivariate-time-series-data

# 原文翻译如下

# Autoformer （NeurIPS 2021）

Autoformer：具有自动关联功能的分解变压器，用于长期系列预测

时间序列预测是实际应用的关键需求。在经典时间序列分析和随机过程理论的启发下，我们提出了Autoformer作为一般序列预测模型[[paper]（https://arxiv.org/abs/2106.13008）]。**Autoformer 超越了 Transformer 系列，首次实现了串联。

在长期预测方面，Autoformer 实现了 SOTA，在六个基准上相对提高了 38%，涵盖了五个实际应用：**能源、交通、经济、天气和疾病**。

：triangular_flag_on_post：**新闻** （2023.08） Autoformer 已被收录于 [Hugging Face]（https://huggingface.co/models?search=autoformer）.请参阅 [博客]（https://huggingface.co/blog/autoformer）.

：triangular_flag_on_post：**新闻** （2023.06） Autoformer 的扩展版本（[Interpretable weather forecasting for worldwide stations with a unified deep model]（https://www.nature.com/articles/s42256-023-00667-9））已作为 [封面文章]（https://www.nature.com/natmachintell/volumes/5/issues/6）发表在 Nature Machine Intelligence 上。

：triangular_flag_on_post：**新闻** （2023.02） Autoformer 已被纳入我们的 [[时间序列库]]（https://github.com/thuml/Time-Series-Library），涵盖长期和短期预测、插补、异常检测和分类。

：triangular_flag_on_post：**新闻资讯** （2022.02-2022.03） Autoformer 已在 [2022 年冬季奥运会]（https://en.wikipedia.org/wiki/2022_Winter_Olympics） 中部署，为比赛场地提供天气预报，包括风速和温度。

## Autoformer 与 Transformers

**1.深度分解架构**

我们将 Transformer 改造为深度分解架构，可以在预测过程中逐步分解趋势和季节性成分。

<p align=“center”>

<img src=“.\pic\Autoformer.png” height = “250” alt=“” align=center />

<br><br>

<b>图 1.</b>Autoformer的整体架构。

</p>

**2.串联自动关联机制**

受随机过程理论的启发，我们设计了自相关机制，该机制可以发现基于周期的依赖关系，并在序列级别聚合信息。这使模型具有固有的对数线性复杂性。这种系列上的联系与之前的自我关注系列形成了鲜明的对比。

<p align=“center”>

<img src=“.\pic\Auto-Correlation.png” height = “250” alt=“” align=center />

<br><br>

<b>图2.</b>自相关机制

</p>

## 开始

1. 安装 Python 3.6、PyTorch 1.9.0。

2. 下载数据。您可以从 [Google Drive]（https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing） 获得所有六个基准测试。**所有数据集都经过良好的预处理**，可以轻松使用。

3. 训练模型。我们在文件夹“./scripts”下提供了所有基准测试的实验脚本。您可以通过以下方式重现实验结果：

'''bash

bash ./scripts/ETT_script/Autoformer_ETTm1.sh
bash ./scripts/ECL_script/Autoformer.sh
bash ./scripts/Exchange_script/Autoformer.sh
bash ./scripts/Traffic_script/Autoformer.sh
bash ./scripts/Weather_script/Autoformer.sh
bash ./scripts/ILI_script/Autoformer.sh

'''

4. 特殊设计的实现
- **加速自动关联：** 我们将自动关联机制构建为批量归一化式模块，使其对内存访问更加友好。详见[论文]（https://arxiv.org/abs/2106.13008）
- **没有位置嵌入：** 由于串联连接本身会保留顺序信息，因此 Autoformer 不需要位置嵌入，这与 Transformer 不同。

### 使用 Docker 重现

要使用 Docker、conda 和 Make 轻松重现结果，您可以按照以下步骤操作：

1. 使用“make init”初始化 docker 映像。
2. 使用“make get_dataset”下载数据集。
3. 在“scripts/”中运行每个脚本，对每个脚本使用 'make run_module module=“bash scripts/ETT_script/Autoformer_ETTm1.sh”'。
4. 或者，一次运行所有脚本：
'''

对于“ls scripts”中的文件;do make run_module module=“bash scripts/$script”;完成

'''

### 一个简单的例子

工作流请参见 'predict.ipynb'（中文）。

## 主要结果

我们在六个基准上进行了实验，涵盖了五个主流应用。我们将我们的模型与十个基线进行比较，包括 Informer、N-BEATS 等。一般来说，对于长期预测设置，Autoformer 实现了 SOTA，与以前的基线相比，相对提高了 38%。

<p align=“center”>

<img src=“.\pic\results.png” height = “550” alt=“” align=center />

</p

## 基线

我们将继续添加序列预测模型来扩展此回购：

- [x] 自动成型机
- [x] 告密者
- [x] 变压器
- [x] 改革者
- [ ] 日志传输
- [ ] N-节拍

## 引用

如果您觉得此存储库有用，请引用我们的论文。

'''
@inproceedings{wu2021autoformer，

title={Autoformer： Decomposition Transformers with {Auto-Correlation} for Long-Term Series Forecasting}，

作者={吴海旭、徐洁辉、王建民、龙明生}，

booktitle={神经信息处理系统进展}，

年份={2021}

}

'''
## 联系方式

如果您有任何疑问或想使用该代码，请联系 wuhx23@mails.tsinghua.edu.cn。
## 致谢

我们非常感谢以下 github 存储库的宝贵代码库或数据集：
https://github.com/zhouhaoyi/Informer2020
https://github.com/zhouhaoyi/ETDataset
https://github.com/laiguokun/multivariate-time-series-data