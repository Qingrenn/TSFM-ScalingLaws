<div align="center">
  <h3><b>(ICLR'25) Towards Neural Scaling Laws for Time Series Foundation Models </b></h2>
</div>

<div align="center">

![](https://img.shields.io/github/last-commit/Qingrenn/TSFM-ScalingLaws?color=green)
![](https://img.shields.io/github/stars/Qingrenn/TSFM-ScalingLaws?color=yellow)
![](https://img.shields.io/github/forks/Qingrenn/TSFM-ScalingLaws?color=lightblue)
![](https://img.shields.io/badge/PRs-Welcome-green)

</div>


<div align="center">

**[<a href="https://arxiv.org/abs/2410.12360">Paper Page</a>]** 
**[<a href="https://iclr.cc/virtual/2025/poster/27992">Poster Page</a>]**
**[<a href="https://mp.weixin.qq.com/s/tSN2gSajYTpS9cDcGlmJlw">时序人中文解读</a>]**

</div>

## 1. 🚀 Install dependencies

```bash
pip install -r requirements.txt
```

## 2. 📚 Prepare the dataset

**Download datasets** from the [Qingren/TSFM-ScalingLaws-Dataset](https://huggingface.co/datasets/Qingren/TSFM-ScalingLaws-Dataset). The directory organization structure is as follows:

```bash
- dataset_train
    |- Lotsa16B
    |- Lotsa1B
    |- Lotsa100M
    |- Lotsa10M
- dataset_test
    |- Lotsa16B
    |- Lotsa1B
    |- Lotsa100M
    |- Lotsa10M
    |- LSF
    |- Monash
```

Create a `.env` file to indicate the pretraining dataset paths.

```bash
LOTSA_16B_PATH=PATH/TO/LOTSA_16B
LOTSA_1B_PATH=PATH/TO/LOTSA_1B
LOTSA_100M_PATH=PATH/TO/LOTSA_100M
LOTSA_10M_PATH=PATH/TO/LOTSA_10M
```

Test data is composed of three parts: in-distribution data `dataset_test/Lotsa[DataSize]`, out-of-distribution data `dataset_test/LSF` and `dataset_test/Monash`.

Take the test data of Lotsa16B as an example, the `storage_path` fields in config file `cli/conf/pretrain/val_data/Lotsa16B_multi.yaml` indicate the test data path. The default path is given as follows:

```yaml
- _target_: tsfm.data.builder.ConcatDatasetBuilder
  _args_:
    - _target_: tsfm.data.builder.simple.SimpleEvalDatasetBuilder
      ...
      storage_path: dataset_test/Monash
- _target_: tsfm.data.builder.ConcatDatasetBuilder
  _args_:
    - _target_: tsfm.data.builder.simple.SimpleEvalDatasetBuilder
      ...
      storage_path: dataset_test/LSF
- _target_: tsfm.data.builder.ConcatDatasetBuilder
  _args_:
    - _target_: tsfm.data.builder.simple.SimpleEvalDatasetBuilder
      ...
      storage_path: dataset_test/Lotsa16B
```

## 3. 🛠 Training Models

The hyperparameters of the model are defined in `cli/conf/pretrain/model/[Model]_[ModelSize].yaml`.

The general training config is defined in `cli/conf/pretrain/default_[ddp/fsdp]_val.yaml`

```bash
# train an encoder
python -m cli.train_val -cp conf/pretrain -cn default_ddp_val_enc \
model=encoder_10M \
data=lotsa16B_weighted \
val_data=lotsa16B_lsf_monash \
trainer.logger.project=demo_scalinglaws \
run_name=encoder10M_lotsa16B

# train a decoder
python -m cli.train_val -cp conf/pretrain -cn default_ddp_val_dec \
model=decoder_10M \
data=lotsa16B_weighted \
val_data=lotsa16B_lsf_monash \
trainer.logger.project=demo_scalinglaws \
run_name=decoder10M_lotsa16B
```

## 4. 📈 Data Analysis

When training models varying different numbers of parameters and different pretraining datasizes, the loss and metrics will be recorded via wandb. We need to rename each experiment in wandb following the format `[encoder/decoder]_[ModelSize]_[DataSize]`, such as `encoder_10M_16B`.

After collecting a series of experiments, download the wandb log and use the Jupyter scripts under `analysis` to fit and visualize the scaling laws.

## 5. 📦 Well-trained Models

The well-trained models are available in the [PeacefulData/TSFM-ScalingLaws-Checkpoints](https://huggingface.co/PeacefulData/TSFM-ScalingLaws-Checkpoints). You can try using the models with the Jupyter scripts in the `demo` directory.


## Citation

> 🙋 Please let us know if you find out a mistake or have any suggestions!

> 🌟 If you find the codebase helpful in your research, please consider to star this repository and cite the
> corresponding [paper](https://arxiv.org/abs/2410.12360):

```
@misc{yao2024towards,
      title={Towards Neural Scaling Laws for Time Series Foundation Models},
      author={Yao, Qingren and Yang, Chao-Han Huck and Jiang, Renhe and Liang, Yuxuan and Jin, Ming and Pan, Shirui},
      year={2024}
      eprint={2410.12360},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2410.12360}
}
```

## Further Reading
1, [**Estimating Time Series Foundation Model Transferability via In-Context Learning**](https://arxiv.org/pdf/2509.23695), *arXiv* 2025.

**Authors**: Qingren Yao, Ming Jin, Chengqi Zhang, Chao-Han Huck Yang, Jun Qi, Shirui Pan

```bibtex
@article{yao2025estimating,
  title={Estimating Time Series Foundation Model Transferability via In-Context Learning},
  author={Yao, Qingren and Jin, Ming and Zhang, Chengqi and Yang, Chao-Han Huck and Qi, Jun and Pan, Shirui},
  journal={arXiv preprint arXiv:2509.23695},
  year={2025}
}
```

2, [**Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts**](https://arxiv.org/pdf/2409.16040), in *ICLR* 2025.
[\[GitHub Repo\]](https://github.com/Time-MoE/Time-MoE)

**Authors**: Xiaoming Shi, Shiyu Wang, Yuqi Nie, Dianqi Li, Zhou Ye, Qingsong Wen, Ming Jin

```bibtex
@inproceedings{shi2024time,
  title={Time-moe: Billion-scale time series foundation models with mixture of experts},
  author={Shi, Xiaoming and Wang, Shiyu and Nie, Yuqi and Li, Dianqi and Ye, Zhou and Wen, Qingsong and Jin, Ming},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

🌟 Please also check out our team’s latest research projects listed below.
* TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models [\[paper\]](https://arxiv.org/pdf/2509.24803)
* T2S: High-resolution Time Series Generation with Text-to-Series Diffusion Models, in IJCAI 2025. [\[paper\]](https://arxiv.org/pdf/2505.02417) [\[GitHub Repo\]](https://github.com/WinfredGe/T2S)
* Time-MQA: Time Series Multi-Task Question Answering with Context Enhancement, in ACL 2025. [\[paper\]](https://arxiv.org/pdf/2503.01875) [\[Hugging Face\]](https://huggingface.co/Time-MQA)
* TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis, in ICLR 2025. [\[paper\]](https://arxiv.org/abs/2410.16032) [\[GitHub Repo\]](https://github.com/kwuking/TimeMixer)

## Acknowledgments

Our implementation builds upon the codebases of [Uni2ts](https://github.com/SalesforceAIResearch/uni2ts), which have been extensively modified to suit our specific requirements. We thank the authors of these implementations for sharing their code and providing related resources, which have been invaluable to this work.
