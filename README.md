# FreqMAE: Frequency-Aware Masked Autoencoder for Multi-Modal IoT Sensing Applications

Authors: **Denizhan Kara**, **Tomoyoshi Kimura**, **Shengzhong Liu**, **Jinyang Li**, **Dongxin Liu**, **Tianshi Wang**, **Ruijie Wang**, **Yizhuo Chen**, **Yigong Hu**, **Tarek Abdelzaher**


Link [[pdf](https://openreview.net/pdf?id=bCW6kFz8fh)]

## Overview

This paper presents **FreqMAE**, a novel self-supervised learning framework that integrates masked autoencoding with physics-informed insights for IoT sensing applications. It captures feature patterns from multi-modal IoT sensing signals, enhancing latent feature space representation to reduce reliance on data labeling and improve AI task accuracy. Unlike methods dependent on data augmentations, FreqMAE's approach eliminates the need for handcrafted transformations by utilizing insights from the frequency domain. We showcase three primary contributions:

1. **Temporal-Shifting Transformer (TS-T) Encoder**: Enables temporal interactions while distinguishing different frequency regions.
2. **Factorized Multimodal Fusion**: Leverages cross-modal correlations while preserving modality-specific features.
3. **Hierarchically Weighted Loss Function**: Prioritizes reconstruction of crucial frequency components and high SNR samples.

Comprehensive evaluations on sensing applications confirm FreqMAE's effectiveness in reducing labeling requirements and enhancing domain shift resilience.

## Installation and Requirements

1. **Dependencies**: Create a conda environment with Python 3.10 for isolation.
    ```bash
    conda create -n freqmae_env python=3.10; conda activate freqmae_env
    ```

2. **Installation**: Clone the repository and install the necessary packages.
    ```bash
    git clone [repo] freqmae_dir
    cd freqmae_dir
    pip install -r requirements.txt
    ```

## Dataset

We evaluate FreqMAE using two IoT sensing applications: Moving Object Detection (MOD) and a self-collected dataset using acoustic and seismic signals for vehicle classification. Data preprocessing involves spectrogram generation and masking, detailed further in the supplementary materials.

### Moving Object Detection (MOD)

MOD is a self-collected dataset that uses acoustic (8000Hz) and seismic (100Hz) signals to classify types of moving vehicles. The pre-training dataset includes data from 10 classes, and the downstream tasks include vehicle classification, distance classification, and speed classification.


- Vehicle data: [Box link](https://uofi.box.com/s/bv37vqfd0a5d9rhnvfoo7gld96gtj8jv)
- Distance speed raw data: [Box link](https://uofi.box.com/s/8yffx3417mrxbdsqqtder4d7kk7rryb8)


### Data Preprocessing

1. Update the configuration in `src/data_preprocess/MOD/preprocessing_configs.py`.
2. Navigate to the data preprocessing directory:
    ```bash
    cd src/data_preprocess/MOD/
    ```
3. Execute the sample extraction and data partitioning scripts as detailed below.

#### Sample Extraction and Data Partition

- **Pretrain Data Extraction**: Generate samples for unsupervised pretraining.
    ```bash
    python extract_pretrain_samples.py
    ```
- **Supervised Data Extraction**: Generate labeled samples for supervised fine-tuning.
    ```bash
    python extract_samples.py
    ```
- **Data Partitioning**: Partition data into training, validation, and test sets.
    ```bash
    python partition_data.py
    ```

## Usage

### Training and Fine-Tuning

- **Supervised Training**: Use the following command for training with a specific model and dataset.
    ```bash
    python train.py -model=[MODEL] -dataset=[DATASET]
    ```
- **FreqMAE Pre-Training**: Pre-train the model using FreqMAE.
    ```bash
    python train.py -model=[MODEL] -dataset=[DATASET] -learn_framework=FreqMAE
    ```
- **Fine-Tuning**: After pre-training, fine-tune for a specific task.
    ```bash
    python train.py -model=[MODEL] -dataset=[DATASET] -learn_framework=FreqMAE -task=[TASK] -stage=finetune -model_weight=[PATH TO MODEL WEIGHT]
    ```

### Model Configurations

See `src/data/*.yaml` for model configurations specific to each dataset.

## License

This project is released under the MIT License. See `LICENSE` for details.

## Citation

Please cite our paper if you use this code or dataset in your research:

```latex
@inproceedings{freqmae2024,
  title={FreqMAE: Frequency-Aware Masked Autoencoder for Multi-Modal IoT Sensing Applications},
  author={Kara, Denizhan and Kimura, Tomoyoshi and Liu, Shengzhong and Li, Jinyang and Liu, Dongxin and Wang, Tianshi and Wang, Ruijie and Chen, Yizhuo and Hu, Yigong and Abdelzaher, Tarek},
  booktitle={Proceedings of the ACM Web Conference 2024},
  year={2024}
}
```

## Contact

For any inquiries regarding the paper or the code, please reach out to us:

- Denizhan Kara: kara4@illinois.edu
- Tomoyoshi Kimura: tkimura4@illinois.edu