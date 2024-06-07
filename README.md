## Introduction

This is the offcial implementation of Paper "Sliced Denoising: A Physics-Informed Molecular Pre-Training Method"


## Requirement


The environment comprises the following packages and versions:

```
pytorch-lightning 1.8.6
torch 1.13.1+cu116
torch-cluster 1.6.0+pt113cu116
torch-geometric 2.3.0
torch-scatter 2.1.0+pt113cu116
torch-sparse 0.6.17+pt113cu116
torch-spline-conv 1.2.1+pt113cu116
torchmetrics 0.11.4
wandb 0.15.3
numpy 1.22.4
scikit-learn 1.2.2
scipy 1.8.1
deepchem 2.7.1
ogb 1.3.6
omegaconf 2.3.0
tqdm 4.66.2
```

The basic software and environment consist of Python 3.8, CUDA 11.6, Ubuntu 20.04.2 (OS version 9.4.0-1ubuntu1~20.04.2), and Linux kernel version 5.4.0-177-generic.

All experiments were conducted on a server equipped with 8 NVIDIA A100-PCIE-40GB GPUs.

Furthermore, an updated Conda environment package is available on Google Drive(https://drive.google.com/file/d/1X9gUELR6UAifUT7VVtgur2ZWfCGl7kcF/view?usp=sharing). You can download the environment package and unzip it into the 'envs' directory of Conda."



## Data and Models

To prepare for pre-training, please download the PCQMv2 Data from the following sources:

1. [OGB Stanford](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/)
2. [Figshare](https://figshare.com/articles/dataset/MOL_LMDB/24961485)

For fine-tuning, you will need to download the QM7 and MD17 datasets:

1. **QM9**: [Figshare](https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904)
2. **MD17**: [SGDML](http://www.sgdml.org/#datasets)



We provide pre-trained models for both QM9 and MD17, accessible through the following links:

Pre-trained model for QM9: [QM9 Pre-trained Model](https://1drv.ms/f/s!Arw91hjKoDJqnhcxVb9f4jYRJcRB?e=WKOmTa)

Pre-trained model for MD17: [MD17 Pre-trained Model](https://1drv.ms/f/s!Arw91hjKoDJqnhj00TABEnRPU_Po?e=HgCB0t)


Alternatively, you can download them from [figshare](https://figshare.com/articles/software/Pre-trained_model_for_Sliced_Denoising_A_Physics-Informed_Molecular_Pre-Training_Method/25990648).




## Pre-training 

- Prior to pre-training, it's essential to extract parameters related to bonds, angles, and torsions from the Open Force Field.


```
python get_force_param.py
```


- The Script for pre-training on the qm9


```
python -u scripts/train.py --conf examples/ET-PCQM4MV2_dih_var0.04_var2_force_mlp_f2d_dev_param.yaml --layernorm-on-vec whitened --job-id qm9_pretrain --sample-num 128 --num-epochs 8
```

- The Script for pre-training model for md17


```
python -u scripts/train.py --conf examples/ET-PCQM4MV2_dih_var0.04_var2_force_mlp_md17_f2d_dev_param.yaml --layernorm-on-vec whitened --job-id md17_pretrain --sample-num 128 --num-epochs 8 --gradient-clip-val 0.5

```





### Fine-tuning

- The Script for fine-tuning on the qm9


```

python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long_f2d.yaml --layernorm-on-vec whitened --job-id f2d_sample128_frad_multask_Areg_homo --dataset-arg homo --pretrained-model ${QM9_Pretrain_model_path} --denoising-weight 0.1 --model equivariant-transformerf2d --bond-length-scale 0.0 --dataset-root ${QM9_data_path}

```

Please replace ```${QM9_Pretrain_model_path}``` with the path to the pre-trained model, and ```${QM9_data_path}``` with the path to the fine-tuned data.


- The Script for fine-tuning model for md17

```
python -u scripts/train.py --conf examples/ET-MD17_FT-angle_f2d.yaml --dataset-root ${MD17_data_path} --job-id f2d_md17_sample128_frad_multask_Areg_omit_loss_nan_gcv0.5_aspirin --dataset-arg aspirin --pretrained-model ${MD17_pretrain_model_path} --dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --bond-length-scale 0.0 --lr 0.0005 --train-loss-type smooth_l1_loss --addh true
```

Please replace ```${MD17_pretrain_model_path}``` with the path to the pre-trained model, and ```${MD17_data_path}``` with the path to the fine-tuned data.


## CITE

If you find our work beneficial to your research or projects, kindly consider citing it.

```
@inproceedings{ni2023sliced,
  title={Sliced Denoising: A Physics-Informed Molecular Pre-Training Method},
  author={Ni, Yuyan and Feng, Shikun and Ma, Wei-Ying and Ma, Zhi-Ming and Lan, Yanyan},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```