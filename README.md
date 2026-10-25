# Sage: Semantic-guided Fine-tuning of Foundation Model for Long-tailed Visual Recognition
This is the official repo for the paper [Semantic-guided Fine-tuning of Foundation Model for Long-tailed Visual Recognition](https://link.springer.com/article/10.1007/s11263-026-02874-4) (IJCV 2026). 


## Requirements
* Python 3.12.3
* PyTorch 2.3.1（CUDA 12.1）
* Torchvision 0.18.1
* TensorBoard 2.17.0

- For other dependencies, please refer to [LIFT](https://github.com/shijxcs/LIFT).


## Quick Start 
```bash
# Sage on CIFAR100-LT, IR=100
python -u main_Sage.py --data cifar100_ir100 --model clip_vit_b16_cifar100_ir100

# Run Sage on CIFAR100-LT with an imbalance ratio of 100 (IR=100), specifying `dataset_dir` as the dataset path and `output_dir` as the directory for saving results.
python -u main_Sage.py --data cifar100_ir100 --model clip_vit_b16_cifar100_ir100 \
  root /dataset_dir \
  output_dir /output_dir

# Sage on CIFAR100-LT, IR=50
python -u main_Sage.py --data cifar100_ir50 --model clip_vit_b16_cifar100_ir50

# Sage on CIFAR100-LT, IR=10
python -u main_Sage.py --data cifar100_ir10 --model clip_vit_b16_cifar100_ir10

# Sage on Places-LT
python -u main_Sage.py --data places_lt --model clip_vit_b16_places_lt

# Sage on ImageNet-LT
python -u main_Sage.py --data imagenet_lt --model clip_vit_b16_imagenet_lt

# Sage on iNaturalist 2018
python -u main_Sage.py --data inat2018 --model clip_vit_b16_inat2018

```


## Results and Models for CIFAR-100-LT and Large-scale Datasets 

* All the Logs and Checkpoints can be downloaded from [link](https://drive.google.com/drive/folders/1sj8T7px4MhPqBMRrMHmk5pDyJlVFEqr7?usp=sharing) . 

| Dataset     | Top-1 Accuracy | Log           | Model |
| ----------- | -------------- | ------------- | ----- |
| CIFAR-100-LT, IR=100 | 83.3%        | [link](https://drive.google.com/file/d/1XEJuE1PB5tQw8dYq0OUfzmyk5dz63yMW/view?usp=sharing)        | [link](https://drive.google.com/file/d/1dgHRrcGAXWGmN0v-9_hHbQJaHR4-PKVT/view?usp=sharing)  |
| CIFAR-100-LT, IR=50 | 84.6%        | [link](https://drive.google.com/file/d/1eoSgM_A2kEV9BhLbqIlzvZrbIzILvBHx/view?usp=sharing)       | [link](https://drive.google.com/file/d/1drafvp0O-5PrRd66XXovDm1ybotIEpkD/view?usp=sharing) |
| CIFAR-100-LT, IR=10 | 86.3%        | [link](https://drive.google.com/file/d/1HN5YuEcu1uCIuAn5QBeI3JOdmeRYvE1f/view?usp=sharing)        | [link](https://drive.google.com/file/d/1yeLCUbBtnMftzOy0tGtJPA8nnEZCg67_/view?usp=sharing)  |
| Places-LT   | 52.7%        | [link](https://drive.google.com/file/d/1PI2TPiJp7EJ76kLRySyR7jFyfx34rXYl/view?usp=sharing)        | [link](https://drive.google.com/file/d/1Ei1i9_kBbLsKBPV3ZfXmXEE6XIoUYbAm/view?usp=sharing)  |
| ImageNet-LT | 78.8%        | [link](https://drive.google.com/file/d/1v3PEAft8Qn7B_Xx57ycZa-AsNOicw-B8/view?usp=sharing)        | [link](https://drive.google.com/file/d/1tF6z05TIHRWOoTtfAgIkqXpP5yi5VsDj/view?usp=sharing)  |
| iNat 2018   | 80.9%        | [link](https://drive.google.com/file/d/1r5T3i2FMrpMj68Rwc6Ke6pYBQCdbshVr/view?usp=sharing)       | [link](https://drive.google.com/file/d/1VLAdoDkemCj7ErV-723DejHRvMNNwZ2X/view?usp=sharing) |


## Acknowledgment

We thank the authors for the following repositories for code reference:
[LIFT](https://github.com/shijxcs/LIFT).


## Citation

If you find this repo useful for your work, please cite as:

```bibtex
@article{peng2026semantic,
  title={Semantic-guided Fine-tuning of Foundation Model for Long-tailed Visual Recognition},
  author={Peng, Yufei and Zhang, Yonggang and Cheung, Yiu-ming},
  journal={International Journal of Computer Vision},
  volume={134},
  number={7},
  pages={322},
  year={2026},
  publisher={Springer}
}
```


