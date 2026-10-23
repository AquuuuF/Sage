# Sage: Semantic-guided Fine-tuning of Foundation Model for Long-tailed Visual Recognition
This is the official repo for the paper [Semantic-guided Fine-tuning of Foundation Model for Long-tailed Visual Recognition](https://link.springer.com/article/10.1007/s11263-026-02874-4) (IJCV 2026). 

The open-source version of our work will be released soon.


## Requirements
* Python 3.12.3
* PyTorch 2.3.1（CUDA 12.1）
* Torchvision 0.18.1
* TensorBoard 2.17.0

- For other dependencies, please refer to [LIFT](https://github.com/shijxcs/LIFT).


## Quick Start 
```bash
# Sage on CIFAR100-LT, IR=100
python -u main_Sage.py --data cifar100_ir100 --model clip_vit_b16_cifar100_ir100 output_dir Sage_cifar100_ir100

# Sage on CIFAR100-LT, IR=50


# Sage on CIFAR100-LT, IR=10


# Sage on Places-LT


# Sage on ImageNet-LT


# Sage on iNaturalist 2018


```

## Results and Models for CIFAR-100-LT and Large-scale Datasets 


| Dataset     | Top-1 Accuracy | Log           | Model |
| ----------- | -------------- | ------------- | ----- |
| CIFAR-100-LT, IR=100 | 83.3%        | [link](https://drive.google.com/file/d/1ZkATPZyLbGrHdYYuGaioOoe6csCOP69w/view?usp=sharing)        | [link](https://drive.google.com/file/d/14QKBaMzS8daJ2ZHyuA0_bxc2TRpOOyX4/view?usp=sharing)  |
| CIFAR-100-LT, IR=50 | 84.6%        | [link](https://drive.google.com/file/d/1P8gmTXwPijSrx3i3JOXZKH-zY1Sx_KtT/view?usp=sharing)       | [link](https://drive.google.com/file/d/1MPF_0e21BB4qnkyMT7GahYuLwZyzJ3hm/view?usp=sharing) |
| CIFAR-100-LT, IR=10 | 86.3%        | [link](https://drive.google.com/file/d/1Q7ljvM4EysaXl0EBbwA_0Kbnq9OsUPxx/view?usp=sharing)        | [link](https://drive.google.com/file/d/1k7Ok45ZoEqJXzhBYCu0G389XFFeNmzjV/view?usp=sharing)  |
| Places-LT   | 52.7%        | [link](https://drive.google.com/file/d/1ElS7X81BDOu-4YDx-6vkgvg3yJNyrQFL/view?usp=sharing)        | [link](https://drive.google.com/file/d/1RRCsj6dTRlkjgr5G7J4_Bubtod3nLTz6/view?usp=sharing)  |
| ImageNet-LT | 78.8%        | [link](https://drive.google.com/file/d/1dKJ_wS5JCr-QfrLgyHmXfD7goqNzCasD/view?usp=sharing)        | [link](https://drive.google.com/file/d/1IrBUsCGB7H63I7einZj7IYoW_Fbis8YB/view?usp=sharing)  |
| iNat 2018   | 80.9%        | [link]()       | [link]() |



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


