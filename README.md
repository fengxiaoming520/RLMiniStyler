# RLMiniStyler: Light-weight RL Style Agent for Arbitrary Sequential Neural Style Generation（IJCAI2025）
*Authors: Jing Hu, [Chengming Feng](https://fengxiaoming520.github.io/), Shu Hu, Ming-Ching Chang, Xin Li, Xi Wu and Xin Wang\**

## Overview
<p align="center">
<img src="https://github.com/fengxiaoming520/RLMiniStyler/blob/main/Figure/introdemo.png" width="100%" height="100%">
</p> 

This repository is the official implementation of [RLMiniStyler: Light-weight RL Style Agent for Arbitrary Sequential Neural Style Generation](http://arxiv.org/abs/2505.04424).

## Framework
<p align="center">
<img src="https://github.com/fengxiaoming520/RLMiniStyler/blob/main/Figure/framework.jpg" width="100%" height="100%">
</p> 

Illustration of our arbitrary style sequence generation process. **Top Left:** Content and Style Images (5 style examples). **Right:** The sequence number of the results. Content images are progressively stylized with increasing strength along prediction sequences (see the index). Our method allows for easy control over stylization degree, preserving content details in early sequences and synthesizing more style patterns in later sequences, resulting in a user-friendly approach.

## Experiment
### Requirements
* Python 3.8
* Pytorch 1.9.0
* Torchvision 0.10.0
* Numpy, PIL
* Tqdm
* Tensorboard

### Testing 
Pretrained models: actor, builder, vgg-model   <br> 
Please download vgg-model and put it into the floder  ./metrics/  <br> 
```
python eval.py  --content figures/xxx/content.jpg --style_dir figures/xxx/style.jpg   
```

### Training  
Style dataset is WikiArt collected from [WikiArt](https://www.wikiart.org/)  <br>  
Content dataset is MS-COCO2014  <br>  
```
python main.py --style_dir ./Datasets/WikiArt/ --content_dir ./Datasets/MS-COCO-2014/
```

### Reference
If you find our work useful in your research, please cite our paper using the following BibTeX entry. Paper Link [pdf](https://arxiv.org/abs/2505.04424)<br> 
```
@article{hu2025rlministyler,
  title={RLMiniStyler: Light-weight RL Style Agent for Arbitrary Sequential Neural Style Generation},
  author={Hu, Jing and Feng, Chengming and Hu, Shu and Chang, Ming-Ching and Li, Xin and Wu, Xi and Wang, Xin},
  journal={arXiv preprint arXiv:2505.04424},
  year={2025}
}
```

### Acknowledgments
We refer to some ideas and codes from **[MicroAST](https://github.com/EndyWon/MicroAST)** and **[RL-I2IT](https://github.com/lesley222/RL-I2IT)**, express gratitude to them.
