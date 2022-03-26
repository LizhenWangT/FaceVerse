# FaceVerse
### FaceVerse: a Fine-grained and Detail-changeable 3D Neural Face Model from a Hybrid Dataset
Lizhen Wang, Zhiyuan Chen, Tao Yu, Chenguang Ma, Liang Li, Yebin Liu  CVPR 2022

[[Dataset]](https://github.com/LizhenWangT/FaceVerse-Dataset)

![teaser](./docs/cover.jpg)

### Abstract
We present FaceVerse, a fine-grained 3D Neural Face Model, which is built from hybrid East Asian face datasets containing 60K fused RGB-D images and 2K high-fidelity 3D head scan models. A novel coarse-to-fine structure is proposed to take better advantage of our hybrid dataset. In the coarse module, we generate a base parametric model from large-scale RGB-D images, which is able to predict accurate rough 3D face models in different genders, ages, etc. Then in the fine module, a conditional StyleGAN architecture trained with high-fidelity scan models is introduced to enrich elaborate facial geometric and texture details. Note that different from previous methods, our base and detailed modules are both changeable, which enables an innovative application of adjusting both the basic attributes and the facial details of 3D face models. Furthermore, we propose a single-image fitting framework based on differentiable rendering. Rich experiments show that our method outperforms the state-of-the-art methods.

![teaser](./docs/results.jpg)
Single-image fitting results using FaceVerse model.

### To do lists
1. Open the single-image fitting source code;

2. Open the training code;

3. Open the video-tracking code using our base model;

### Citation
If you use this dataset for your research, please consider citing:
```
@InProceedings{wang2022faceverse,
title={FaceVerse: a Fine-grained and Detail-changeable 3D Neural Face Model from a Hybrid Dataset},
author={Wang, Lizhen and Chen, Zhiyua and Yu, Tao and Ma, Chenguang and Li, Liang and Liu, Yebin},
booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR2022)},
month={June},
year={2022},
}
```

### Contact
- Lizhen Wang [(wlz18@mails.tsinghua.edu.cn)](wlz18@mails.tsinghua.edu.cn)
- Zhiyuan Chen [(juzhen.czy@antfin.com)](juzhen.czy@antfin.com)
- Yebin Liu [(liuyebin@mail.tsinghua.edu.cn)](mailto:liuyebin@mail.tsinghua.edu.cn)
