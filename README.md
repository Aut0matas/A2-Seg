### A²Seg: ***A***dversarial ***Seg***mentation with ***A***ttention-Unet

### Dependencies

* python 3.11  (below should be fine)
* Pytorch 2.0  (below should be fine)
* CUDA

### **Dataset**

- [ISBI International Skin Imaging Collaboration (ISIC) 2017 challenge, Part I Lesion Segmentation](https://challenge.isic-archive.com/data/#2017)

### **Project Structure**

| File          | Description                                                                                                     |
| ------------- | --------------------------------------------------------------------------------------------------------------- |
| dataLoader.py | (as it says)                                                                                                    |
| transform.py  | Functions for data augmentation                                                                                 |
| segmentor.py  | Model that predict segmentation masks, based on[Attention U-net](https://github.com/LeeJunHyun/Image_Segmentation) |
| critic.py     | Model that takes masked images to produce multi-scale feature map                                              |
| train.py      | (as it says)                                                                                                    |

### **Model Structure**

![1687022387504](image/README/1687022387504.png)

### **References**

- The code of this project is derived from [SegAN](https://github.com/YuanXue1993/SegAN) ([Yuan Xue, et al., 2018](https://arxiv.org/abs/1706.01805)), sincere respects to their work and briliant ideas.
