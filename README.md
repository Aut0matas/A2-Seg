### A²Seg: ***A***dversarial learning with channel-wise ***A***ttention for skin lesion ***Seg***mentation

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
| segmentor.py  | Model that predict segmentation masks, based on Attention U-net                                                 |
| critic.py     | Model that takes masked images to produce multi-scale feature map                                               |
| train.py      | (as it says)                                                                                                    |

### **Model Structure**

![model_fig](Images/A2-Seg.svg)

### **Results**

![result](Images/result.png)

### **References**

- The code of this project is greatly inspired by [SegAN](https://github.com/YuanXue1993/SegAN) ([Yuan Xue, et al., 2018](https://arxiv.org/abs/1706.01805)), sincere respects to their work and briliant ideas.
