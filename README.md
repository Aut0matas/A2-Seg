# A2-Seg

 ***A***dversarial ***Seg***mentation with ***A***ttention-Unet

### Dependencies

* python 3.11
* Pytorch 2.0

### **Dataset**

- [ISBI International Skin Imaging Collaboration (ISIC) 2017 challenge, Part I Lesion Segmentation](https://challenge.kitware.com/#challenge/n/ISIC_2017%3A_Skin_Lesion_Analysis_Towards_Melanoma_Detection)

### **Project Structure**

| File          | Description                                                        |
| ------------- | ------------------------------------------------------------------ |
| dataLoader.py | (as it says)                                                       |
| transform.py  | Functions for data augmentation                                    |
| segmentor.py  | Model that predict segmentation masks, based on Attention U-net    |
| critic.py     | Model that takes masked images to produce multi-scale feature map |
| train.py      | (as it says)                                                       |

### **References**

- The code of this project is derived from [SegAN](https://github.com/YuanXue1993/SegAN) ([Yuan Xue, et al., 2018](https://arxiv.org/abs/1706.01805)), sincere respects to their work and briliant ideas.
