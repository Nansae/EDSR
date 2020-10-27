# EDSR
The repository has implemented the **Enhanced Deep Residual Networks for Single Image Super-Resolution**   

**Requirements**
* pytorch   
* pillow
* matplotlib
* tqdm
* natsort

## Concept
EDSR is one way to solve the super resolution problem of a single image. This repository is a study implementation of EDSR. The original link is in the reference.  
<img src="https://user-images.githubusercontent.com/11286586/97261602-7d1f5880-1862-11eb-9acd-d5f4070f35b1.png" width="50%" height="50%"><img src="https://user-images.githubusercontent.com/11286586/97261604-7e508580-1862-11eb-8f0a-ef41405b76c5.png" width="50%" height="50%">

## Files and Directories
* data.py : This file loads the 50x50 images and 100x100 images
* model.py : This file has implemented EDSR network by pytorch
* test.py : This file inputs 50x50 images in the test folder into the model and compares it with the 100x100x label image.
* train.py : This file training the 50x50 images  as a 100x100 label image in the data/train folder.

```
data
  |
  |--- train
  |     |
  |     |---- 50x
  |     |       |---0.jpg
  |     |       |---1.jpg...        
  |     |---- 100x
  |--- test
  |     |
  |     |---- 50x
  |     |       |---0.jpg
  |     |       |---1.jpg...        
  |     |---- 100x
```

* I used General-100 dataset for training and testing. 
* If you run the train.py file, the saved_models folder will store the training results.  
* The run test.py to see the results
* Training results are saved as *.pth in the **saved_models** folder.
```
python train.py
python test.py
```

## Result
input : 50x50 image that can be entered. [(Download)](https://drive.google.com/file/d/0B7tU5Pj1dfCMVVdJelZqV0prWnM/view)  
output : The result of converting the 50x50 image to EDSR.  
origin : 100x100 original image.  

<img src="https://user-images.githubusercontent.com/11286586/97263032-71816100-1865-11eb-9c6f-55eb671cca19.png" width="50%" height="50%">

## Reference
- https://github.com/thstkdgus35/EDSR-PyTorch
- https://github.com/jmiller656/EDSR-Tensorflow
- [arXiv](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf)

