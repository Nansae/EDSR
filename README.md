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

* If you run the train.py file, the saved_models folder will store the training results.  
* The run test.py to see the results
```
python train.py
python test.py
```

## Result
## Reference
- https://github.com/thstkdgus35/EDSR-PyTorch
- https://github.com/jmiller656/EDSR-Tensorflow
- [arXiv]: https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf

