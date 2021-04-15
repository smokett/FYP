Dependencies:

Python 3.7 or above
PyTorch 1.8.1 (CUDA 10.2 or above for GPU computing)
numpy 1.19.3 or above
Mediapipe 0.8.2
OpenCV-contrib-python 4.5 or above
matplotlib 3.3.3 or above (for overfitting checking)
Matlab (for spearman correlation)
--------------------------

Datasets urls:

http://hasc.jp/HASC_BDD/ Ballroom Dance Dataset

http://rtis.oit.unlv.edu/datasets.html UNLV-Diving Dataset

--------------------------

Descriptions for the code:

extract_labels.py
-extract labels for dance movements segmentation and dance steps segmentation using BlazePose 

extract_RGB.py
-extract RGB images from dance dataset

RGB_crop.py
-apply corner cropping and scale jittering to RGB images

RGB_resize.py
-resize the processed RGB images to (224,224)

extract_OpticalFlow.py
-extract optical flow images from dance dataset

OpticalFlow_crop.py
-apply corner cropping and scale jittering to optical flow images

OpticalFlow_resize.py
-resize the processed optical flow images to (224,224)

check_data.py
-check if all the images were extracted

check_labels.py
-check if all the labels are correct.

RGBNet.py
-modify AlexNet to the spatial network of TSN

OpticalFlowNet.py
-modify AlexNet to the temporal network of TSN

basic_ops.py
-tools to compute segment consensus

gen_dataset.py
-generate the dataset file for dataloader

dataset.py
-PyTorch dataset class for PyTorch dataloader

train.py
-train the Siamese version of TSN

val.py
-evaluation for the testing data

calc_pairwise.py
-calculate the pairwise precision

calc_spearman.m (Matlab file)
-calculate the spearman correlation

overfitting.py
-plot the training loss and validation loss to check overfitting





