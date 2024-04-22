# semi-proto-seg
designed for semi-supervised prostate segmentation with lesion prototype used 
![model](semi_seg-main_net.jpeg)
Figure1. The workflow of the proposed network that consists of four paths, labelled Paths 1-4. Paths 1 and 4 process support (labelled) images and Paths 2 and 3 process query (unlabelled) images.
## dependencies
Python 3.6 +
PyTorch 1.0.1
torchvision 0.2.1
NumPy, SciPy, PIL
pycocotools
sacred 0.7.5
tqdm 4.32.2

## code
Drawing inspiration from the PANet repository, the code has undergone extensive modifications tailored specifically to the nuanced demands of semi-supervised segmentation within medical imaging.
## usage
```
python train.py
```
change configuration in "config.py" 
## data
I have uploaded the processed publicly available dataset "prostateX" into Drive. If you need to run the demo, please download the dataset to your [DataFolder].
To download the dataset:https://portland-my.sharepoint.com/:u:/g/personal/wenyan6-c_my_cityu_edu_hk/Ebwb8KxOpIlCj0p2tIlyF4QB68j65H0OcrHt9pWvrRvirg?e=0FfJF9
