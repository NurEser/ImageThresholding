# ImageThresholding
This repository includes implementations of global thresholding using Otsu's method and mean thresholding as an adaptive thresholding technique.

<p align="center">
  <img src="https://github.com/NurEser/ImageThresholding/blob/main/photo-1606590056137-c3c4f42074d4_processed.jpg"  alt="Global output" width="45%"/>
  <img src="https://github.com/NurEser/ImageThresholding/blob/main/photo-1606590056137-c3c4f42074d4_processed.jpg" alt="Adaptive Output" width="45%"/> 
</p>


## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)

## Introduction 

Image thresholding is a technique in image processing used for various applications such as object detection, segmentation, and feature extraction. This project implements two main types of image thresholding:
		Global Thresholding: In this method, a single threshold value is used to binarize the entire image. For global thresholding, Otsu's method is utilized to determine the optimal threshold value. 
		Adaptive Thresholding: This method calculates different threshold values for different regions of the image. With this method, I aimed for aesthetically appealing pictures by employing the mean thresholding technique. 

## Installation

### Dependencies

-Python
- NumPy
- PIL (Pillow)
- Matplotlib
- SciPy


```
$ pip install numpy Pillow matplotlib scipy
```

        
## Usage

The script can be executed from the command line by providing the path to an input image or directory of images, the output directory path, and the thresholding type.

```
$ python image_thresholding.py <input_path> <output_directory> <threshold_type>
```

	input_path: Path to an image file or a directory containing multiple images.
	output_directory: Path where the processed images will be saved.
	threshold_type: Either 'global' for Otsu's method or 'adaptive' for mean thresholding.




