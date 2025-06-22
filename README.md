# Specialization Capstone Project: TIFF Image Classification with cuDNN

This project performs image classification on `.tiff` or `.tif` images stored in a specified directory. It utilizes NVIDIA CUDA and cuDNN libraries to handle GPU-accelerated image preprocessing and inference simulation. The classification output includes predicted class indices and associated probabilities for each image, saved in a results text file.

## Overview

The program performs the following steps:

1. **Initialization**:
   Initializes CUDA and cuDNN libraries and displays GPU properties such as compute capability.

2. **Image Loading and Preprocessing**:
   Loads TIFF images using the libTIFF library and normalizes RGB channels to the \[0, 1] range. Each image is converted into a cuDNN-compatible 4D tensor (NHWC format).

3. **Mock Inference**:
   A mock forward pass simulates a classification model by generating random class probabilities for a fixed number of output classes.

4. **Result Logging**:
   For each image, the predicted class and the full list of class probabilities are written to a specified output text file.

5. **Cleanup**:
   Releases all allocated GPU memory and destroys the cuDNN handle to ensure clean termination.

## Dependencies

This project requires the following libraries and tools:

* CUDA Toolkit (version 10.2 or higher)
* cuDNN (compatible with your CUDA version)
* libTIFF (`libtiff-dev` on Linux)
* C++17-compatible compiler (e.g., `g++`, `clang++`)



## Usage

Run the executable by providing the input folder path containing TIFF images and the path to the output results file:

```bash
./classify_images <input_folder_path> <output_file.txt>
```

Example:

```bash
./classify_images ./images ./results.txt
```

## Output Format

The output text file will contain entries for each processed image in the following format:

```
File: images/image1.tiff
Predicted class: 7
Class probabilities: 0.12 0.03 0.08 0.05 0.01 0.10 0.15 0.22 0.14 0.10

File: images/image2.tiff
Predicted class: 2
Class probabilities: 0.09 0.07 0.21 0.12 0.10 0.09 0.08 0.05 0.08 0.11
```
