# IMAGE DENOISING

This Jupyter notebook demonstrates the use of a deep convolutional neural network (CNN) autoencoder for image denoising. The model is trained to remove noise from images, producing cleaner versions as output.

## Table of Contents
1. [Installation of Libraries](#installation-of-libraries)
2. [Loading and Preprocessing Images](#loading-and-preprocessing-images)
3. [Splitting the Dataset](#splitting-the-dataset)
4. [PSNR Function and Custom PSNR Callback](#psnr-function-and-custom-psnr-callback)
5. [Model Architecture](#model-architecture)
6. [Training and Testing Loss Plot](#training-and-testing-loss-plot)
7. [PSNR Score Calculation](#psnr-score-calculation)
8. [Predicted Images](#predicted-images)
9. [Noisy Images](#noisy-images)
10. [Clean Images](#clean-images)

## Installation of Libraries
This section covers the installation of necessary libraries. Ensure you have the required libraries installed before running the notebook.

## Loading and Preprocessing Images
In this section, images are loaded and preprocessed to prepare them for training the autoencoder model. This includes resizing images and adding noise to create the training dataset.

## Splitting the Dataset
The dataset is split into training and testing sets. This step is crucial to evaluate the performance of the model on unseen data.

## PSNR Function and Custom PSNR Callback
Here, the Peak Signal-to-Noise Ratio (PSNR) function is defined along with a custom callback to monitor PSNR during training.

## Model Architecture
This section defines the architecture of the deep CNN autoencoder. The model is built using convolutional layers that help in learning the features of the images for denoising.

## Training and Testing Loss Plot
After training the model, this section plots the training and testing loss over epochs to visualize the learning process.

## PSNR Score Calculation
The PSNR score for the testing data is calculated in this section to measure the quality of the denoised images.

## Predicted Images
Here, the denoised (predicted) images are displayed to show the results of the model's performance.

## Noisy Images
This section displays the noisy images that were used as input for the model.

## Clean Images
The clean images (ground truth) are displayed for comparison with the predicted images.

## How to Run the Notebook
1. Clone the repository or download the notebook file.
2. Install the necessary libraries as mentioned in the first section.
3. Load and preprocess the images as described.
4. Split the dataset into training and testing sets.
5. Define the PSNR function and custom callback.
6. Build and train the model using the defined architecture.
7. Plot the training and testing loss.
8. Calculate the PSNR score for the testing data.
9. Display the predicted, noisy, and clean images for evaluation.

## Requirements
- Python 3.x
- Jupyter Notebook
- Necessary Python libraries (e.g., TensorFlow, NumPy, Matplotlib)

## Acknowledgements
This project is based on the principles of deep learning and image processing. Special thanks to the authors and contributors of the libraries used in this notebook.
