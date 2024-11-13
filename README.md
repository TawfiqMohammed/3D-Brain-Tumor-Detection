# 3D MRI Brain Tumor Segmentation using U-Net

![test_gif_BraTS20_Training_001_flair](https://github.com/user-attachments/assets/687109a0-40e3-4a57-9540-7716af22875b)

This notebook demonstrates how to build, train, and evaluate a 3D U-Net model for brain tumor segmentation using MRI images. U-Net is a convolutional neural network architecture designed specifically for medical image segmentation tasks.

## Project Overview

Brain tumor segmentation is a critical step in medical diagnosis and treatment planning. Manual segmentation is time-consuming and prone to human error. This notebook uses deep learning techniques to automate the segmentation of brain tumors from 3D MRI scans using a U-Net architecture.

## Key Features

- **Dataset**: 3D MRI images containing brain scans with ground truth labels for tumor regions.
- **Model Architecture**: 3D U-Net model tailored for volumetric medical image segmentation.
- **Preprocessing**: Normalization and resizing of 3D MRI scans.
- **Training**: Implementation of loss functions, optimizers, and metrics for model training.
- **Evaluation**: Quantitative metrics such as Dice coefficient and visual inspection of segmented regions.

## Contents

- **Data Loading and Preprocessing**: How to load MRI data, normalize it, and prepare it for model training.
- **U-Net Model Implementation**: Implementation of a 3D U-Net model designed for segmentation tasks.
- **Model Training and Validation**: Code for training the U-Net on the MRI dataset, along with real-time validation to monitor performance.
- **Segmentation Results**: Visualization of the predicted tumor segmentations on MRI slices and comparison to ground truth labels.
- **Evaluation Metrics**: Dice score, precision, recall, and other metrics to assess segmentation performance.

## Prerequisites

Before running the notebook, ensure you have the following dependencies installed:

- Python 3.x
- TensorFlow or PyTorch (depending on the framework used)
- NumPy
- SciPy
- Matplotlib
- Nibabel (for working with MRI data)
- Scikit-learn

You can install these dependencies using the following command:

```bash
pip install numpy scipy matplotlib nibabel scikit-learn tensorflow
```

## Usage

1. **Data Preparation**: Ensure your dataset of 3D MRI brain images is properly formatted and loaded into the notebook.
2. **Run the Notebook**: Execute the notebook cells in order to preprocess data, train the model, and visualize results.
3. **Model Training**: Adjust hyperparameters such as learning rate, batch size, and number of epochs to improve performance.
4. **Evaluation**: After training, the notebook will provide a detailed evaluation of the model's segmentation performance.

## Results

The results section of the notebook will display both qualitative and quantitative analysis of the U-Net model's segmentation accuracy. Example output includes:

- 3D visualizations of segmented tumors.
- Dice coefficient and accuracy metrics.
  
## Customization

To adapt this notebook for your specific dataset:

1. **Modify Data Loading Section**: Ensure the MRI data is loaded in the correct format and paths are updated.
2. **Model Architecture**: Adjust the U-Net layers or add additional custom layers if needed.
3. **Hyperparameters**: Tune hyperparameters like learning rate, batch size, or number of epochs for optimal performance.

## Acknowledgements

- The U-Net architecture was originally introduced in the paper "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Olaf Ronneberger et al.
- MRI dataset was sourced from BraTS2020 Dataset - [Link](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation).
  
## License
This project is licensed under the MIT License - see the LICENSE file for details.
