
# Blood Cell Detection Using Deep Learning and OpenCV

## Abstract

In this study, a system has been developed that automatically detects blood cells (RBC, WBC, and Platelets) in microscope images taken from peripheral blood smears. The system can identify and classify blood cells with high accuracy using modern computer vision techniques and deep learning methods. In the study, two different datasets obtained from Kaggle and GitHub were combined to create a comprehensive training set, and a transfer learning model based on the MobileNetV2 architecture was developed. The system can be used as an auxiliary tool in medical diagnosis and research.

## 1. Introduction

Automatic detection and classification of blood cells is an important application in hematology and medical diagnosis. Traditionally, blood cell analysis is performed manually by expert hematologists. However, this process is time-consuming, subjective, and prone to errors. Automatic blood cell detection systems can speed up this process, standardize it, and reduce human errors.

In recent years, developments in deep learning and computer vision have offered new possibilities for medical image analysis. Particularly, Convolutional Neural Networks (CNNs) have shown impressive results in image classification and object detection tasks. In this study, a transfer learning approach based on the MobileNetV2 architecture was used for blood cell detection.

## 2. Dataset

Two different datasets were used in this study:

1. **Kaggle Blood Cell Detection Dataset**: This dataset consists of blood cell images and a CSV file containing their bounding box coordinates.

2. **GitHub BCCD Dataset**: This dataset contains blood cell images and labels in XML format.

These two datasets were combined to create a more comprehensive training set. A total of 7228 cell images were processed and distributed as follows:

- RBC (Red Blood Cells): 6392 images
- WBC (White Blood Cells): 475 images
- Platelets: 361 images

The dataset was split into training (70%), test (15%), and validation (15%).

## 3. Methodology

### 3.1. Data Preparation

In the data preparation phase, the following steps were followed:

1. Loading images and extracting labels
2. Cropping cell images using bounding boxes
3. Splitting the dataset into training, test, and validation
4. Applying data augmentation techniques

The following transformations were used for data augmentation:
- Rotation (±20 degrees)
- Horizontal and vertical shift (±20%)
- Shear (±20%)
- Zoom (±20%)
- Horizontal flip

### 3.2. Model Architecture

In this study, a transfer learning approach based on the MobileNetV2 architecture was used. MobileNetV2 is a lightweight CNN architecture optimized for mobile and embedded devices. By using pre-trained weights on the ImageNet dataset, the model was adapted to the blood cell classification task.

The model architecture was designed as follows:
1. MobileNetV2 base model (with ImageNet weights)
2. Global Average Pooling layer
3. Fully connected layer with 128 neurons (with ReLU activation)
4. Dropout layer (with a rate of 0.5)
5. Output layer with 3 neurons (with Softmax activation)

### 3.3. Training Strategy

The training process was carried out in two phases:

1. **Initial Training Phase**: In this phase, all layers of the MobileNetV2 base model were frozen, and only the added classification layers were trained. This aims to quickly adapt the model to the blood cell classification task.

2. **Fine-Tuning Phase**: In this phase, the last 30 layers of the base model were made trainable, and training continued with a lower learning rate. This allows the model to learn features specific to blood cell images.

The following hyperparameters were used for training:
- Batch size: 32
- Initial learning rate: 0.001
- Fine-tuning learning rate: 0.0001
- Optimizer: Adam
- Loss function: Categorical Cross-Entropy

Additionally, the following callbacks were used to prevent overfitting and optimize the training process:
- ModelCheckpoint: To save the best model
- EarlyStopping: To stop training when validation loss does not improve
- ReduceLROnPlateau: To reduce the learning rate when validation loss plateaus

## 4. Results and Evaluation

The model was evaluated on the test dataset and the following metrics were obtained:

- Accuracy: 99.17%
- Precision: 98.92%
- Recall: 99.05%
- F1 Score: 98.98%

The confusion matrix and classification report show the model's performance for each cell type in detail.

## 5. User Interface

The developed system is equipped with a user-friendly graphical interface. This interface allows users to upload blood cell images, perform automatic detection, and visualize the results.

The interface includes the following features:
1. Image selection and loading
2. Automatic cell type detection
3. Bar chart showing prediction probabilities
4. Detailed display of detection results

## 6. Discussion and Future Work

In this study, a system that detects blood cells using deep learning and computer vision techniques has been developed. The system can classify three different blood cell types (RBC, WBC, and Platelets) with high accuracy.

In future work, the following improvements can be made:
- Increasing model performance with larger and more diverse datasets
- Developing versions that can run on mobile devices using lighter models
- Adding additional features such as cell counting and morphology analysis
- Developing a web-based interface to provide remote access
- Developing specialized models for the diagnosis of different blood diseases

## 7. Conclusion

This study demonstrates that deep learning and computer vision techniques can be effectively used in medical image analysis applications such as blood cell detection. The developed system can be used as an auxiliary tool in medical diagnosis and research and can save time and resources by automating manual analysis processes.

## References

1. Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., ... & Adam, H. (2017). MobileNets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861.
2. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4510-4520).
3. Acevedo, A., Merino, A., Alférez, S., Molina, Á., Boldú, L., & Rodellar, J. (2020). A dataset of microscopic peripheral blood cell images for development of automatic recognition systems. Data in Brief, 30, 105474.
4. Hegde, R. B., Prasad, K., Hebbar, H., & Singh, B. M. K. (2019). Development of a robust algorithm for detection of nuclei and classification of white blood cells in peripheral blood smear images. Journal of Medical Systems, 43(8), 1-12.
