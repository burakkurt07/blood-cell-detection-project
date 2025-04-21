# Blood Cell Detection Project

This project aims to develop a system that detects blood cells (RBC, WBC, and Platelets) using OpenCV and deep learning. The project can automatically identify and classify blood cells using image processing and machine learning techniques.

## About the Project

Blood cell detection is an important application in medical diagnosis and research. This project uses modern computer vision and deep learning techniques to detect blood cells in microscope images taken from peripheral blood smears.

Two different datasets were used in the project:
1. "Blood Cell Detection Dataset" downloaded from Kaggle
2. "BCCD Dataset" downloaded from GitHub

These datasets were combined to create a more comprehensive training set.

## Features

- Processing and preparing blood cell images
- MobileNetV2-based transfer learning model
- Classification of three different blood cell types (RBC, WBC, Platelet)
- User-friendly graphical interface
- Detailed time tracking and performance evaluation

## Project Structure

```
blood_cell_recognition/
├── data/                      # Datasets and processed data
│   ├── kaggle/                # Dataset downloaded from Kaggle
│   ├── github/                # Dataset downloaded from GitHub
│   └── processed/             # Processed dataset (training, test, validation)
├── models/                    # Trained models and evaluation results
├── src/                       # Source code
│   ├── data_preparation.py    # Data preparation script
│   ├── model_training.py      # Model training script
│   ├── user_interface.py      # User interface
│   ├── github_upload.py       # GitHub upload script
│   ├── requirements.txt       # Required packages
│   └── start.sh               # Startup script
├── time_report.md             # Time tracking report
└── README.md                  # This file
```

## Installation

Follow these steps to run the project:

1. Clone the repository:
```bash
git clone https://github.com/burakkurt07/blood-cell-detection.git
cd blood-cell-detection
```

2. Install the required packages:
```bash
pip install -r src/requirements.txt
```

3. Run the startup script:
```bash
bash src/start.sh
```

## Data Preparation

In the data preparation phase, datasets downloaded from Kaggle and GitHub were combined and processed. This process includes:

1. Loading images and extracting labels
2. Cropping cell images
3. Splitting the dataset into training (70%), test (15%), and validation (15%)
4. Visualizing the dataset distribution

Dataset distribution:
- RBC (Red Blood Cells): 6392 images
- WBC (White Blood Cells): 475 images
- Platelets: 361 images

## Model Training

Model training was performed using a transfer learning approach. The MobileNetV2 architecture was used as a base and customized for blood cell classification.

Training process:
1. Loading the MobileNetV2 base model (with ImageNet weights)
2. Adding classification layers
3. First training phase (base model frozen)
4. Fine-tuning phase (last layers of the base model trainable)
5. Model evaluation and optimization

## User Interface

The user interface allows you to detect blood cells using the trained model. The interface includes the following features:

1. Image selection and loading
2. Automatic cell type detection
3. Bar chart showing prediction probabilities
4. Detailed display of detection results

## Performance Evaluation

The model was evaluated on the test dataset and the following metrics were obtained:

- Accuracy: 99.17%
- Precision: 98.92%
- Recall: 99.05%
- F1 Score: 98.98%

## Time Tracking

The time spent on each phase of the project has been tracked in detail. The time tracking report can be found in the `time_report.md` file.

## Future Work

- Improving model performance with larger and more diverse datasets
- Developing versions that can run on mobile devices using lighter models
- Adding additional features such as cell counting and morphology analysis
- Developing a web-based interface to provide remote access

## Contributors

- Burak Kurt - Project Developer

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

You can contact us through GitHub for questions or feedback.

---

*This project demonstrates the effectiveness of using modern computer vision and deep learning techniques for blood cell detection. It can be used as an auxiliary tool in medical diagnosis and research.*
