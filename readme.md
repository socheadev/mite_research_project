# MITE Research Project

For this research project based on image classification of harmful content using cnn with transfer learning.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Configuration](#configuration)
- [Model Training](#model-training)
- [Data Processing](#data-processing)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

Ensure you have the necessary dependencies installed for image classification and model training. The required libraries
include TensorFlow, Keras, NumPy, OpenCV, Matplotlib, Plotly, and Scikit-learn.

```pip install tensorflow keras numpy opencv-python matplotlib plotly scikit-learn```

## Usage

Follow the steps below to train the model, monitor training progress, and evaluate performance.

TensorBoard is used to monitor training progress and visualize metrics such as loss, accuracy, precision, and recall. Follow these steps to run TensorBoard:

```tensorboard --logdir=output/log/{version}```


## Configuration

The project includes configurable paths for dataset storage, model output, logging, and plots. The dataset is structured
in labeled folders, and hyperparameters such as image size, batch size, learning rate, and epochs are predefined.

## Data Processing

The dataset is loaded from structured directories, where images are preprocessed by resizing, normalizing, and
converting them into a format suitable for model training. The dataset is then shuffled and split into training,
validation, and testing sets.

## Model Training

The model is trained using a deep learning architecture, leveraging pretrained models for feature extraction. Training
includes data augmentation techniques, optimization strategies, and performance monitoring through logging and
callbacks. Fine-tuning is applied after the initial training phase to enhance accuracy.

## Contributing

We welcome contributions! Feel free to fork the repository, make improvements, and submit a pull request.

## License

This project is open-source and available under the specified license.

## Contact

For any inquiries or collaborations, please reach out to the project maintainers.