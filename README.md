# weather-forecasting-using-deep-learning
Weather Forecasting Using Long Short Term Memory(LSTM) With Artificial Neural Networks
## Overview
This project utilizes a Long Short-Term Memory (LSTM) neural network to predict temperature based on the Jena Climate dataset. The model is implemented using TensorFlow and Keras.

## Dataset
The Jena Climate dataset used in this project can be found [here](https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip). It contains climate data from 2009 to 2016.

## Getting Started

### Prerequisites
- Python 3.x
- TensorFlow
- Pandas
- Matplotlib
- NumPy

### Installation
```bash
pip install tensorflow pandas matplotlib numpy


Creating a README file for your project is a good practice to provide essential information and instructions for others who may want to use or contribute to your project. Below is a template for a README file for your project. You can customize it based on your specific project details:

markdown
Copy code
# Jena Climate LSTM Model

## Overview
This project utilizes a Long Short-Term Memory (LSTM) neural network to predict temperature based on the Jena Climate dataset. The model is implemented using TensorFlow and Keras.

## Dataset
The Jena Climate dataset used in this project can be found [here](https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip). It contains climate data from 2009 to 2016.

## Getting Started

### Prerequisites
- Python 3.x
- TensorFlow
- Pandas
- Matplotlib
- NumPy

### Installation
```bash
pip install tensorflow pandas matplotlib numpy
Usage
Clone the repository:

bash
Copy code
git clone <repository-url>
cd <repository-directory>
Download the Jena Climate dataset:

bash
Copy code
# Run the provided script to download and extract the dataset
python download_dataset.py
Run the main script:

bash
Copy code
python main.py
Project Structure
main.py: Main script to train the LSTM model and make predictions.
download_dataset.py: Script to download and extract the Jena Climate dataset.
model1/: Directory to save the trained model checkpoint.
Model Architecture
The implemented LSTM model consists of:

Input layer: 5 time steps, 1 feature
LSTM layer with 64 units
Dense layer with 8 units and ReLU activation
Output layer with 1 unit and linear activation
Training
The model is trained for 10 epochs using the Adam optimizer with a learning rate of 0.0001.

Results
The training predictions and actual values are visualized in the Train Predictions vs Actuals plot.

License
This project is licensed under the MIT License - see the LICENSE file for details.


