# MNIST Handwritten Digit Classification

This project implements an Artificial Neural Network (ANN) to classify handwritten digits from the famous MNIST dataset.

## Performance
- **Accuracy:** 97.6%
- **Loss:** 0.0763

## Technologies Used
- **Python**
- **Keras / TensorFlow**
- **Matplotlib** (for visualization)

## How to Run
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run `main.py` to see the results.

## Model Architecture
The model consists of 3 Dense layers:
- Input layer: 512 units (ReLU)
- Hidden layer: 256 units (Tanh)
- Output layer: 10 units (Softmax)
