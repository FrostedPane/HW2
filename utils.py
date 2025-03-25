import pandas as pd
import torch
from models import NeuralNetwork  # Import NeuralNetwork from models.py

# Data preprocessing
def preprocess_data(data):
    features = data.iloc[:, 1:4].values
    labels = data.iloc[:, 4].values
    return features, labels

# Save model
def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)

# Load model
def load_model(file_path):
    model = NeuralNetwork()  # Initialize model
    model.load_state_dict(torch.load(file_path))
    return model
