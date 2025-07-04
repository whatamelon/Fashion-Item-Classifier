"""
Data loading and preprocessing utilities for Fashion MNIST dataset.
"""

import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


class DataLoader:
    """
    Data loader for Fashion MNIST dataset.
    """
    
    def __init__(self, data_path=None):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to the CSV data file
        """
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_test_images = None
        
    def load_from_csv(self, csv_path):
        """
        Load data from CSV file.
        
        Args:
            csv_path: Path to the CSV file
        """
        print(f"Loading data from {csv_path}")
        self.data = pd.read_csv(csv_path)
        print(f"Loaded {len(self.data)} samples with {len(self.data.columns)-1} features")
        
    def load_from_zip(self, zip_path):
        """
        Load data from ZIP file containing CSV.
        
        Args:
            zip_path: Path to the ZIP file
        """
        print(f"Extracting data from {zip_path}")
        
        # Create temporary directory for extraction
        extract_dir = "temp_extracted"
        os.makedirs(extract_dir, exist_ok=True)
        
        # Extract ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Find CSV file in extracted directory
        csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No CSV file found in the ZIP archive")
        
        csv_path = os.path.join(extract_dir, csv_files[0])
        self.load_from_csv(csv_path)
        
        # Clean up temporary directory
        import shutil
        shutil.rmtree(extract_dir)
        
    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        Preprocess the loaded data.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_from_csv() or load_from_zip() first.")
        
        print("Preprocessing data...")
        
        # Separate features and labels
        X = self.data.iloc[:, 1:].values / 255.0  # Normalize pixel values
        y = self.data.iloc[:, 0].values
        
        # Convert labels to categorical
        y_cat = to_categorical(y)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_cat, test_size=test_size, random_state=random_state
        )
        
        # Reshape test images for visualization
        self.X_test_images = self.X_test.reshape(-1, 28, 28)
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        print(f"Input shape: {self.X_train.shape[1]}")
        print(f"Number of classes: {self.y_train.shape[1]}")
        
    def get_training_data(self):
        """Get training data."""
        return self.X_train, self.y_train
        
    def get_test_data(self):
        """Get test data."""
        return self.X_test, self.y_test
        
    def get_test_images(self):
        """Get test images reshaped for visualization."""
        return self.X_test_images
        
    def get_data_info(self):
        """Get information about the loaded data."""
        if self.data is None:
            return "No data loaded"
        
        info = {
            "total_samples": len(self.data),
            "features": len(self.data.columns) - 1,
            "classes": len(self.data.iloc[:, 0].unique()),
            "class_distribution": self.data.iloc[:, 0].value_counts().to_dict()
        }
        return info 