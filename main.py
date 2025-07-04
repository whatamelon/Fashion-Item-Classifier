#!/usr/bin/env python3
"""
Main script for Fashion Item Classification.

This script demonstrates the complete pipeline for training and evaluating
a neural network classifier on the Fashion MNIST dataset.
"""

import os
import sys
from fashion_classifier import FashionClassifier, DataLoader
from fashion_classifier.utils import (
    plot_confusion_matrix, 
    plot_predictions, 
    print_classification_report,
    LABEL_MAP
)


def main():
    """Main function to run the fashion classification pipeline."""
    
    print("=" * 60)
    print("FASHION ITEM CLASSIFIER")
    print("=" * 60)
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Check if data file exists
    data_file = "fashion-mnist_test.csv"
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found.")
        print("Please ensure the Fashion MNIST CSV file is in the current directory.")
        sys.exit(1)
    
    try:
        # Load and preprocess data
        print("\n1. Loading and preprocessing data...")
        data_loader.load_from_csv(data_file)
        
        # Display data information
        data_info = data_loader.get_data_info()
        print(f"Data info: {data_info}")
        
        # Preprocess data
        data_loader.preprocess_data(test_size=0.2, random_state=42)
        
        # Get training and test data
        X_train, y_train = data_loader.get_training_data()
        X_test, y_test = data_loader.get_test_data()
        X_test_images = data_loader.get_test_images()
        
        # Initialize and build model
        print("\n2. Building neural network model...")
        classifier = FashionClassifier(input_shape=(784,), num_classes=10)
        classifier.build_model(hidden_layers=[128, 64], activation='relu')
        
        # Train the model
        print("\n3. Training the model...")
        classifier.train(
            X_train, y_train,
            epochs=10,
            batch_size=128,
            validation_split=0.1
        )
        
        # Evaluate the model
        print("\n4. Evaluating the model...")
        results = classifier.evaluate(X_test, y_test)
        
        # Print evaluation results
        print(f"\nTest Accuracy: {results['test_accuracy']:.4f}")
        print(f"Test Loss: {results['test_loss']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        
        # Get predictions for visualization
        y_pred = results['predictions']
        y_true = [i for i in range(len(y_test))]  # Convert back to class indices
        
        # Plot confusion matrix
        print("\n5. Generating visualizations...")
        plot_confusion_matrix(y_true, y_pred)
        
        # Plot sample predictions
        plot_predictions(X_test_images, y_true, y_pred, num_samples=10)
        
        # Print detailed classification report
        print_classification_report(y_true, y_pred)
        
        # Save the model
        print("\n6. Saving the trained model...")
        model_path = "fashion_classifier_model"
        classifier.save_model(model_path)
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
