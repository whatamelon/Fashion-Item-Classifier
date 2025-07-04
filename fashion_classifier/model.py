"""
Neural network model for Fashion MNIST classification.
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class FashionClassifier:
    """
    Neural network classifier for Fashion MNIST dataset.
    """
    
    def __init__(self, input_shape=(784,), num_classes=10):
        """
        Initialize the FashionClassifier.
        
        Args:
            input_shape: Shape of input data (default: 784 for flattened 28x28 images)
            num_classes: Number of output classes (default: 10 for Fashion MNIST)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self, hidden_layers=[128, 64], activation='relu', dropout_rate=0.0):
        """
        Build the neural network model.
        
        Args:
            hidden_layers: List of hidden layer sizes
            activation: Activation function for hidden layers
            dropout_rate: Dropout rate (0.0 for no dropout)
        """
        self.model = Sequential()
        
        # Input layer
        self.model.add(Dense(hidden_layers[0], activation=activation, input_shape=self.input_shape))
        if dropout_rate > 0:
            from tensorflow.keras.layers import Dropout
            self.model.add(Dropout(dropout_rate))
        
        # Hidden layers
        for units in hidden_layers[1:]:
            self.model.add(Dense(units, activation=activation))
            if dropout_rate > 0:
                self.model.add(Dropout(dropout_rate))
        
        # Output layer
        self.model.add(Dense(self.num_classes, activation='softmax'))
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model built successfully!")
        self.model.summary()
        
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=10, batch_size=128, validation_split=0.1,
              callbacks=None):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of training data to use for validation
            callbacks: List of Keras callbacks
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Set up callbacks
        if callbacks is None:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
            ]
        
        print(f"Training model for {epochs} epochs...")
        
        # Train the model
        if X_val is not None and y_val is not None:
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        else:
            self.history = self.model.fit(
                X_train, y_train,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        
        print("Training completed!")
        
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities and class labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get predictions
        y_pred_probs = self.model.predict(X)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        
        return y_pred_probs, y_pred_classes
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Get predictions
        y_pred_probs, y_pred_classes = self.predict(X_test)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred_classes,
            'probabilities': y_pred_probs
        }
        
        return results
        
    def save_model(self, filepath):
        """
        Save the trained model.
        
        Args:
            filepath: Path where to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")
        
    def get_model_summary(self):
        """Get model summary as string."""
        if self.model is None:
            return "No model built yet."
        
        from io import StringIO
        summary_io = StringIO()
        self.model.summary(print_fn=lambda x: summary_io.write(x + '\n'))
        return summary_io.getvalue() 