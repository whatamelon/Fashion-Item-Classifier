"""
Tests for the FashionClassifier model.
"""

import pytest
import numpy as np
from fashion_classifier.model import FashionClassifier


class TestFashionClassifier:
    """Test cases for FashionClassifier class."""
    
    def test_initialization(self):
        """Test model initialization."""
        classifier = FashionClassifier(input_shape=(784,), num_classes=10)
        assert classifier.input_shape == (784,)
        assert classifier.num_classes == 10
        assert classifier.model is None
        assert classifier.history is None
    
    def test_build_model(self):
        """Test model building."""
        classifier = FashionClassifier()
        classifier.build_model(hidden_layers=[64, 32])
        
        assert classifier.model is not None
        assert len(classifier.model.layers) == 3  # 2 hidden + 1 output
        
        # Check layer configurations
        assert classifier.model.layers[0].units == 64
        assert classifier.model.layers[1].units == 32
        assert classifier.model.layers[2].units == 10
    
    def test_build_model_with_dropout(self):
        """Test model building with dropout."""
        classifier = FashionClassifier()
        classifier.build_model(hidden_layers=[64], dropout_rate=0.2)
        
        assert classifier.model is not None
        # Should have more layers due to dropout
        assert len(classifier.model.layers) > 2
    
    def test_predict_without_training(self):
        """Test that prediction fails without training."""
        classifier = FashionClassifier()
        classifier.build_model()
        
        X_test = np.random.random((10, 784))
        
        with pytest.raises(ValueError, match="Model not trained"):
            classifier.predict(X_test)
    
    def test_evaluate_without_training(self):
        """Test that evaluation fails without training."""
        classifier = FashionClassifier()
        classifier.build_model()
        
        X_test = np.random.random((10, 784))
        y_test = np.random.random((10, 10))
        
        with pytest.raises(ValueError, match="Model not trained"):
            classifier.evaluate(X_test, y_test)
    
    def test_save_model_without_training(self):
        """Test that saving fails without training."""
        classifier = FashionClassifier()
        
        with pytest.raises(ValueError, match="No model to save"):
            classifier.save_model("test_model")
    
    def test_get_model_summary(self):
        """Test getting model summary."""
        classifier = FashionClassifier()
        
        # Before building
        summary = classifier.get_model_summary()
        assert "No model built yet" in summary
        
        # After building
        classifier.build_model()
        summary = classifier.get_model_summary()
        assert "Model:" in summary
        assert "Total params:" in summary 