"""
Tests for utility functions.
"""

import pytest
import numpy as np
from fashion_classifier.utils import LABEL_MAP, get_label_name


class TestUtils:
    """Test cases for utility functions."""
    
    def test_label_map(self):
        """Test that LABEL_MAP contains all expected labels."""
        expected_labels = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]

        assert len(LABEL_MAP) == 32
        for i, label in enumerate(expected_labels):
            assert LABEL_MAP[i] == label
    
    def test_get_label_name_valid(self):
        """Test get_label_name with valid label IDs."""
        assert get_label_name(0) == "T-shirt/top"
        assert get_label_name(5) == "Sandal"
        assert get_label_name(9) == "Ankle boot"
    
    def test_get_label_name_invalid(self):
        """Test get_label_name with invalid label IDs."""
        assert get_label_name(32) == "Unknown (32)"
        assert get_label_name(-1) == "Unknown (-1)"
        assert get_label_name(100) == "Unknown (100)"
    
    def test_get_label_name_edge_cases(self):
        """Test get_label_name with edge cases."""
        # Test with None (should raise TypeError)
        with pytest.raises(TypeError):
            get_label_name(None)
        
        # Test with string (should work if it can be converted)
        assert get_label_name("0") == "T-shirt/top"
        
        # Test with float
        assert get_label_name(0.0) == "T-shirt/top" 