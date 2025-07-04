"""
Fashion Item Classifier

A neural network-based fashion item classifier using Fashion MNIST dataset.
"""

__version__ = "0.1.0"
__author__ = "Fashion Item Classifier Team"

from .model import FashionClassifier
from .data_loader import DataLoader
from .utils import LABEL_MAP

__all__ = ["FashionClassifier", "DataLoader", "LABEL_MAP"] 