"""
Utility functions and constants for the Fashion Item Classifier.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for terminal environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Label mapping for Fashion MNIST

LABEL_MAP = {
    0: "10-10-11",
    1: "10-10-12",
    2: "10-10-13",
    3: "10-10-14",
    4: "10-10-15",
    5: "10-20-21",
    6: "10-20-22",
    7: "10-20-23",
    8: "10-30-31",
    9: "10-30-32",
    10: "10-50-00",
    11: "20-10-11",
    12: "20-10-12",
    13: "20-10-13",
    14: "20-10-14",
    15: "20-10-15",
    16: "20-10-16",
    17: "20-20-21",
    18: "20-20-23",
    19: "20-20-24",
    20: "20-30-31",
    21: "20-30-32",
    22: "20-30-33",
    23: "20-40-41",
    24: "20-40-42",
    25: "20-51-51",
    26: "20-51-52",
    27: "20-51-53",
    28: "20-51-54",
    29: "20-51-55",
    30: "20-52-00",
    31: "20-53-00"
}

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path="confusion_matrix.png"):
    """
    Plot confusion matrix with seaborn.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_predictions(images, y_true, y_pred, num_samples=10, save_path="predictions.png"):
    """
    Plot sample predictions with actual and predicted labels.
    
    Args:
        images: Image data (reshaped to 28x28)
        y_true: True labels
        y_pred: Predicted labels
        num_samples: Number of samples to plot
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    for i in range(min(num_samples, len(images))):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"True: {LABEL_MAP[y_true[i]]}\nPred: {LABEL_MAP[y_pred[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Predictions plot saved to {save_path}")

def print_classification_report(y_true, y_pred):
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=list(LABEL_MAP.values())))

def get_label_name(label_id):
    """
    Get label name from label ID.
    
    Args:
        label_id: Label ID (0-31)
        
    Returns:
        Label name as string
    """
    return LABEL_MAP.get(label_id, f"Unknown ({label_id})") 