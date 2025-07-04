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
    0: "T-shirt/top",
    1: "Trouser", 
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
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
        label_id: Label ID (0-9)
        
    Returns:
        Label name as string
    """
    return LABEL_MAP.get(label_id, f"Unknown ({label_id})") 