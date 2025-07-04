# 👕 Fashion Item Classifier

A neural network-based fashion item classifier using the Fashion MNIST dataset. This project demonstrates a complete machine learning pipeline from data loading to model evaluation, built with modern Python tooling and best practices.

## 🚀 Features

- **Modular Architecture**: Clean, maintainable code structure with separate modules for data loading, model building, and utilities
- **Modern Python Tooling**: Uses `uv` for fast dependency management and virtual environment handling
- **Comprehensive Testing**: Unit tests for all major components
- **Easy-to-Use API**: Simple interface for training and evaluating models
- **Rich Visualizations**: Confusion matrices, prediction plots, and detailed classification reports
- **Jupyter Notebooks**: Interactive demos and tutorials

## 📦 Installation

This project uses `uv` for dependency management. Make sure you have `uv` installed:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup the project:

```bash
# Clone the repository
git clone <repository-url>
cd Fashion-Item-Classifier

# Initialize uv environment
uv init --python 3.11

# Install dependencies
uv add pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter

# Install development dependencies
uv add --dev black isort flake8 pytest

# Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows
```

## 🏗️ Project Structure

```
Fashion-Item-Classifier/
├── fashion_classifier/          # Main package
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── model.py                # Neural network model
│   └── utils.py                # Utility functions and constants
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── test_model.py
│   └── test_utils.py
├── notebooks/                  # Jupyter notebooks
│   ├── GroupNo_12.ipynb        # Original notebook
│   └── fashion_classification_demo.ipynb
├── data/                       # Data files (not in git)
├── main.py                     # Main execution script
├── pyproject.toml             # Project configuration
├── .gitignore
└── README.md
```

## 🧠 Model Architecture

The neural network consists of:

- **Input Layer**: 784 neurons (flattened 28×28 images)
- **Hidden Layer 1**: 128 neurons with ReLU activation
- **Hidden Layer 2**: 64 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation

## 📊 Dataset

The project uses the **Fashion MNIST** dataset with 10 clothing categories:

| Label | Description |
| ----- | ----------- |
| 0     | T-shirt/top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle boot  |

## 🚀 Quick Start

### Using the Main Script

```bash
# Make sure you have the data file
# fashion-mnist_test.csv should be in the project root

# Run the complete pipeline
python main.py
```

### Using Python API

```python
from fashion_classifier import FashionClassifier, DataLoader
from fashion_classifier.utils import plot_confusion_matrix, plot_predictions

# Load and preprocess data
data_loader = DataLoader()
data_loader.load_from_csv('fashion-mnist_test.csv')
data_loader.preprocess_data()

# Get training data
X_train, y_train = data_loader.get_training_data()
X_test, y_test = data_loader.get_test_data()

# Build and train model
classifier = FashionClassifier()
classifier.build_model(hidden_layers=[128, 64])
classifier.train(X_train, y_train, epochs=10)

# Evaluate model
results = classifier.evaluate(X_test, y_test)
print(f"Test Accuracy: {results['test_accuracy']:.4f}")

# Visualize results
y_pred = results['predictions']
y_true = np.argmax(y_test, axis=1)
plot_confusion_matrix(y_true, y_pred)
```

### Using Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook notebooks/
```

Then open `fashion_classification_demo.ipynb` for an interactive tutorial.

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_model.py
```

## 🔧 Development

### Code Formatting

```bash
# Format code with black
uv run black .

# Sort imports with isort
uv run isort .

# Check code style with flake8
uv run flake8 .
```

### Adding Dependencies

```bash
# Add production dependency
uv add package-name

# Add development dependency
uv add --dev package-name
```

## 📈 Performance

Typical performance metrics on the Fashion MNIST test set:

- **Accuracy**: ~85-90%
- **Precision**: ~85-90%
- **Recall**: ~85-90%
- **F1-Score**: ~85-90%

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Fashion MNIST dataset creators
- TensorFlow/Keras team for the deep learning framework
- The open-source Python community

## 📞 Support

If you encounter any issues or have questions, please:

1. Check the existing issues
2. Create a new issue with detailed information
3. Include your Python version and environment details

---

**Happy classifying! 🎉**
# Fashion-Item-Classifier
