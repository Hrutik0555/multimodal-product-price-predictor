# Multimodal Product Price Predictor

An end-to-end Machine Learning and Deep Learning pipeline designed to predict e-commerce product prices using multimodal data (textual descriptions and product images).

## 📌 Overview

Accurate price prediction in e-commerce requires evaluating both textual metadata (titles, specifications, descriptions) and visual cues (product design, brand aesthetics, packaging). 

The **Multimodal Product Price Predictor** combines text features (extracted using NLP transformers/embeddings) and image features (extracted via Vision models like ResNet, EfficientNet, or CLIP) into a unified architecture to perform robust regression for price prediction.

---

## 📁 Repository Structure

```
multimodal-product-price-predictor/
│
├── data/                       # Directory for raw and preprocessed datasets
│   ├── .gitkeep
│   └── README.md               # Guidelines for dataset structure and formatting
│
├── models/                     # Directory for saved model checkpoints & weights
│   └── .gitkeep
│
├── notebooks/                  # Jupyter notebooks for EDA and experimentation
│   └── exploration.ipynb
│
├── outputs/                    # Log files, metrics, and generated plots
│   └── .gitkeep
│
├── src/                        # Modular source code
│   ├── __init__.py             # Package initialization
│   ├── config.py               # Global configuration, paths, and hyperparameters
│   ├── preprocessing.py        # Data cleaning and tabular pipeline utilities
│   ├── text_features.py       # Text embedding generation and feature extraction
│   ├── image_utils.py          # Image loading, transformations, and feature extraction
│   ├── models.py               # PyTorch/TensorFlow multimodal neural network models
│   ├── train.py                # Model training and validation loops
│   ├── evaluate.py             # Model evaluation metrics (RMSE, MAE, MAPE, $R^2$)
│   └── predict.py              # Inference script for new product samples
│
├── .gitignore                  # Git ignore directives
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

---

## ✨ Features

- **Multimodal Integration:** Fuses textual features (TF-IDF / BERT embeddings) and image embeddings (CNN / Vision Transformers) with tabular attributes.
- **Flexible Pipeline:** Modular modules for data preprocessing, feature engineering, training, and evaluation.
- **Config-Driven:** Easily tune hyperparameters, paths, and model architectures in `src/config.py`.
- **Inference Ready:** Includes `predict.py` for direct evaluation and inference on unseen product items.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Hrutik0555/multimodal-product-price-predictor.git
cd multimodal-product-price-predictor
```

### 2. Create a Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start & Usage

### 1. Prepare Data
Place your raw datasets inside the `data/` directory. See [`data/README.md`](./data/README.md) for expected schema formatting (e.g., product text fields, image paths, and target price $y$).

### 2. Configure Experiment Settings
Adjust hyperparameters, model configurations, and file paths in `src/config.py`:

```python
# Example setup in src/config.py
TEXT_MODEL_NAME = "bert-base-uncased"
IMAGE_MODEL_NAME = "resnet50"
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10
```

### 3. Train the Model

Run the training pipeline:

```bash
python -m src.train
```

### 4. Evaluate the Model

Evaluate performance metrics (RMSE, MAE, $R^2$) on test data:

```bash
python -m src.evaluate
```

### 5. Run Predictions / Inference

To run inference on new product inputs:

```bash
python -m src.predict --image_path path/to/image.jpg --description "Product Description"
```

---

## 📊 Architecture Overview

1. **Text Pipeline:** Textual descriptions pass through `src/text_features.py` to extract contextual sequence embeddings $E_{text}$.
2. **Vision Pipeline:** Images pass through `src/image_utils.py` for scaling, normalization, and feature extraction $E_{img}$.
3. **Multimodal Fusion:** Features $E_{text}$ and $E_{img}$ are concatenated and passed through dense classification layers defined in `src/models.py`.
4. **Target:** Regresses continuous target value representing the predicted price $\hat{y}$.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

1. Fork the Repository
2. Create your Feature Branch (`git checkout -b feature/NewFeature`)
3. Commit your Changes (`git commit -m 'Add NewFeature'`)
4. Push to the Branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
