import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
import re
import joblib
import requests
from PIL import Image
from io import BytesIO
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Configuration
DATA_DIR = "dataset"  # Update this to your dataset directory
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "predictions.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
MODEL_DIR = "models"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Helper functions for downloading images
def download_image(url, retry=3):
    """Download image from URL with retry mechanism"""
    for attempt in range(retry):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            print(f"Error downloading {url}: {e}. Attempt {attempt+1}/{retry}")
            if attempt == retry - 1:
                return None
    return None

def download_and_save_images(df, image_dir):
    """Download and save images from the dataframe"""
    os.makedirs(image_dir, exist_ok=True)
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processing image {idx}/{len(df)}")
        
        sample_id = row['sample_id']
        image_url = row['image_link']
        image_path = os.path.join(image_dir, f"{sample_id}.jpg")
        
        # Skip if image already exists
        if os.path.exists(image_path):
            continue
            
        # Download and save image
        img = download_image(image_url)
        if img:
            img.save(image_path)

# Text preprocessing
def preprocess_text(text):
    """Clean and preprocess text data"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Feature extraction from text
def extract_text_features(df):
    """Extract features from catalog content"""
    # Extract IPQ (Item Pack Quantity) if available
    df['ipq'] = df['catalog_content'].str.extract(r'IPQ:?\s*(\d+)', flags=re.IGNORECASE).astype('float')
    
    # Extract dimensions if available
    df['has_dimensions'] = df['catalog_content'].str.contains(r'\d+\s*x\s*\d+\s*x\s*\d+', flags=re.IGNORECASE).astype('int')
    
    # Extract weight if available
    df['has_weight'] = df['catalog_content'].str.contains(r'\d+\s*(kg|g|pound|lb|oz|ounce)', flags=re.IGNORECASE).astype('int')
    
    # Extract brand mentions
    df['has_brand'] = df['catalog_content'].str.contains(r'brand|manufacturer', flags=re.IGNORECASE).astype('int')
    
    # Extract material mentions
    df['has_material'] = df['catalog_content'].str.contains(r'material|made of|made from', flags=re.IGNORECASE).astype('int')
    
    # Clean text for TF-IDF
    df['clean_text'] = df['catalog_content'].apply(preprocess_text)
    
    return df

# Image feature extraction using pretrained CNN
class ImageFeatureExtractor:
    def __init__(self, model_name='resnet18'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pretrained model
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-1])  # Remove classification layer
        elif model_name == 'efficientnet':
            self.model = models.efficientnet_b0(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image_path):
        """Extract features from a single image"""
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(img_tensor)
                features = features.squeeze().cpu().numpy()
            
            return features
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return np.zeros(512)  # Return zeros for failed images
    
    def batch_extract_features(self, image_paths):
        """Extract features from multiple images"""
        features = []
        for path in image_paths:
            feat = self.extract_features(path)
            features.append(feat)
        return np.array(features)

# Product Dataset class
class ProductDataset:
    def __init__(self, csv_file, image_dir=None, transform=None, is_test=False):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test
        
        # Preprocess data
        self.preprocess()
    
    def preprocess(self):
        """Preprocess the dataset"""
        # Extract text features
        self.data = extract_text_features(self.data)
        
        # Create image paths
        if self.image_dir:
            self.data['image_path'] = self.data['sample_id'].apply(
                lambda x: os.path.join(self.image_dir, f"{x}.jpg")
            )
    
    def get_data(self):
        """Return the processed dataframe"""
        return self.data

# Model building and training
class PricePredictionModel:
    def __init__(self):
        self.text_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.image_extractor = ImageFeatureExtractor()
        self.model = None
        self.text_features = None
        self.image_features = None
    
    def extract_features(self, dataset):
        """Extract features from dataset"""
        data = dataset.get_data()
        
        # Extract text features
        print("Extracting text features...")
        self.text_vectorizer.fit(data['clean_text'])
        text_features = self.text_vectorizer.transform(data['clean_text']).toarray()
        
        # Extract numerical features
        numerical_features = data[['ipq', 'has_dimensions', 'has_weight', 'has_brand', 'has_material']].fillna(0).values
        
        # Extract image features if available
        image_features = None
        if 'image_path' in data.columns:
            print("Extracting image features...")
            image_paths = data['image_path'].tolist()
            image_features = np.zeros((len(image_paths), 512))
            
            for i, path in enumerate(image_paths):
                if os.path.exists(path):
                    image_features[i] = self.image_extractor.extract_features(path)
                if i % 100 == 0:
                    print(f"Processed {i}/{len(image_paths)} images")
        
        # Combine features
        if image_features is not None:
            combined_features = np.hstack((text_features, numerical_features, image_features))
        else:
            combined_features = np.hstack((text_features, numerical_features))
        
        return combined_features
    
    def train(self, train_dataset, val_size=0.2):
        """Train the price prediction model"""
        data = train_dataset.get_data()
        
        # Extract features
        X = self.extract_features(train_dataset)
        y = data['price'].values
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=42)
        
        print(f"Training with {X_train.shape[0]} samples, validating with {X_val.shape[0]} samples")
        
        # Train model
        print("Training model...")
        self.model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_preds = self.model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_preds)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        
        print(f"Validation MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")
        
        # Calculate SMAPE
        smape = 100 * np.mean(np.abs(val_preds - y_val) / ((np.abs(y_val) + np.abs(val_preds)) / 2))
        print(f"Validation SMAPE: {smape:.4f}%")
        
        return val_mae, val_rmse, smape
    
    def predict(self, test_dataset):
        """Generate predictions for test dataset"""
        data = test_dataset.get_data()
        
        # Extract features
        X_test = self.extract_features(test_dataset)
        
        # Generate predictions
        predictions = self.model.predict(X_test)
        
        # Ensure predictions are positive
        predictions = np.maximum(predictions, 0.01)
        
        # Create output dataframe
        output_df = pd.DataFrame({
            'sample_id': data['sample_id'],
            'price': predictions
        })
        
        return output_df
    
    def save(self, model_path):
        """Save the trained model"""
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
    
    def load(self, model_path):
        """Load a trained model"""
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")

# Main execution
def main():
    print("Starting Product Price Prediction")
    
    # Load and preprocess training data
    print("Loading training data...")
    train_dataset = ProductDataset(TRAIN_FILE, image_dir=IMAGE_DIR)
    
    # Train model
    model = PricePredictionModel()
    val_mae, val_rmse, val_smape = model.train(train_dataset)
    
    # Save model
    model.save(os.path.join(MODEL_DIR, "price_model.pkl"))
    
    # Load and preprocess test data
    print("Loading test data...")
    test_dataset = ProductDataset(TEST_FILE, image_dir=IMAGE_DIR, is_test=True)
    
    # Generate predictions
    print("Generating predictions...")
    predictions = model.predict(test_dataset)
    
    # Save predictions
    predictions.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE}")
    
    print("Done!")
if __name__ == "__main__":
    main()
