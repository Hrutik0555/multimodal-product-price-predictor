import joblib
from src.model import get_model

def train_model(X_train, y_train):
    model = get_model()

    model.fit(X_train, y_train)

    joblib.dump(model, "models/model.pkl")

    return model
