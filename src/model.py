from sklearn.ensemble import RandomForestRegressor

def get_model():
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    return model
