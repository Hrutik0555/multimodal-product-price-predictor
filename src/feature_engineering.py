from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_features(text_data):
    vectorizer = TfidfVectorizer(max_features=5000)

    features = vectorizer.fit_transform(text_data)

    return features, vectorizer
