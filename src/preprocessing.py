from sklearn.preprocessing import LabelEncoder

def handle_missing_values(df):
    df.fillna("Unknown", inplace=True)
    return df

def encode_column(df, column):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])

    return df
