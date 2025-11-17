# import libraries to train and save the model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# load the dataset
df = pd.read_csv("diabetes.csv")

# define the feature columns and target the column for training
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"

# label the features and target
X = df[FEATURES]
y = df[TARGET]

# split the data into training and testing sets using 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# build the pipeline with scaling and logistic regression
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

# train the model
pipeline.fit(X_train, y_train)

# make predictions and evaluate the model
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# save the trained model and feature list to apply to data input by the user
joblib.dump({
    "model": pipeline,
    "features": FEATURES
}, "models/diabetes_model.pkl")

print("Model saved to models/diabetes_model.pkl")
