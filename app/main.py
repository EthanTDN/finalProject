# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

from fastapi.middleware.cors import CORSMiddleware   # <<< NEW

# load the trained model at startup
MODEL_PATH = os.getenv("MODEL_PATH", "models/diabetes_model.pkl")
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
FEATURES = artifact["features"]

app = FastAPI(
    title="Diabetes Prediction API",
    description=(
        "Educational diabetes risk prediction model. "
    ),
    version="1.0.0"
)

# Allow frontend (http://localhost:5500) to call the API
origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],    
    allow_headers=["*"],
)

class PatientData(BaseModel):
    Pregnancies: float = Field(..., ge=0)
    Glucose: float = Field(..., ge=0)
    BloodPressure: float = Field(..., ge=0)
    SkinThickness: float = Field(..., ge=0)
    Insulin: float = Field(..., ge=0)
    BMI: float = Field(..., ge=0)
    DiabetesPedigreeFunction: float = Field(..., ge=0)
    Age: float = Field(..., ge=0)

class PredictionResponse(BaseModel):
    diabetic_probability: float
    predicted_class: int  # 0 = non-diabetic, 1 = diabetic

@app.get("/")
def root():
    return {
        "message": "Diabetes prediction API"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    x = np.array([[getattr(patient, f) for f in FEATURES]])
    prob_diabetic = float(model.predict_proba(x)[0][1])
    predicted_class = int(model.predict(x)[0])
    return PredictionResponse(
        diabetic_probability=prob_diabetic,
        predicted_class=predicted_class
    )

