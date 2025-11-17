# imports

# api library that exposes REST API endpoints
from fastapi import FastAPI

# allows for request data validation and formatting for inputs
from pydantic import BaseModel, Field

# library that lets us store the trained model into a pipeline
import joblib

# allows for the input data to be converted into feature vectors
import numpy as np

# used for built in function to get environmental variables
import os

# library that solved problem with browser security blocking communication between front end and backend ports
from fastapi.middleware.cors import CORSMiddleware

# get the model path from the environment variable
MODEL_PATH = os.getenv("MODEL_PATH", "models/diabetes_model.pkl")

# load the trained model artifact, which has the model itself and the features
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
FEATURES = artifact["features"]

# basic initialization of FastAPI
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

# Apply the CORS middleware which allows requests from the frontend between browsers
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],    
    allow_headers=["*"],
)

# this shows the structure based on the incoming inputs for the prediction request. The only constraint on the inputs is that they are non negative

class PatientData(BaseModel):
    Pregnancies: float = Field(..., ge=0)
    Glucose: float = Field(..., ge=0)
    BloodPressure: float = Field(..., ge=0)
    SkinThickness: float = Field(..., ge=0)
    Insulin: float = Field(..., ge=0)
    BMI: float = Field(..., ge=0)
    DiabetesPedigreeFunction: float = Field(..., ge=0)
    Age: float = Field(..., ge=0)

# uses base model to define the response structure that is returned by the /predict endpoint. the predicted class is either 0 = non-diabetic, 1 = diabetic
class PredictionResponse(BaseModel):
    diabetic_probability: float
    predicted_class: int  

# root endpoint message
@app.get("/")
def root():
    return {
        "message": "Diabetes prediction API"
    }

# processes the incoming patient data, formats it, and makes the prediction then returns the results
@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    
    # covert the incoming data into an ordered array for the model
    x = np.array([[getattr(patient, f) for f in FEATURES]])
    
    # predict the probability percentage
    prob_diabetic = float(model.predict_proba(x)[0][1])
    
    # predict the final label
    predicted_class = int(model.predict(x)[0])
    
    # return the result in the structure defined above
    return PredictionResponse(
        diabetic_probability=prob_diabetic,
        predicted_class=predicted_class
    )

