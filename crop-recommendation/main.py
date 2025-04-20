from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

# Load model and scaler
model = joblib.load("crop_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define input schema
class CropInput(BaseModel):
    Nitrogen: float
    Phosphorus: float
    Potassium: float
    Temperature: float
    Humidity: float
    pH_Value: float
    Rainfall: float

@app.post("/predict")
async def predict(data: CropInput):
    try:
        features = np.array([[
            data.Nitrogen,
            data.Phosphorus,
            data.Potassium,
            data.Temperature,
            data.Humidity,
            data.pH_Value,
            data.Rainfall
        ]])
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]
        return {"crop": prediction}
    except Exception as e:
        return {"error": str(e)}