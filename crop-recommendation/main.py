from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model and scaler
model = joblib.load("crop_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define input data structure
class CropInput(BaseModel):
    Nitrogen: float
    Phosphorus: float
    Potassium: float
    Temperature: float
    Humidity: float
    pH_Value: float
    Rainfall: float

@app.post("/predict")
def predict_crop(data: CropInput):
    # Convert input to numpy array
    input_data = np.array([[data.Nitrogen, data.Phosphorus, data.Potassium, 
                            data.Temperature, data.Humidity, data.pH_Value, 
                            data.Rainfall]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    return {"recommended_crop": prediction}