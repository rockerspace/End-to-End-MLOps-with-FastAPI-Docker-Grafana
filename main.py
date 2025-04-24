from fastapi import FastAPI, HTTPException
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from pydantic import BaseModel
import joblib
import numpy as np

# Create FastAPI app instance
app = FastAPI()

# Define input data format for prediction
class PredictionInput(BaseModel):
    input: list

# Load model at startup
@app.on_event("startup")
async def load_model():
    global model
    try:
        # Load the model from the specified path
        model = joblib.load("src/models/model.pkl")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Model loading failed")

# Prediction endpoint
@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Ensure input data is in the correct shape for the model
        input_array = np.array(input_data.input).reshape(1, -1)
        
        # Make the prediction using the loaded model
        prediction = model.predict(input_array)
        
        # Return the prediction result
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

# Root endpoint (to check if the app is running)
@app.get("/")
async def root():
    return {"message": "FastAPI app is running!"}
@app.get("/")
def read_root():
    return {"message": "FastAPI app is running!"}

# Add this at the bottom of your file
Instrumentator().instrument(app).expose(app)

REQUEST_COUNT = Counter("predict_requests_total", "Total number of prediction requests")

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict(input_data: InputData):
    REQUEST_COUNT.inc()
    ...