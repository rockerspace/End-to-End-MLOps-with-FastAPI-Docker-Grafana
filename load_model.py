import joblib
from fastapi import HTTPException

def load_model(model_path: str):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Model loading failed")
