from fastapi import FastAPI
import uvicorn

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

app = FastAPI(title="House Price Prediction API")

# Setup templates
templates = Jinja2Templates(directory="projects/src/templates")

# Nếu bạn có static files (CSS, JS, images)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model (giả sử bạn đã train và save model)
model = joblib.load('projects/artifacts/model_trainer/model.joblib')

# Pydantic model cho validation
class HouseFeatures(BaseModel):
    date: str
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: int
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Trang chủ - Form nhập liệu"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    date: str = Form(...),
    bedrooms: int = Form(...),
    bathrooms: float = Form(...),
    sqft_living: int = Form(...),
    sqft_lot: int = Form(...),
    floors: float = Form(...),
    waterfront: int = Form(...),
    view: int = Form(...),
    condition: int = Form(...),
    grade: int = Form(...),
    sqft_above: int = Form(...),
    sqft_basement: int = Form(...),
    yr_built: int = Form(...),
    yr_renovated: int = Form(...),
    zipcode: int = Form(...),
    lat: float = Form(...),
    long: float = Form(...),
    sqft_living15: int = Form(...),
    sqft_lot15: int = Form(...)
):

    # ĐÚNG tên cột theo dữ liệu train (df.drop('log_price'))
    feature_names = [
        "date", "log_bedrooms", "bathrooms", "log_sqft_living", "log_sqft_lot", "floors",
        "waterfront", "view", "condition", "grade", "log_sqft_above", "log_sqft_basement",
        "yr_built", "is_renovated", "zipcode", "lat", "long", "log_sqft_living15", "log_sqft_lot15"
    ]

    row = {
        "date": date,
        "log_bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "log_sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": waterfront,
        "view": view,
        "condition": condition,
        "grade": grade,
        "log_sqft_above": sqft_above,
        "log_sqft_basement": sqft_basement,
        "yr_built": yr_built,
        "is_renovated": int(yr_renovated == 0),
        "zipcode": zipcode,
        "lat": lat,
        "long": long,
        "log_sqft_living15": sqft_living15,
        "log_sqft_lot15": sqft_lot15,
    }

    X = pd.DataFrame([row], columns=feature_names)
    prediction = float(model.predict(X)[0])


    context = {
        "request": request,
        "prediction": prediction,
        "date": date,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": waterfront,
        "view": view,
        "condition": condition,
        "grade": grade,
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "yr_built": yr_built,
        "yr_renovated": yr_renovated,
        "zipcode": zipcode,   # FIX: phải truyền xuống template
        "lat": lat,
        "long": long,
        "sqft_living15": sqft_living15,
        "sqft_lot15": sqft_lot15
    }

    return templates.TemplateResponse("result.html", context)



# API endpoint cho JSON response (nếu cần)
@app.post("/api/predict")
async def predict_api(features: HouseFeatures):
    """API endpoint trả về JSON"""
    
    # Chuẩn bị features
    feature_array = np.array([[
        features.bedrooms, features.bathrooms, features.sqft_living,
        features.sqft_lot, features.floors, features.waterfront,
        features.view, features.condition, features.grade,
        features.sqft_above, features.sqft_basement, features.yr_built,
        features.yr_renovated, features.zipcode, features.lat,
        features.long, features.sqft_living15, features.sqft_lot15
    ]])
    
    # Prediction
    prediction = model.predict(feature_array)[0]
    
    return {
        "status": "success",
        "prediction": float(prediction),
        "currency": "USD",
        "features": features.dict()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "House Price Prediction API"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
