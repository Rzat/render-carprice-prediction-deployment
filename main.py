import pickle

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd

app = FastAPI()

templates = Jinja2Templates(directory="templates")

model = joblib.load("models/model.pkl")


@app.get("/health")
def health_check():
    return {"status": "healthy: Server is up and running"}

@app.get("/metadata")
def get_metadata():
    return modelInspection(model)

@app.post("/predict")
def predict(payload_dict: dict):
    input_data = pd.DataFrame([{
        "Manufacturer": payload_dict["Manufacturer"],
     "Model": payload_dict["Model"],
      "Fuel type": payload_dict["Fuel type"], 
      "Engine size": payload_dict["Engine size"],
      "Mileage": payload_dict["Mileage"],
      "Year of manufacture": payload_dict["Year of manufacture"],

      "age": 2025 - payload_dict["Year of manufacture"],
      "mileage_per_year": payload_dict["Mileage"] / (2025 - payload_dict["Year of manufacture"]),
      "vintage": 1 if 2025 - payload_dict["Year of manufacture"] >= 20 else 0,
      }])
    
    prediction = model.predict(input_data)
    
    return {"Predicted Price: ": round(float(prediction[0]),2)}


@app.get("/", response_class=HTMLResponse)
def render_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})


@app.post("/ui/predict", response_class=HTMLResponse)
def ui_predict(request: Request,
               Manufacturer: str = Form(...),
               Model: str = Form(...),
               Fuel_type: str = Form(...),
               Engine_size: float = Form(...),
               Mileage: float = Form(...),
               Year_of_manufacture: int = Form(...)):
    """Handle form submission and render prediction."""
    input_data = pd.DataFrame([{
        "Manufacturer": Manufacturer,
        "Model": Model,
        "Fuel type": Fuel_type,
        "Engine size": Engine_size,
        "Mileage": Mileage,
        "Year of manufacture": Year_of_manufacture,
        "age": 2025 - Year_of_manufacture,
        "mileage_per_year": Mileage / (2025 - Year_of_manufacture) if (2025 - Year_of_manufacture) != 0 else 0,
        "vintage": 1 if 2025 - Year_of_manufacture >= 20 else 0,
    }])

    prediction = model.predict(input_data)
    predicted_price = round(float(prediction[0]), 2)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": predicted_price,
        "form_values": {
            "Manufacturer": Manufacturer,
            "Model": Model,
            "Fuel_type": Fuel_type,
            "Engine_size": Engine_size,
            "Mileage": Mileage,
            "Year_of_manufacture": Year_of_manufacture,
        }
    })


def modelInspection(model):
   #print(model.steps[0])
   print(model.steps[1])
   return {
        "type": type(model).__name__,
        "Required features": list(model.feature_names_in_),
        "Number of features": model.n_features_in_,
        "pipeline_info": {
            "step_2": f"{model.steps[1][0]} ({type(model.steps[1][1]).__name__})"
        }
    }