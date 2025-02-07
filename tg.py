import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
import os

print(os.getcwd())  # Show the working directory for debugging

app = FastAPI()

# Add CORS middleware to allow requests from different origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; change this for security in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Load the trained model and encoders
try:
    model = joblib.load("travel_recommendation_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading model or encoders: {e}")

# Define the input data structure
class TravelInput(BaseModel):
    destination_type: str
    trip_purpose: str
    budget_range: str
    travel_season: str
    travel_duration: str
    accommodation_preference: str
    interest: str
    food_interests: str
    travel_type_group: str

@app.post("/predict/")
async def predict_travel_destination(input_data: TravelInput):
    try:
        # Prepare the data for prediction
        new_data = {
            "Destination Type": input_data.destination_type,
            "Trip Purpose": input_data.trip_purpose,
            "Budget Range": input_data.budget_range,
            "Travel Season": input_data.travel_season,
            "Travel Duration": input_data.travel_duration,
            "Accommodation Preference": input_data.accommodation_preference,
            "Interest": input_data.interest,
            "Food Interests": input_data.food_interests,
            "Travel Type / Group": input_data.travel_type_group
        }

        # Encode categorical features
        new_data_encoded = {}
        for col in new_data:
            if col in label_encoders:
                if new_data[col] in label_encoders[col].classes_:
                    new_data_encoded[col] = label_encoders[col].transform([new_data[col]])[0]
                else:
                    raise HTTPException(status_code=422, detail=f"'{new_data[col]}' is an unseen category for '{col}'.")

        new_data_encoded_df = pd.DataFrame([new_data_encoded])

        # Make a prediction
        prediction = model.predict(new_data_encoded_df)
        predicted_country = label_encoders["Country_Info"].inverse_transform(prediction)[0]

        return {"predicted_destination": predicted_country}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# Run Uvicorn when the script is executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
