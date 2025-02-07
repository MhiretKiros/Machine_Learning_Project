from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
import os
print(os.getcwd())  # This will show where your script is running.

app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; you can restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load the trained model and encoders
model = joblib.load('travel_recommendation_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Define the input structure
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
    # Prepare the data for prediction
    new_data = {
        'Destination Type': input_data.destination_type,
        'Trip Purpose': input_data.trip_purpose,
        'Budget Range': input_data.budget_range,
        'Travel Season': input_data.travel_season,
        'Travel Duration': input_data.travel_duration,
        'Accommodation Preference': input_data.accommodation_preference,
        'Interest': input_data.interest,
        'Food Interests': input_data.food_interests,
        'Travel Type / Group': input_data.travel_type_group
    }
    
    # Encode the features
    new_data_encoded = {}
    for col in new_data:
        if col in label_encoders:
            if new_data[col] in label_encoders[col].classes_:
                new_data_encoded[col] = label_encoders[col].transform([new_data[col]])[0]
            else:
                raise HTTPException(status_code=422, detail=f"'{new_data[col]}' is an unseen category for '{col}'.")
    
    new_data_encoded_df = pd.DataFrame([new_data_encoded])

    # Make prediction
    prediction = model.predict(new_data_encoded_df)
    predicted_country = label_encoders['Country_Info'].inverse_transform(prediction)[0]
    
    return {"predicted_destination": predicted_country}
