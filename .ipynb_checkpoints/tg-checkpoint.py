from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model and label encoders
model = joblib.load('travel_recommendation_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Define FastAPI app
app = FastAPI()

# Input model
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

# Define prediction route
@app.post("/predict")
def predict_travel_destination(input_data: TravelInput):
    # Prepare the input data
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

    # Encode input data
    new_data_encoded = {}
    categorical_cols = ['Destination Type', 'Trip Purpose', 'Budget Range', 'Travel Season', 'Travel Duration', 
                        'Accommodation Preference', 'Interest', 'Food Interests', 'Travel Type / Group']
    for col in categorical_cols:
        if col in label_encoders:
            if new_data[col] in label_encoders[col].classes_:
                new_data_encoded[col] = label_encoders[col].transform([new_data[col]])[0]
            else:
                return {"error": f"'{new_data[col]}' is an unseen category for '{col}'."}

    # Make prediction
    new_data_encoded = pd.DataFrame([new_data_encoded])
    prediction = model.predict(new_data_encoded)
    predicted_destination = label_encoders['Country_Info'].inverse_transform(prediction)[0]
    return {"predicted_destination": predicted_destination}
