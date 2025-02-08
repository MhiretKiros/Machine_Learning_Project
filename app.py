import streamlit as st
import requests

# FastAPI endpoint
API_URL = "https://machine-learning-project-14.onrender.com/predict/"

# Streamlit App Layout
st.title("🌍 Travel Destination Recommendation")

# Create input form
with st.form("travel_form"):
    destination_type = st.selectbox("🏝️ Destination Type", 
                                    ["Mountains", "Countryside", "Beach", "Historical Sites", "City", "Island", "Other"])
    
    trip_purpose = st.selectbox("🎯 Trip Purpose", 
                                 ["Adventure", "Family Vacation", "Honeymoon", "Spiritual Retreat", "Other"])
    
    budget_range = st.selectbox("💰 Budget Range", 
                                ["Mid-Range", "Budget-Friendly", "Luxury"])
    
    travel_season = st.selectbox("🌦️ Travel Season", 
                                 ["Spring", "Summer", "Autumn", "Winter"])
    
    travel_duration = st.selectbox("⏳ Travel Duration", 
                                   ["A Month", "3-days", "Weekend", "Long(+month)"])
    
    accommodation_preference = st.selectbox("🏨 Accommodation Preference", 
                                            ["Camping", "Hotel", "Hostel", "Resort", "Apartment"])
    
    interest = st.selectbox("🎭 Interest", 
                            ["Culture", "Relaxation", "Adventure", "Nature"])
    
    food_interests = st.selectbox("🍽️ Food Interests", 
                                  ["Vegetarian", "Non-Vegetarian", "Vegan"])
    
    travel_type_group = st.selectbox("🚶‍♂️ Travel Type / Group", 
                                     ["Friends Group", "Solo Traveler", "Couples", "Family with Kids", "Business Group"])

    # Submit Button
    submit_button = st.form_submit_button("Get Recommendation")

# Handle form submission
if submit_button:
    input_data = {
        "destination_type": destination_type,
        "trip_purpose": trip_purpose,
        "budget_range": budget_range,
        "travel_season": travel_season,
        "travel_duration": travel_duration,
        "accommodation_preference": accommodation_preference,
        "interest": interest,
        "food_interests": food_interests,
        "travel_type_group": travel_type_group
    }

    try:
        response = requests.post(API_URL, json=input_data)
        if response.status_code == 200:
            predicted_destination = response.json()["predicted_destination"]
            st.success(f"🏆 Recommended Travel Destination: {predicted_destination}")
        else:
            st.error(f"⚠️ Error: {response.json()['detail']}")
    except Exception as e:
        st.error(f"❌ Failed to connect to API: {str(e)}")