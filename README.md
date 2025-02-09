Personalized Travel Destination Recommender

Overview
The Personalized Travel Destination Recommender is an AI-powered system designed to help users find vacation destinations tailored to their unique preferences. By analyzing inputs such as budget, travel season, personal interests, and accommodation preferences, the system provides personalized destination suggestions. The backend uses FastAPI for real-time predictions, while the user-friendly interface is built with Streamlit.

Key Features
•	Personalized Recommendations: Offers travel destination suggestions based on user preferences like budget, trip purpose, accommodation type, and interests.
•	Real-Time Predictions: FastAPI backend ensures quick responses for travel recommendations.
•	Interactive Web Interface: Built with Streamlit for an intuitive and user-friendly experience.
•	Comprehensive Insights: Recommends not only destinations but also transportation and accommodation options.
•	Scalable: Easily extendable to incorporate additional features and models.

Technology Stack
•	Machine Learning: RandomForestClassifier and RandomForestRegressor from scikit-learn for prediction tasks.
•	Backend: FastAPI for building the API to serve the model and handle real-time user requests.
•	Frontend: Streamlit for creating the web interface.
•	Serialization: Joblib for saving and loading the trained models.
•	Data Processing: Pandas and scikit-learn for data manipulation, preprocessing, and encoding.

Installation Guide
Prerequisites
Make sure you have the following installed:
•	Python 3.7 or higher
•	pip (Python's package installer)
•	A code editor (e.g., VS Code, PyCharm)

Steps to Install and Run Locally
1.	Clone the Repository:
bash
CopyEdit
git clone https://github.com/MhiretKiros/Machine_Learning_Project.git
cd Machine_Learning_Project

3.	Set up a Virtual Environment: For Windows:
bash
CopyEdit
python -m venv venv
.\venv\Scripts\activate
5.	Install Dependencies:
bash
CopyEdit
pip install -r requirements.txt

7.	Start the FastAPI Backend:
bash
CopyEdit
uvicorn app.main:app --reload
The backend will be available at http://127.0.0.1:8000.

8.	Start the Streamlit Frontend:
bash
CopyEdit
streamlit run app/frontend.py
The frontend will be available at http://localhost:8501.
Usage

Interacting with the System
1.	Provide User Inputs: In the Streamlit app, users can input preferences such as destination type, budget, travel season, etc.
2.	Prediction: The backend processes the inputs and uses the trained model to recommend the best travel destination.
3.	Recommendations: The system returns the destination suggestion along with additional travel insights (transportation, accommodation, etc.).
4.	Adjustable Preferences: Users can modify their preferences to see how changes affect the recommendations.
FastAPI Endpoint
The backend exposes a POST endpoint at /predict/ for travel destination predictions.
Example Input Format:

json
CopyEdit
{
  "destination_type": "Beach",
  "trip_purpose": "Adventure",
  "budget_range": "Budget-Friendly",
  "travel_season": "Summer",
  "travel_duration": "Weekend",
  "accommodation_preference": "Hotel",
  "interest": "Adventure",
  "food_interests": "Vegetarian",
  "travel_type_group": "Solo"
}

Streamlit Frontend
The Streamlit app provides a simple form where users can input their preferences and view the recommended destination. Upon submission, it sends the data to the FastAPI backend for prediction.

Model Training and Evaluation
The model is trained using historical data on travel destinations and user preferences. Key steps:
1.	Data Preprocessing: Handling missing values, encoding categorical features, and scaling numerical values.
2.	Model Training: Using a RandomForestClassifier for classification tasks.
3.	Model Evaluation: Evaluating model performance using metrics like accuracy, precision, recall, and F1-score.
Backend API (FastAPI)
The FastAPI backend serves the trained model and handles incoming requests for predictions.
Example:
python
CopyEdit
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load the trained model
model = joblib.load("travel_recommendation_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

@app.post("/predict/")
async def predict_travel_destination(input_data: TravelInput):
    encoded_data = {...}  # Process and encode input data
    prediction = model.predict(pd.DataFrame([encoded_data]))
    predicted_destination = label_encoders["Country_Info"].inverse_transform(prediction)[0]
    return {"predicted_destination": predicted_destination}
    
Testing the Model
After deployment, the API and UI should be tested to ensure that:
•	The API returns correct predictions for user inputs.
•	The Streamlit UI displays the correct recommendations.
•	The model performs well on the test set.

Deployment
The system can be deployed using cloud platforms:
•	Backend (FastAPI): Deployed on Render. Access the API at https://machine-learning-project-11-muoj.onrender.com/docs
•	Frontend (Streamlit): Deployed on Streamlit Cloud. Access the app at https://mhiretkiros-machine-learning-project-app-v7ztrl.streamlit.app/

Conclusion
This project demonstrates the power of AI in the travel industry by providing personalized recommendations to users based on various factors. The integration of FastAPI, Streamlit, and machine learning allows for a scalable and interactive system. Future improvements can include adding more features, integrating real-time data, or enhancing the UI for an even better user experience.

Links:
•	GitHub Repository: https://github.com/MhiretKiros/Machine_Learning_Project
•	Streamlit App: https://mhiretkiros-machine-learning-project-app-v7ztrl.streamlit.app/
•	FastAPI Documentation: https://machine-learning-project-11-muoj.onrender.com/docs

Contact
For any questions or suggestions, feel free to reach out:
•	Email: kirosmhret97@gmail.com
•	LinkedIn:  https://www.linkedin.com/in/mhret-kiros-8aa2ba332/

