Personalized Travel Destination Recommender
Overview
The Personalized Travel Destination Recommender is an advanced machine learning-based system that recommends vacation destinations tailored to individual preferences. By analyzing inputs such as budget, travel season, personal interests, and accommodation preferences, the system provides personalized recommendations to enhance users' travel planning experience. Whether users are seeking an adventurous vacation or a relaxing getaway, this system ensures they receive destination suggestions that align with their desires.
Key Features
•	Personalized Recommendations: Offers tailored vacation destination recommendations based on user-specific inputs like budget, travel season, and interests.
•	Real-Time Predictions: The backend is powered by FastAPI, ensuring fast and efficient real-time predictions.
•	Interactive Web App: Built with Streamlit, the web interface is user-friendly and intuitive, allowing users to easily input their preferences and get travel suggestions.
•	Comprehensive Insights: Includes transportation and accommodation suggestions to complement the travel destination, providing users with a holistic travel plan.
•	Scalability: Designed for easy expansion, allowing for the integration of more features, additional models, or API endpoints.
Technology Stack
•	Machine Learning: The core recommendation logic is powered by the RandomForestClassifier (for classification tasks) and RandomForestRegressor (for regression tasks) from scikit-learn.
•	Backend: FastAPI is used to create a robust and fast API for handling real-time requests.
•	Frontend: Streamlit is used to build an intuitive web interface for easy interaction with the recommendation system.
•	Serialization: The trained machine learning models and encoders are saved using Joblib to persist and load them when required.
•	Data Processing: The project relies heavily on Pandas for data manipulation and scikit-learn for preprocessing tasks like encoding categorical features and scaling numerical values.
Installation Guide
Prerequisites
Before you begin, ensure you have the following installed:
•	Python 3.7 or higher
•	pip (Python's package installer)
•	A code editor (e.g., VS Code, PyCharm)
Steps to Install and Run Locally
1.	Clone the repository:
bash
CopyEdit
git clone https://github.com/yourusername/travel-destination-recommender.git
cd travel-destination-recommender
2.	Set up a virtual environment:
For Windows:
bash
CopyEdit
python -m venv venv
.\venv\Scripts\activate
3.	Install the required dependencies:
bash
CopyEdit
pip install -r requirements.txt
4.	Start the FastAPI backend:
bash
CopyEdit
uvicorn app.main:app –reload
(app.main(tg.py)
The backend will be available at http://127.0.0.1:8000.
5.	Start the Streamlit frontend:
bash
CopyEdit
streamlit run app/frontend.py
The frontend will be available at http://localhost:8501.
Usage
Interacting with the System
1.	Provide User Inputs: The Streamlit app provides a form where users can input their travel preferences. These inputs include details like destination type, budget, trip purpose, and more.
2.	Prediction: After the user submits the form, the backend (FastAPI) processes the inputs and makes a prediction based on the trained machine learning model.
3.	Recommendations: The system returns the predicted travel destination along with relevant suggestions for transportation, accommodation, and other travel details.
4.	Adjustable Preferences: Users can adjust their preferences to see how different inputs affect the recommended destinations.
FastAPI Endpoint
The FastAPI backend exposes a POST endpoint at /predict/ which takes user inputs and returns a predicted destination. Here's an example of the input format:
json
CopyEdit
{
  "destination_type": "Beach",
  "trip_purpose": "Family Vacation",
  "budget_range": "Mid-Range",
  "travel_season": "Summer",
  "travel_duration": "A Month",
  "accommodation_preference": "Hotel",
  "interest": "Relaxation",
  "food_interests": "Vegetarian",
  "travel_type_group": "Family with Kids"
}
Streamlit Frontend
The Streamlit app allows users to interact with the system in a simple and engaging way. The app includes:
•	A form for submitting travel preferences.
•	A display section for the recommended destination.
•	Error handling for invalid inputs (e.g., unseen categories).
Model Training and Evaluation
The machine learning model is trained using historical data about various travel destinations and user preferences. The dataset is preprocessed to handle missing values, encode categorical features, and scale numerical values. After training, the model is evaluated using performance metrics such as accuracy (for classification tasks) or mean squared error (for regression tasks).
Data Preprocessing
•	Missing Value Handling: Missing data is either imputed or removed from the dataset.
•	Categorical Encoding: Categorical columns are encoded using label encoding to convert them into numerical representations.
•	Feature Scaling: Numerical features are scaled using standardization techniques to ensure that all features are on a similar scale.
Model Evaluation
The model is evaluated based on various metrics:
•	Classification: Accuracy score, precision, recall, and F1-score.
•	Regression: Mean Absolute Error (MAE), Mean Squared Error (MSE).
Saving and Loading the Model
Once the model is trained, it is serialized and saved using Joblib. This allows the model to be loaded and reused in the backend for prediction without the need to retrain it every time the system is started.
python
CopyEdit
import joblib

# Save the model
joblib.dump(model, "travel_recommendation_model.pkl")

# Load the model
model = joblib.load("travel_recommendation_model.pkl")
Deployment
The backend API and the frontend are designed to be deployed in a production environment. Currently, the system is deployed using Render for hosting the FastAPI backend and Streamlit app, ensuring it is accessible to users online.
Deploying the Backend
To deploy the FastAPI backend, you can use cloud platforms like Heroku, Render, or AWS. After deployment, you can access the API using the hosted URL.
https://machine-learning-project-11-muoj.onrender.com/docs (deployed by render) 
Deploying the Frontend
The Streamlit app can be deployed using Streamlit Sharing, Heroku, or any other platform that supports Python applications.
https://mhiretkiros-machine-learning-project-app-v7ztrl.streamlit.app/ ( deployd by streamit)
Contact
For any questions or feedback, feel free to reach out:
•	GitHub: https://github.com/mihretkiros
•	Email: kirosmhret97@gmail.com

