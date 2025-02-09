# 🌍 Personalized Travel Destination Recommender

## ✈️ Overview
The **Personalized Travel Destination Recommender** is an AI-powered system designed to help users find vacation destinations tailored to their unique preferences. By analyzing inputs such as **budget, travel season, personal interests, and accommodation preferences**, the system provides personalized travel recommendations. The backend uses **FastAPI** for real-time predictions, while the user-friendly interface is built with **Streamlit**.

## 🚀 Key Features
✅ **Personalized Recommendations** – Get travel suggestions based on budget, trip purpose, accommodation type, and interests.  
✅ **Real-Time Predictions** – **FastAPI** backend ensures quick responses.  
✅ **Interactive Web Interface** – Built with **Streamlit** for an intuitive user experience.  
✅ **Comprehensive Insights** – Recommends **transportation and accommodation** options in addition to destinations.  
✅ **Scalable & Extendable** – Can be expanded with more features and models.  

## 🛠️ Technology Stack
- **🤖 Machine Learning** – `RandomForestClassifier` & `RandomForestRegressor` from `scikit-learn`.
- **🔧 Backend** – `FastAPI` for serving predictions.
- **🎨 Frontend** – `Streamlit` for an interactive UI.
- **💾 Serialization** – `Joblib` for saving and loading models.
- **📊 Data Processing** – `Pandas` and `scikit-learn` for data manipulation and preprocessing.

## 🏗️ Installation Guide

### 📌 Prerequisites
Ensure you have the following installed:
- 🐍 Python 3.7 or higher
- 📦 pip (Python's package manager)
- 💻 Code editor (VS Code, PyCharm, etc.)

### 📥 Steps to Install and Run Locally
1️⃣ **Clone the Repository:**
```bash
 git clone https://github.com/MhiretKiros/Machine_Learning_Project.git
 cd Machine_Learning_Project
```
2️⃣ **Set up a Virtual Environment (Windows):**
```bash
 python -m venv venv
 .\venv\Scripts\activate
```
3️⃣ **Install Dependencies:**
```bash
 pip install -r requirements.txt
```
4️⃣ **Start the FastAPI Backend:**
```bash
 uvicorn app.main:app --reload
```
🖥️ The backend will be available at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

5️⃣ **Start the Streamlit Frontend:**
```bash
 streamlit run app/frontend.py
```
🌐 The frontend will be available at: [http://localhost:8501](http://localhost:8501)

## 📌 Usage Guide

### 🏝️ Interacting with the System
1. **Enter User Preferences** – Choose your destination type, budget, travel season, accommodation, etc.
2. **Generate Recommendations** – The backend processes inputs and suggests the best travel destination.
3. **Explore Insights** – Get additional details like **transportation and accommodation options**.
4. **Modify Preferences** – Adjust inputs to explore different recommendations.

### 📡 FastAPI Endpoint
The backend exposes a **POST** endpoint at `/predict/` for travel destination predictions.

#### 📜 Example JSON Input:
```json
{
  "destination_type": "Beach",
  "trip_purpose": "Adventure",
  "budget_range": "Budget-Friendly",
  "travel_season": "Summer",
  "travel_duration": "Weekend",
  "accommodation_preference": "Hotel",
  "interest": "Adventure",
  "food_interests": "Vegetarian",
  "travel_type_group": "Solo Traveler"
}
```

## 🎨 Streamlit Frontend
The **Streamlit** app offers an easy-to-use interface where users enter their preferences and receive recommendations in real-time!

## 📊 Model Training & Evaluation
🛠️ **Steps in Model Development:**
1. **Data Preprocessing** – Handling missing values, encoding categorical features, and scaling numerical values.
2. **Model Training** – `RandomForestClassifier` used for classification tasks.
3. **Model Evaluation** – Accuracy, precision, recall, and F1-score metrics assessed.

## ⚡ Backend API (FastAPI) – Sample Code
```python
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load the trained model
model = joblib.load("travel_recommendation_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

@app.post("/predict/")
async def predict_travel_destination(input_data: dict):
    encoded_data = {...}  # Process and encode input data
    prediction = model.predict(pd.DataFrame([encoded_data]))
    predicted_destination = label_encoders["Country_Info"].inverse_transform(prediction)[0]
    return {"predicted_destination": predicted_destination}
```

## 🧪 Testing
✔️ Ensure the **API returns accurate predictions**.  
✔️ Verify the **Streamlit UI correctly displays recommendations**.  
✔️ Assess **model performance using a test set**.  

## 🚀 Deployment
🔹 **Backend (FastAPI)** – Hosted on **Render**: (https://machine-learning-project-11-muoj.onrender.com/docs)  
🔹 **Frontend (Streamlit)** – Deployed on **Streamlit Cloud**: (https://mhiretkiros-machine-learning-project-app-v7ztrl.streamlit.app/)  

## 🏁 Conclusion
🎯 The **Personalized Travel Destination Recommender** leverages **AI & machine learning** to enhance travel planning. The integration of **FastAPI, Streamlit, and ML models** ensures a seamless and interactive experience. Future improvements may include **real-time data integration** and **an improved UI design**.

## 🔗 Useful Links
- **📂 GitHub Repository**: [Machine Learning Project](https://github.com/MhiretKiros/Machine_Learning_Project)
- **🌍 Streamlit App**: [Personalized Travel Recommender](https://mhiretkiros-machine-learning-project-app-v7ztrl.streamlit.app/)
- **📜 FastAPI Docs**: [API Documentation](https://machine-learning-project-11-muoj.onrender.com/docs)

## 📩 Contact
📧 **Email**: [kirosmhret97@gmail.com](mailto:kirosmhret97@gmail.com)  
🔗 **LinkedIn**: [Mhiret Kiros](https://www.linkedin.com/in/mhret-kiros-8aa2ba332/)  

---
💡 _"Travel smarter with recommendations!"_ ✨

