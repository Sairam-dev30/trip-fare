# TripFare: NYC Taxi Fare Prediction

A Machine Learning project that predicts taxi fares in New York City based on trip details, and deploys it as a Streamlit web app.

---

## 🚀 Project Structure

TripFare_Project/
├── taxi_fare.csv # Raw dataset (NYC taxi trip records)
├── tripfare_project.py # Data processing, model training, evaluation, and best-model pickle
├── best_model.pkl # Trained regression model (generated after running tripfare_project.py)
├── app.py # Streamlit app for real-time fare prediction
├── README.md # Project documentation (this file)
└── requirements.txt # Python dependencies


---

## 🛠️ Prerequisites

- Python 3.8+  
- pip  

---

## 📦 Installation

1. **Clone the repository** (or unzip the project folder):
   ```bash
   git clone <your-repo-url>
   cd TripFare_Project

2 .Install dependencies:
pip install -r requirements.txt

 Training the Model
Ensure taxi_fare.csv is in the project root.

Run the training and evaluation script:

python tripfare_project.py


3.This script will:

Load and preprocess the data

Perform feature engineering (distance, time features, outlier handling, encoding)

Select top features via Random Forest importance

Train 5 regression models (Linear, Ridge, Lasso, Random Forest, Gradient Boosting)

Evaluate and print performance metrics (R², RMSE, MAE)

Save the best performing model to best_model.pkl

📊 Exploring the Data
The tripfare_project.py script also generates EDA visualizations:

Correlation heatmap

Feature-importance bar chart

Distribution plots for distance and fare-per-km

(You can enable/disable plotting in the script as needed.)

Running the Streamlit App
Make sure the trained model file best_model.pkl is present.

Launch the app:
streamlit run app.py


The app will open in your browser. Enter:

Pickup & dropoff coordinates

Passenger count

Pickup date & time (EDT)

Click Predict Fare to see the estimated taxi fare and a map of the trip.

