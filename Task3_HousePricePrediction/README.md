🏡 Task 6: House Price Prediction

AI/ML Internship Project
Predict house prices using machine learning based on property features such as size, bedrooms, and location.

🔍 Project Overview

Goal: Build a regression model to predict house prices based on features like square footage, number of bedrooms, bathrooms, and city.

Key Steps Implemented:

Load dataset from data/house_prices.csv.

Inspect dataset and handle missing values.

Encode categorical features (e.g., city).

Split dataset into training and testing sets.

Train regression models:

Linear Regression

Gradient Boosting Regressor

Evaluate models using:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

Visualize results:

Actual vs Predicted Prices

Feature Importance

Save trained models and plots for later use.

🛠️ Step 1: Install Dependencies

Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate       # Linux/macOS
# .\venv\Scripts\activate     # Windows


Install required Python packages:

pip install -r requirements.txt


Example requirements.txt:

pandas>=2.0
numpy>=1.24
scikit-learn>=1.2
matplotlib>=3.5
seaborn>=0.12
joblib>=1.2

📥 Step 2: Load and Inspect Dataset

The dataset should be located in data/house_prices.csv.
Typical columns:

Column	Description
square_feet	House size in square feet
bedrooms	Number of bedrooms
bathrooms	Number of bathrooms
city	House location
price	Target variable (house price)

Run the script to check dataset info:

python3 predict_house_price.py


Output includes:

Dataset shape

Column types

Missing values per column

🔧 Step 3: Data Preprocessing

The script handles:

Missing values (if any)

Categorical encoding for city

Feature scaling for numerical data

Separating features (X) and target (y)

🧠 Step 4: Train Regression Models

Models trained:

Linear Regression

Gradient Boosting Regressor

📊 Step 5: Evaluate Models

Metrics calculated:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

Plots generated:

images/actual_vs_predicted.png — Scatter plot of predicted vs actual prices

images/feature_importances.png — Feature importance for Gradient Boosting model

▶️ Step 6: Run the Project
source venv/bin/activate
python3 predict_house_price.py


Script outputs:

Model evaluation metrics

Saved plots in images/

Saved trained model in models/house_price_model.pkl
