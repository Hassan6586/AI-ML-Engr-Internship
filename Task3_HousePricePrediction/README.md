# Task 6 — House Price Prediction 🏡

> **AI/ML Internship – Task 6**  
> Predict house prices using property features like size, number of bedrooms, and location.  
> This repository contains a complete pipeline: data loading, preprocessing, feature engineering, model training, evaluation (MAE, RMSE), feature importance, and prediction visualization.

---

## 🔎 Project Overview

**Goal:** Build a regression model that predicts house prices based on property features.

**Main steps implemented:**
1. Load dataset (`data/house_prices.csv`)
2. Data preprocessing:
   - Handle missing values
   - Encode categorical features (city/location)
   - Scale numerical features
3. Train regression models:
   - Linear Regression
   - Gradient Boosting Regressor
4. Evaluate model performance using:
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
5. Generate visualizations:
   - Actual vs Predicted Prices (`images/actual_vs_predicted.png`)
   - Feature Importance (`images/feature_importances.png`)
6. Save trained models in `models/`

---

## 📁 Repository Structure

HousePricePrediction/
├── data/
│ └── house_prices.csv
├── images/
│ ├── actual_vs_predicted.png
│ └── feature_importances.png
├── models/
├── predict_house_price.py
├── requirements.txt
└── README.md ← (this file)

yaml
Copy code

---

## 🛠️ Requirements

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# .\venv\Scripts\activate      # Windows PowerShell
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Example requirements.txt:


shell
Copy code
pandas>=2.0,<3
numpy>=1.24,<2
scikit-learn>=1.2,<2
matplotlib>=3.5,<4
seaborn>=0.12,<1
joblib>=1.2
📥 Dataset
House Price Dataset (data/house_prices.csv)
Columns include:


Copy code
square_feet, bedrooms, bathrooms, city, price
Make sure the target column is named price.



▶️ Run the pipeline
bash
Copy code
source venv/bin/activate
python3 predict_house_price.py
This will:

Print dataset shape, column types, and missing value summary

Preprocess features

Train regression models

Print MAE and RMSE

Save plots in images/

Save trained model in models/



📷 Visualizations
images/actual_vs_predicted.png — scatter plot comparing predicted vs actual prices

images/feature_importances.png — bar plot showing most important features for price prediction

(Replace placeholders with images generated after running the script.)

🧠 Model Training Steps
Load CSV into pandas DataFrame.

Inspect data: df.head(), df.info(), df.describe().

Handle missing values.

Encode categorical variables (e.g., city → one-hot encoding).

Split dataset into X (features) and y (target price).

Train/test split (e.g., 80/20).

Train models:

Linear Regression

Gradient Boosting Regressor

Evaluate performance using:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

Visualize results:

Actual vs Predicted prices scatter plot

Feature importance bar chart

Save trained models with joblib.

🧰 Tips / Troubleshooting
ModuleNotFoundError → Make sure venv is activated and dependencies installed.

Missing price column → Rename your dataset column to price.

Incorrect visualization → Ensure images/ folder exists and script has write permissions.

