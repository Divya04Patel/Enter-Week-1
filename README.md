# ⚡ Electric Vehicle Range Prediction  

## 🧭 Overview  
This project aims to **predict the driving range of Electric Vehicles (EVs)** using their **technical specifications** such as battery capacity, motor power, efficiency, and vehicle weight.  
By applying **machine learning regression models**, the project estimates how far an EV can travel on a full charge — helping **manufacturers, researchers, and buyers** make informed decisions about EV performance and efficiency.  

---

## 📊 Dataset  
**Source:** [Electric Vehicle Specifications and Prices – Kaggle](https://www.kaggle.com/datasets/fatihilhan/electric-vehicle-specifications-and-prices)  

**Description:**  
The dataset contains detailed information about various EV models, including the following key attributes:  
- 🔋 **Battery Capacity (kWh)**  
- ⚙️ **Power (kW or HP)**  
- 🚘 **Vehicle Weight**  
- 🕐 **Acceleration (0–100 km/h)**  
- 💰 **Price**  
- ⚡ **Energy Consumption**  

**Target Variable:**  
- 🚗 **Electric Range (km / miles)**  

---

## 🎯 Objective  
To develop a **machine learning regression model** that can accurately predict an EV’s **driving range** based on its physical and technical specifications.  

---

## ⚙️ Workflow / Methodology  
1. **Data Preprocessing**  
   - Handle missing or inconsistent data  
   - Convert categorical data (like brand/type) into numerical form  
   - Normalize numerical features for better model performance  

2. **Exploratory Data Analysis (EDA)**  
   - Visualize relationships between features and EV range  
   - Identify correlations between power, battery, and range  

3. **Model Training**  
   - Train regression models such as:  
     - Linear Regression  
     - Random Forest Regressor  
     - XGBoost Regressor  

4. **Model Evaluation**  
   - Use performance metrics like:  
     - MAE (Mean Absolute Error)  
     - RMSE (Root Mean Squared Error)  
     - R² Score  

---

## 📈 Expected Outcome  
- ✅ Predict EV range with high accuracy  
- 📊 Visualize how features (battery, weight, etc.) impact range  
- 🔍 Provide insights to improve EV design and efficiency  

---
<img width="1919" height="810" alt="image" src="https://github.com/user-attachments/assets/f7ef0270-82b4-4a65-af04-4129768432fe" />
<img width="1919" height="830" alt="image" src="https://github.com/user-attachments/assets/0116d187-d1a3-4750-a61b-9ae707cd32c3" />
<img width="1917" height="811" alt="image" src="https://github.com/user-attachments/assets/eb4469a7-468f-4866-ac94-4b2ca7143c43" />




## 💻 Tech Stack  
| Category | Tools / Libraries |
|-----------|-------------------|
| Programming Language | Python 🐍 |
| Libraries | pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost |
| IDE / Notebook | VS Code |
| Platform | Kaggle |






