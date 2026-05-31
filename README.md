# 📊 Data Science Internship: Machine Learning 

Welcome to my repository for **Week 4** of the Data Science Internship program with AnalystLab Africa! This week, the project transitioned from exploratory statistical boundaries into predictive **Supervised Machine Learning**, building and optimizing both regression and classification pipelines.

This repository contains the end-to-end machine learning workflows for two distinct datasets: **The Housing Prices Dataset** and **The Titanic Passenger Manifest**.

---

## 🛠️ Technologies & Tools Used
* **Language:** Python 3.10
* **Core ML Library:** `scikit-learn`
* **Data Engineering Libraries:** `pandas`, `numpy`, `seaborn`, `matplotlib`
* **Environment:** Jupyter Notebook

---

## 🏠 Project 1: Housing Prices Dataset (Regression Task)

### 📋 Objective
Predict continuous house prices by training a Linear Regression model using an 80/20 train/test split.

### 🔍 Iterative Model Optimization & Results
* **The Baseline Model:** Initialized using only the primary key features identified in Week 3 (`area`, `bathrooms`, `prefarea`). 
  * *Baseline Performance:* $\text{RMSE} = \mathbf{1,644,312.74}$
* **The Optimized Model:** Expanded the feature matrix to include crucial structural and convenience context (`bedrooms`, `stories`, `mainroad`, `airconditioning`, `parking`). Categorical columns were processed via one-hot encoding, and a logarithmic transformation was applied to the target variable to neutralize the impact of extreme luxury outliers.
  * *Optimized Performance:* $\text{RMSE} = \mathbf{1,389,197.54}$
  * *Variance Explanation ($R^2$ Score):* $\mathbf{0.6182}$

### 💡 Key Insight
Optimizing the feature set dropped the model's prediction error by **255,115.20 units (a 15.5% improvement)**. The $R^2$ score confirms that **61.8%** of the variation in housing market prices is successfully explained by the engineered features. This proves that while property footprint and location dictate baseline value, structural infrastructure heavily controls final valuation.

---

## 🚢 Project 2: Titanic Dataset 

### 📋 Objective
Predict binary passenger survival outcomes (Survived vs. Perished) by training a Logistic Regression model using an 80/20 train/test split.

### 🔍 Model Performance & Results
* **Feature Selection:** Built upon statistical dependencies verified in Week 3, utilizing Passenger Class (`Pclass`), Gender (`Sex`), Age (`Age`), and Ticket Fare (`Fare`).
* **Categorical Handling:** Cleanly mapped string values into binary numeric indicators using one-hot encoding to prevent data format constraints.
* **Model Evaluation Metric:** Accuracy Score
* **Final Performance:** $\text{Accuracy} = \mathbf{80.45\%}$

### 💡 Key Insight
The Logistic Regression model correctly classified passenger survival outcomes in approximately **80 out of every 100 unseen cases**. This exceptional baseline accuracy mathematically validates that demographic indicators hold massive predictive power, successfully picking up on historical emergency evacuation protocols ("women and children first").

---

## 🚀 Challenges Overcome
1. **Handling Categorical Constraints:** Resolved string text limitations by mapping features into machine-readable format using `pd.get_dummies(drop_first=True)`.
2. **Target Skewness:** Corrected the heavy right-hand luxury property price skew using a logarithmic target transformation, stabilizing model weights and drastically cutting down evaluation penalties (RMSE).
