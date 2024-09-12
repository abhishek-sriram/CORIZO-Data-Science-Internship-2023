# Corizo Internship 

## Internship Overview

During my internship at **Corizo**, I worked on two data-driven projects: **Stock Price Prediction** and **Wine Quality Analysis**. The internship provided hands-on experience in data analysis, feature engineering, model building, and prediction using various machine learning algorithms. This README outlines the details of the projects and the methodologies used.

---

## 1. Stock Price Prediction

### Project Description
The Stock Price Prediction project aimed to predict the future prices of stocks using historical data. The goal was to build a model that could forecast the next day's closing price based on historical trends and market behavior.

### Steps Involved
1. **Data Collection**:
   - Gathered historical stock price data, including features like opening price, closing price, high, low, and volume.

2. **Data Preprocessing**:
   - Cleaned and handled missing values in the dataset.
   - Scaled features for better model performance.

3. **Feature Engineering**:
   - Created new features like moving averages (SMA, EMA) and technical indicators to enhance prediction accuracy.

4. **Modeling**:
   - Implemented various regression models such as:
     - Linear Regression
     - Random Forest Regressor
     - Long Short-Term Memory (LSTM) model for time-series prediction.

5. **Evaluation**:
   - Used metrics like Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) to evaluate model performance.

6. **Result**:
   - Achieved satisfactory accuracy with the LSTM model for predicting the next day's closing price, demonstrating strong predictive power.

---

## 2. Wine Quality Analysis

### Project Description
The Wine Quality Analysis project focused on building a classification model to predict the quality of wines based on physicochemical properties. The dataset included features such as acidity, alcohol content, sugar levels, and pH value.

### Steps Involved
1. **Data Collection**:
   - Used the publicly available **Wine Quality dataset** from UCI Machine Learning Repository.

2. **Data Preprocessing**:
   - Performed normalization of data to ensure all features are on the same scale.
   - Balanced the dataset to address class imbalance.

3. **Exploratory Data Analysis (EDA)**:
   - Visualized the relationships between various features and wine quality.
   - Analyzed the correlation between variables using heatmaps and pair plots.

4. **Modeling**:
   - Implemented multiple classification models such as:
     - Logistic Regression
     - Decision Trees
     - Random Forest Classifier
     - Gradient Boosting
   - Used cross-validation to tune model parameters and improve performance.

5. **Evaluation**:
   - Metrics used: Accuracy, Precision, Recall, F1-Score.
   - Random Forest and Gradient Boosting showed the best performance in predicting wine quality.

6. **Result**:
   - The classification models were able to achieve an accuracy of over 85%, indicating strong model performance for wine quality prediction.

---

## Tools & Technologies Used

- **Programming Language**: Python
- **Libraries**: Pandas, Numpy, Scikit-Learn, Matplotlib, Seaborn, Keras/TensorFlow (for LSTM)
- **Version Control**: Git
- **Platforms**: Jupyter Notebook

---

## Key Learnings

- Understanding the end-to-end process of a data science project.
- Application of machine learning techniques to real-world datasets.
- Experience with regression and classification models.
- Feature engineering and the impact of data preprocessing on model performance.
- Improved skills in data visualization and exploratory data analysis (EDA).

---

This internship gave me valuable insights into the practical application of machine learning and data science concepts. The experience significantly enhanced my ability to build models for both prediction and classification tasks, which I will carry forward in my future projects.
