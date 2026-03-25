import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # Linear curve
from sklearn.preprocessing import PolynomialFeatures # Quadratic curve
from sklearn.metrics import mean_squared_error, r2_score

# 1. Data load
df = pd.read_csv("StudentPerformanceFactors.csv").dropna()

# 2. Features and Target
features = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Tutoring_Sessions']
X = df[features]
y = df['Exam_Score']

# 3. Data Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. --- EFFICIENCY UPGRADE: MODEL TRAINING---
# Model : Degree 4
poly_4 = PolynomialFeatures(degree=4)
X_train_p4 = poly_4.fit_transform(X_train)
X_test_p4 = poly_4.transform(X_test)

model_4 = LinearRegression().fit(X_train_p4, y_train)
p4_pred = model_4.predict(X_test_p4)

