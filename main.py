import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # Linear curve
from sklearn.preprocessing import PolynomialFeatures # Quadratic curve
from sklearn.metrics import mean_squared_error, r2_score

# 1. Data load
df = pd.read_csv("StudentPerformanceFactors.csv").dropna()
