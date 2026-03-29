# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset, remove unnecessary columns, convert categorical data to dummy variables, and separate features (X) and target (y).

2. Standardize both features and target values, then split the data into training and testing sets.

3. Train an SGD Regressor model on the training data and generate predictions on the test data.

4. Evaluate using MSE, MAE, R², display coefficients, and plot actual vs predicted values.

## Program:
```
/*
Program to implement SGD Regressor for linear regression.
Developed by: Lohith V
RegisterNumber:  25013313
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("CarPrice_Assignment.csv")

print(data.head())
print(data.info())
data = data.drop(['CarName', 'car_ID'], axis=1)
data = pd.get_dummies(data, drop_first=True)
X = data.drop('price', axis=1)
y = data['price']
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(np.array(y).reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3)
sgd_model.fit(X_train, y_train)
y_pred = sgd_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 =r2_score(y_test,y_pred)
print('Name:Lohith v')
print('Reg. No:25013313')
print(f"Mean Square Error:",mse)
print(f"Mean Absolute Error:",mae)
print(f"R-Squared Score:", r2)
print("\nModel Coefficients:")
print("Coefficients:", sgd_model.coef_)
print("Intercept:", sgd_model.intercept_)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red') # Perfect prediction line
plt.show()

*/
```

## Output:
![alt text](<Screenshot 2026-03-09 110835.png>)
![alt text](<Screenshot 2026-03-09 110852.png>)
![alt text](<Screenshot 2026-03-09 110903.png>)
![alt text](<Screenshot 2026-03-09 110915.png>)
![alt text](<Screenshot 2026-03-09 110925.png>)

## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
