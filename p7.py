import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = fetch_california_housing(as_frame=True)
x, y = data.data[['AveRooms']], data.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)

sorted_idx = x_test['AveRooms'].argsort()
x_sorted = x_test.iloc[sorted_idx]
y_pred_lr_sorted = y_pred_lr[sorted_idx]

plt.scatter(x_test, y_test, label='Actaul')
plt.plot(x_sorted, y_pred_lr_sorted, color='red' , label='Linear fit')
plt.title('Linear Regression - California Housing')
plt.xlabel('Average Rooms')
plt.ylabel('Housing prices')
plt.legend()
plt.show()

print("Linear Regression: ")
print("MSE: ", mean_squared_error(y_test, y_pred_lr))
print("R2: ", r2_score(y_test, y_pred_lr))

# Polynomial Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv('auto-mpg.data', sep=r'\s+', header=None, na_values='?')
x = data.iloc[:,2].values.reshape(-1,1)
y = data.iloc[:,0].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
poly_model = make_pipeline(
  PolynomialFeatures(degree=2),
  StandardScaler(),
  LinearRegression()
)
poly_model.fit(x_train, y_train)
y_pred = poly_model.predict(x_test)
plt.scatter(x_test, y_test, color='red', label='Actual')
plt.scatter(x_test, y_pred, color='green', label='Predicted')
plt.legend()
plt.xlabel("Displacement")
plt.ylabel("MPG-miles per gallon")
plt.title("Polynomial Regression - Auto MPG Dataset")
plt.show()

print("MSE: ", mean_squared_error(y_test, y_pred))
print("R2: ", r2_score(y_test, y_pred))




















