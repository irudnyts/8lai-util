from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import statistics
import math

def count_number_of_rows(data):
  return len(data.index)

def slice_head(data):
  return data.head()

def describe_numeric_variable(data, variable):
  return data.loc[:, variable].describe().loc[['min', 'mean', 'max']]

def describe_categorical_variable(data, variable):
  return data.loc[:, variable].value_counts()

def fit_simple_linear_regression(data, y, x):
  x = data[x].to_numpy().reshape(-1, 1)
  y = data[y].to_numpy()
  model = LinearRegression()
  model.fit(x, y)
  return model

def fit_linear_regression(data, y, x):
  x = data[x].to_numpy().reshape(-1, len(x))
  y = data[y].to_numpy()
  model = LinearRegression()
  model.fit(x, y)
  return model

def get_coefficients(model):
    coefs = model.coef_.tolist()
    coefs.insert(0, model.intercept_)
    return coefs

def predict(model, x):
  if isinstance(x, list):
    return model.predict(np.array([x]))[0]
  else:
    return model.predict(np.array([[x]]))[0]

def get_rmse(model, data, y, x):
  x_length = len(x) if isinstance(x, list) else 1
  y_pred = model.predict(data[x].to_numpy().reshape(-1, x_length))
  y_true = data[y]
  return math.sqrt(statistics.mean((y_pred - y_true) ** 2))
