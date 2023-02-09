from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import statistics
import math
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

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

def draw_row(x, i):
  size = int(math.sqrt(x.shape[1]))
  image = tf.transpose(tf.reshape(tf.convert_to_tensor(x.iloc[i, :]), [size, size]))
  fig = plt.figure
  plt.imshow(image, cmap='gray')
  plt.show()

def fit_dense_neural_network(x, y, layers = [100]):
  if len(layers) < 1:
    sys.exit("Specify at least one hidden layer")
  n_features = int(x.shape[1])
  n_categories = y.value.unique().shape[0]
  y = tf.convert_to_tensor(y)
  x = tf.convert_to_tensor(x)
  x = tf.divide(x, 255)
  model = keras.models.Sequential()
  model.add(keras.layers.Dense(layers[0], activation="relu", input_shape = [n_features]))
  if len(layers) > 1:
    layers.pop(0)
    for neurons in layers:
      model.add(keras.layers.Dense(neurons, activation="relu"))
  model.add(keras.layers.Dense(n_categories, activation = "softmax"))
  model.compile(
      loss = keras.losses.sparse_categorical_crossentropy,
      metrics = keras.metrics.sparse_categorical_accuracy,
      optimizer = keras.optimizers.RMSprop()
  )
  model.fit(x, y, epochs=10, batch_size = 32)
  return model

def predict_class(model, x, i):
  tensor = tf.reshape(tf.convert_to_tensor(x.iloc[i, :]), [1, int(x.shape[1])])
  return np.argmax(model.predict(tensor, verbose = 0))
