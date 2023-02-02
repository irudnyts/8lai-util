from sklearn.linear_model import LinearRegression

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

def get_coefficients(model):
  return [model.intercept_, model.coef_[0]]

def predict(model, x):
  return model.predict(np.array([[x]]))
