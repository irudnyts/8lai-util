def count_number_of_rows(data):
  return len(data.index)

def slice_head(data):
  return data.head()

def describe_numeric_variable(data, variable):
  return data.loc[:, variable].describe().loc[['min', 'mean', 'max']]

def describe_categorical_variable(data, variable):
  return data.loc[:, variable].value_counts()
