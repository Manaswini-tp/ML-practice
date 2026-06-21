import pandas as pd

def find_S_algorithm(file_path):
  data = pd.read_csv(file_path)
  print('Training Data: ')
  print(data)
  
  attributes = data.columns[:-1]
  class_label = data.columns[-1]
  hypothesis = None
  
  for index, row in data.iterrows():
    if row[class_label]=='Yes':
      if hypothesis is None:
        hypothesis = list(row[attributes])
      else:
        for i, value in enumerate(row[attributes]):
          if hypothesis[i]!=value:
            hypothesis[i]='?'
  return hypothesis

file_path = ('training_data.csv')
hypothesis = find_S_algorithm(file_path)
print('\n Final hypothesis: ', hypothesis)
