# Import the pandas library to read CSV files
from pandas import read_csv

# Import the numpy library for numerical operations
import numpy as np

# Import the Model class from the keras library
from keras import Model

# Import the Layer class from the keras library
from keras.layers import Layer

# Import the keras backend
import keras.backend as K

# Import the Input, Dense, and SimpleRNN classes from the keras library
from keras.layers import Input, Dense, SimpleRNN

# Import the MinMaxScaler class from the sklearn library for data scaling
from sklearn.preprocessing import MinMaxScaler

# Import the Sequential class from the keras library to create a model
from keras.models import Sequential

# Import the mean_squared_error function from the keras library
from keras.metrics import mean_squared_error

# Import the os library for operating system related operations
import os

# Import the time library for operating system related operations
import time

import warnings
file_name="inputData.txt"
scale_data=True
f = open(file_name, 'r')
input_sequence=[i for i in f.read().split('\n') if not len(i)==0]
input_sequence=[int(i) for i in input_sequence]
f.close()

# Initialize an empty list for data scaler
data_scaler = []
# write elements of list

# If scale_data is True, scale the data between 0 and 1
if scale_data:
    max_val = max(input_sequence)
    input_sequence.append(1.5*max_val)
    # Initialize a MinMaxScaler
    data_scaler = MinMaxScaler(feature_range=(0, 1))
    # Reshape the sequence to fit the scaler
    input_sequence = np.reshape(input_sequence, (len(input_sequence), 1))
    # Fit and transform the sequence with the scaler
    input_sequence = data_scaler.fit_transform(input_sequence).flatten()        
    
# Return the sequence and the scaler
