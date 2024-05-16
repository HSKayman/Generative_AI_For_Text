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

# Define a function to generate a sequence of numbers
# =============================================================================
# This function normalize_input_sequence(file_name, scale_data=True) reads a sequence of numbers from a file, converts them into integers, and optionally scales them between 0 and 1 using the MinMaxScaler from the sklearn.preprocessing module.
#
# It opens a file with the name file_name in read mode.
# It reads the file line by line, splits each line into separate strings using the newline character as the delimiter, and excludes any empty strings.
# It converts each string in the list to an integer.
# If scale_data is True, it scales the data between 0 and 1 using the MinMaxScaler. This is done by:
# Reshaping the sequence to fit the scaler.
# Fitting and transforming the sequence with the scaler.
# It returns the sequence and the scaler.
# Example 1:
# Assume 'input1.txt' contains:
# 1
# 2
# 3
# 4
# 5
# Output: [0.   0.25 0.5  0.75 1.  ]
# =============================================================================
def normalize_input_sequence(file_name, scale_data=True):
    # open file in read mode
    f = open(file_name, 'r')
    input_sequence=[i for i in f.read().split('\n') if not len(i)==0]
    input_sequence=[int(i) for i in input_sequence]
    f.close()
    
    # Initialize an empty list for data scaler
    data_scaler = []
    # write elements of list
    
    # If scale_data is True, scale the data between 0 and 1
    if scale_data:
        # Initialize a MinMaxScaler
        data_scaler = MinMaxScaler(feature_range=(0, 1))
        # Reshape the sequence to fit the scaler
        input_sequence = np.reshape(input_sequence, (len(input_sequence), 1))
        # Fit and transform the sequence with the scaler
        input_sequence = data_scaler.fit_transform(input_sequence).flatten()        
        
    # Return the sequence and the scaler
    return input_sequence, data_scaler

# Define a function to generate a dataset
# =============================================================================
# This function prepare_dataset(file_name, time_steps, train_percent, scale_data=True) prepares a dataset for time series prediction. It reads a sequence of numbers from a file, converts them into integers, optionally scales them between 0 and 1, and then splits the sequence into training and testing sets.
# 
# It generates a sequence of numbers and a data scaler by calling the normalize_input_sequence function.
# It generates the indices for the target variable Y.
# It generates the target variable Y using the indices.
# It generates the input variable X by stacking shifted versions of the data.
# It generates a random permutation of the indices.
# It splits the data into training and testing sets based on the train_percent parameter.
# It reshapes the input data to fit the model.
# It returns the training and testing data, and the scaler.
# Example 1:
# Assume 'input1.txt' contains:
# 1
# 2
# 3
# 4
# 5
# trainX, trainY, testX, testY, data_scaler = prepare_dataset('input1.txt', 2, 0.8)
# Output: 
# trainX = [
#           [[0.  ]
#            [0.25]]
#
#           [[0.25]
#            [0.5 ]]
#                     ] 
#
# trainY = [0.5  0.75]

# testX =  [
#           [[0.5 ]
#            [0.75]]
#                    ]
# testY = [1.]
# =============================================================================
def prepare_dataset(total_numbers, time_steps, train_percent, scale_data=True):
    # Generate a sequence of numbers
    data, data_scaler = normalize_input_sequence(total_numbers, scale_data)    
    # Generate the indices for the target variable
    Y_indices = np.arange(time_steps, len(data), 1)
    # Generate the target variable
    Y = data[Y_indices]
    # Generate the input variable
    num_rows_x = len(Y)
    X = data[0:num_rows_x]
    for i in range(time_steps-1):
        temp = data[i+1:num_rows_x+i+1]
        X = np.column_stack((X, temp))
    # Generate a random permutation with a fixed seed   
    random_generator = np.random.RandomState(seed=13)
    indices = random_generator.permutation(num_rows_x)
    # Split the data into training and testing sets
    split_index = int(train_percent*num_rows_x)
    train_indices = indices[0:split_index]
    test_indices = indices[split_index:]
    trainX = X[train_indices]
    trainY = Y[train_indices]
    testX = X[test_indices]
    testY = Y[test_indices]
    # Reshape the input data to fit the model
    trainX = np.reshape(trainX, (len(trainX), time_steps, 1))    
    testX = np.reshape(testX, (len(testX), time_steps, 1))
    # Return the training and testing data, and the scaler
    return trainX, trainY, testX, testY, data_scaler

# Set the number of time steps
time_steps = 5
# Set the number of hidden units
hidden_units = 2
# Set the number of epochs
epochs = 5000

# Define a custom layer for attention mechanism WILL CEHCK
# =============================================================================
# The provided Python code defines a custom AttentionLayer class and a function create_RNN_with_attention to 
# create a Recurrent Neural Network (RNN) model with an attention mechanism.
# 
# The AttentionLayer class is a custom layer that computes an attention score for each input element. 
# The attention score is computed by applying a tanh activation to the dot product of the input and the attention weights, plus the attention bias. 
# The attention weights are then computed by applying a softmax activation to the alignment scores. 
# The context vector is computed by element-wise multiplying the input by the attention weights and summing over the sequence dimension.
# 
# The create_RNN_with_attention function creates an RNN model with the custom AttentionLayer. 
# The model is compiled with the Mean Squared Error (MSE) loss function and the Adam optimizer.
# =============================================================================
class AttentionLayer(Layer):
    # Initialize the layer
    def __init__(self,**kwargs):
        super(AttentionLayer,self).__init__(**kwargs)

    # Build the layer with the given input shape
    def build(self,input_shape):
        # Add a weight matrix for the attention weights
        self.attention_weights=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        # Add a bias vector for the attention weights
        self.attention_bias=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(AttentionLayer, self).build(input_shape)

    # Define the forward pass of the layer
    def call(self,x):
        # Compute the alignment scores by applying a tanh activation to the dot product of the input and the attention weights, plus the attention bias
        alignment_scores = K.tanh(K.dot(x,self.attention_weights)+self.attention_bias)
        # Remove the last dimension of the alignment scores
        alignment_scores = K.squeeze(alignment_scores, axis=-1)   
        # Compute the attention weights by applying a softmax activation to the alignment scores
        alpha = K.softmax(alignment_scores)
        # Add an extra dimension to the attention weights
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector by element-wise multiplying the input by the attention weights and summing over the sequence dimension
        context_vector = x * alpha
        context_vector = K.sum(context_vector, axis=1)
        # Return the context vector
        return context_vector

# Define a function to create a RNN model with attention
def create_RNN_with_attention(hidden_units, dense_units, input_shape, activation):
    # Define the input layer
    input_layer=Input(shape=input_shape)
    # Add a SimpleRNN layer
    rnn_layer = SimpleRNN(hidden_units, return_sequences=True, activation=activation)(input_layer)
    # Add the custom attention layer
    attention_layer = AttentionLayer()(rnn_layer)
    # Add a Dense output layer
    output_layer=Dense(dense_units, trainable=True, activation=activation)(attention_layer)
    # Create the model
    rnn_attention_model=Model(input_layer,output_layer)
    # Compile the model with mean squared error loss and Adam optimizer
    rnn_attention_model.compile(loss='mse', optimizer='adam')  
    # Return the model
    return rnn_attention_model     

# Create a RNN model with attention
rnn_attention_model = create_RNN_with_attention(hidden_units=hidden_units, dense_units=1, 
                                  input_shape=(time_steps,1), activation='tanh')
# Print the summary of the model
print(rnn_attention_model.summary())
trainX, trainY, testX, testY, data_scaler  = prepare_dataset(200, time_steps, 0.95)
# prompt user to train or load an RNN model
option_list = ['1','2']
option = ' '
while option not in option_list:  
    print('\nOPTIONS:')
    print('1 - Train a new RNN model')
    print('2 - Load an existing model')
    
    option = input('\nSelect an option by entering a number: \n')
    if option not in option_list:
        message = 'Invalid input: Input must be one of the following - '
        print(message, option_list)
        time.sleep(2)
        
#isFile = os.path.isfile("PerfectRNNModel.h5")

if option == '1':
    ## OPTION 1: TRAIN A NEW RNN MODEL
    
    print('\n********* NOW TRAINING A NEW RNN MODEL *********')
    time.sleep(3)
    rnn_attention_model.fit(trainX, trainY, epochs=epochs, verbose=0)
    # Evaluate model
    print('\n\n********** RNN training complete **********\n\n') 
    train_error = rnn_attention_model.evaluate(trainX, trainY)
    test_error = rnn_attention_model.evaluate(testX, testY)
    # Print error
    print("Train set MSE = ", train_error)
    print("Test set MSE = ", test_error)
    
    message = 'Enter the file name of the RNN Model you want to save: \n'
    load_file = input(message)
    #load_file = input('It must be a .h5 file')
    
    ## if file name does not end with '.h5', add '.h5' to the file name
    if load_file[-3:] != '.h5':
        load_file += '.h5'
    rnn_attention_model.save_weights(load_file)
    
       
elif option == '2':
    ## OPTION 2: LOAD RNN MODEL FROM FILE
    
    message = 'Enter the file name of the RNN Model you want to load: \n'
    load_file = input(message)
    #load_file = input('It must be a .h5 file')
    
    ## if file name does not end with '.h5', add '.h5' to the file name
    if load_file[-3:] != '.h5':
        load_file += '.h5'
    ## load the RNN model from load_file
    rnn_attention_model.load_weights(load_file)
    print('\n\n****** SUCCESSFULLY LOADED RNN MODEL ', load_file,'******')   
else:
    print('ERROR: INVALID OPTION SELECTED')
    
    raise ValueError()
    

message = 'Enter a sequence of consecutive 5 numbers (e.g., 10 11 12 13 14):\n'
user_input = input(message)
user_input = [int(i) for i in user_input.split(' ') if not len(i)== 0]
# X1

print("--- User entered input ---")
for i in user_input:
    scaled_user_input = data_scaler.transform(np.array([[i*i]]))[0][0]
    print(round(scaled_user_input, 4), end= "\t")
    scaled_user_input_sq = i*i
    print(round(scaled_user_input_sq, 4), end= "\t")
    print(i)
print("--- expected output for user entered input ---")
real_Y = (user_input[-1]+1)**2
scaled_Y = data_scaler.transform(np.array([[real_Y]]))
print('\t\033[93m',round(scaled_Y[0][0], 4), end= "\t")
print(round(real_Y, 4), end= "\t")
print(round(real_Y**0.5, 4), '\033[0m')
print("--- predicted output for user entered input ---")
scaled_user_input_for_model = np.array([data_scaler.transform(np.array([[scaled_element]]))[0] for scaled_element in [element**2 for element in user_input]])
predicted_scaled_Y = rnn_attention_model.predict(np.array([scaled_user_input_for_model]),verbose=0)
print('\t\033[92m',round(predicted_scaled_Y[0][0], 4), end= "\t")
predicted_Y = data_scaler.inverse_transform(np.array(predicted_scaled_Y))
print(round(predicted_Y[0][0], 4) , end='\t')
print(round(predicted_Y[0][0]**0.5, 4) , end='\t')
print("Error: %",round([abs(predicted_Y-real_Y)/real_Y*100][0][0][0], 4), '\033[0m')
print("------------------------------------------------------------------------")