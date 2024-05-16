# Written by: Hasan Suca Kayman
# City College of New York, CUNY
# April 2024
# chatbot_RNNwAtt.py is a simple chatbot that uses a Recurrent Neural Network with Attention mechanism to generate responses to user queries.

# Importing necessary libraries
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import os
import time
from datetime import datetime
import warnings


# Ignoring certain warnings for cleaner output
# warnings.filterwarnings('ignore')


# Setting a random seed for reproducibility
seed = 301
tf.keras.utils.set_random_seed(seed)

# Log path to store results
log_path = 'log_RNNwAtt.txt'
# Example queries path
queries_path = 'questions.txt'
# Data path
data_path = 'cleaned_dialogs.txt'

# Retrieve script arguments for the number of elements, batch size, units, and epochs
# first_n_elements = int(sys.argv[1])
# batch_size = int(sys.argv[2])
# number_of_hidden_layer = int(sys.argv[3])
# number_of_epochs = int(sys.argv[4])
# mode = int(sys.argv[5])

first_n_elements = 100
batch_size = 32
number_of_hidden_layer = 256
number_of_epochs = 100

# Preprocess a given sentence with start and end tokens
def tagger(w):
    w = '<start> ' + w.lower()  + ' <end>'
    return w

# Removes start and end tags from a sentence
def remove_tags(sentence):
    # Extract the content between start and end tags
    return sentence.split("<start>")[-1].split("<end>")[0]

# Tokenizes a list of sentences and pads them to ensure equal length
def tokenize(lang):
    # Create a tokenizer with no filters
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    
    # Fit tokenizer to the given text
    lang_tokenizer.fit_on_texts(lang)

    # Convert text to sequences of numbers
    tensor = lang_tokenizer.texts_to_sequences(lang)

    # Pad sequences
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,padding='post')

    return tensor, lang_tokenizer

# Loads a dataset and tokenizes both input and target languages
def load_dataset(data):

    # Unpack the given data into target and input languages
    targ_lang, inp_lang, = data

    # Tokenize both the input and target languages
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    # Return tokenized inputs, targets, and their tokenizers
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn = tf.keras.layers.RNN(tf.keras.layers.SimpleRNNCell(self.enc_units),
                                        return_sequences=True,
                                        return_state=True)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.rnn(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))
   
class Attention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
    
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn = tf.keras.layers.RNN(tf.keras.layers.SimpleRNNCell(self.dec_units),
                                        return_sequences=True,
                                        return_state=True)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = Attention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.rnn(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights

# Trains the model for one step and returns the loss
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

# Trains the model for one step and returns the loss
@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * batch_size, 1)

        for t in range(1, targ.shape[1]):

            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

# Given an input sentence, generate a response using the model
def ask(sentence):
    # Preprocess the input sentence
    sentence = tagger(sentence)

    # Tokenize the sentence and pad it to the max_length
    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],maxlen=vocab_inp_size,padding='post')
    
    # Convert to a tensor and reshape for the RNN model
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, number_of_hidden_layer))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)
    
    # Generate predictions
    # Extract the most likely tokens and convert them back to words
    for t in range(vocab_tar_size):
        predictions, dec_hidden, attention_weights = decoder(dec_input,dec_hidden,enc_out)
        attention_weights = tf.reshape(attention_weights, (-1, ))
        predicted_id = tf.argmax(predictions[0]).numpy() # Get the token with the highest probability
        result += targ_lang.index_word[predicted_id] + ' '
        if targ_lang.index_word[predicted_id] == '<end>':   # Stop if we reach the end token
            return remove_tags(result), remove_tags(sentence)
        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)
    
    # Return the original input and the predicted output
    return remove_tags(result), remove_tags(sentence)
# Create the file if it does not exist
if not os.path.exists(log_path):
    with open(log_path, 'w') as f:
        pass  # Creates an empty file

# List of queries for the model to respond to
file = open(queries_path,'r').read()
queries = [question for question in file.split('\n')]

# Read dialogs from a text file
file = open(data_path,'r').read()

# Split the dialog into questions and answers
qna_list = [f.split('\t') for f in file.split('\n')][:-1]
questions = [x[0] for x in qna_list][:first_n_elements+1]
answers = [x[1] for x in qna_list][:first_n_elements+1]

# Tagging the questions and answers
pre_questions = [tagger(w) for w in questions]
pre_answers = [tagger(w) for w in answers]

# Load and tokenize the dataset
data = pre_answers, pre_questions
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(data)
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

# Calculate the alphabet size from the training data
alphabet_size = len(input_tensor_train)

# Define the number of steps per epoch and embedding dimensions
steps_per_epoch = len(input_tensor_train)//batch_size
embedding_dim = 256

# Set the unit count for the RNN and vocabulary sizes
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

# Create a TensorFlow dataset from the training data
dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(alphabet_size)
dataset = dataset.batch(batch_size, drop_remainder=True) # Batch the dataset

# Fetch a batch of data from the dataset
example_input_batch, example_target_batch = next(iter(dataset))

# Initialize the encoder
encoder = Encoder(vocab_inp_size, embedding_dim, number_of_hidden_layer, batch_size)

# Sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)

# Initialize the Attention Layer
attention_layer = Attention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

# Initialize the decoder
decoder = Decoder(vocab_tar_size, embedding_dim, number_of_hidden_layer, batch_size)
sample_decoder_output, _, _ = decoder(tf.random.uniform((batch_size, 1)),sample_hidden, sample_output)

# Define the optimizer and loss function for training
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

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

# Prepare information about the current run
training_info = 'N{}_B{}_U{}_E{}\n'.format(first_n_elements,batch_size,number_of_hidden_layer,number_of_epochs)
training_info += "*"*20+"\n"
training_info += "Number of elements: "+str(first_n_elements)+"\n"
training_info += "Batch size: "+str(batch_size)+"\n"
training_info += "Unit size: "+str(number_of_hidden_layer)+"\n"
training_info += "Number of epochs: "+str(number_of_epochs)+"\n"

if option == '1':
    ## OPTION 1: TRAIN A NEW RNN MODEL
    
    print('\n********* NOW TRAINING A NEW RNN MODEL *********')
    time.sleep(3)

    # Start timing the training process
    start_time = time.time()

    # Store epoch information and training progress
    epoch_info = "\n"
    for epoch in range(1, number_of_epochs + 1):
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

        if(epoch + 10 > number_of_epochs):
            epoch_info+= 'Epoch:{:3d} Loss:{:.4f}\n'.format(epoch,total_loss / steps_per_epoch)
            print('Epoch:{:3d} Loss:{:.4f}\n'.format(epoch,total_loss / steps_per_epoch))

    end_time = time.time()   
    
    # Calculate the duration
    duration = end_time - start_time  # Duration in seconds
    training_info += "Duration:{:.4f} seconds\n".format(duration)
    training_info +=  epoch_info

    # Evaluate model
    print('\n\n********** RNN training complete **********\n\n') 

elif option == '2':
    ## OPTION 2: LOAD RNN MODEL FROM FILE
    
    message = 'Enter the file name of the RNN Model you want to load: \n'
    load_file = input(message)
    
    ## if file name does not end with '.h5', add '.h5' to the file name
    if load_file[-3:] != '.h5':
        load_file += '.h5'

    ## load the RNN model from load_file
    encoder.load_weights(os.path.join(load_file, 'encoder'))
    decoder.load_weights(os.path.join(load_file, 'decoder'))
    print('\n\n****** SUCCESSFULLY LOADED RNN MODEL ', load_file,'******')   
else:
    print('ERROR: INVALID OPTION SELECTED')
    
    raise ValueError()

# prompt user to test or save an RNN model
option_list = ['1','2','3','4']
while option != '4':
    option = ' '
    while option not in option_list:  
        print('\nOPTIONS:')
        print('1 - Test this RNN model with input')
        print('2 - Test queries in the file on an existing model.')
        print('3 - Save existing model and logs')
        print('4 - Exit')
        
        option = input('\nSelect an option by entering a number: \n')
        if option not in option_list:
            message = 'Invalid input: Input must be one of the following - '
            print(message, option_list)
            time.sleep(2)

    if option == '1':
    ## OPTION 1: TEST MODEL
        try:
            message = '!! Leave space before punction\n!!be sure your words in the dictionary\nEnter your input:'
            user_input = input(message)
            result, sentence = ask(user_input)
            training_info += 'Question: {} \n'.format(sentence)
            training_info += 'Predicted Output: {} \n'.format(result)
            print('Question: {} \n'.format(sentence))
            print('Predicted Output: {} \n'.format(result))
        
        except Exception as e:  # Catch all exceptions and store the exception object
            print(f"An error occurred: {e}. \nplease check your queries.")  # Print the error message
            print("\nEnsure that the words are in the dictionary, and write punctuation separately.")  # Print the error message
            continue

    elif option == '2':
    ## OPTION 2: TEST QUERIES
        try:
            for query in queries:
                result, sentence = ask(query)
                training_info += 'Question: {} \n'.format(sentence)
                training_info += 'Predicted Output: {} \n'.format(result)
                print('Question: {} \n'.format(sentence))
                print('Predicted Output: {} \n'.format(result))
        
        except Exception as e:  # Catch all exceptions and store the exception object
            print(f"An error occurred: {e}. \nplease check your queries.")  # Print the error message
            print("\nEnsure that the words are in the dictionary, and write punctuation separately.")  # Print the error message
            continue
    
    elif option == '3':
    ## OPTION 3: SAVE LOGS AND MODEL
        with open(log_path, 'a') as f:  
            f.write(training_info)   
    
        model_path = f'RNNwAtt_MODEL_N{first_n_elements}_B{batch_size}_U{number_of_hidden_layer}_E{number_of_epochs}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.h5'
        encoder.save(os.path.join(model_path, 'encoder'))
        decoder.save(os.path.join(model_path, 'decoder'))
        print("Succesfully Saved {}".format(model_path))
