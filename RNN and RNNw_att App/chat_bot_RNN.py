# Written by: Hasan Suca Kayman
# City College of New York, CUNY
# April 2024
# chatbot_RNN is a simple chatbot that uses a Recurrent Neural Network to generate responses to user queries.

# Importing necessary libraries
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
import time
import warnings

# Ignoring certain warnings for cleaner output
# warnings.filterwarnings('ignore')


# Setting a random seed for reproducibility
seed = 301
tf.keras.utils.set_random_seed(seed)

# Log path to store results
log_path = 'log_RNN.txt'
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

# Creates a simple RNN model with specified parameters
def create_rnn_model(vocab_size, embedding_dim, rnn_units):

    # Define a sequential model with embedding, RNN, and output layer
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),  # Embedding layer
        tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True), # Simple RNN layer
        tf.keras.layers.Dense(vocab_size)  # Dense output layer
    ])
    return model

# Trains the model for one step and returns the loss
def train_step(model, optimizer, loss_function, inp, targ):
    # Initialize the loss
    loss = 0

    # Use GradientTape to record operations for automatic differentiation
    with tf.GradientTape() as tape:

        # Make predictions using the model
        predictions = model(inp)
        # Calculate loss for each target token
        for t in range(1, targ.shape[1]):
            loss += loss_function(targ[:, t], predictions[:, t, :])

    # Calculate gradients and apply them
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Return average loss per target token
    return loss / int(targ.shape[1])

# Given an input sentence, generate a response using the model
def ask(sentence, model, inp_tokenizer, targ_tokenizer, max_length=20):

    # Preprocess the input sentence
    preprocessed_sentence = tagger(sentence)

    # Tokenize the sentence and pad it to the max_length
    input_sequence = inp_tokenizer.texts_to_sequences([preprocessed_sentence])
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_sequence, maxlen=max_length, padding='post')

    # Convert to a tensor and reshape for the RNN model
    input_tensor = tf.convert_to_tensor(input_tensor)

    # Generate predictions
    prediction = model.predict(input_tensor)
    
    # Extract the most likely tokens and convert them back to words
    predicted_sequence = ['<start> ']
    for pred in prediction[0]:
        predicted_id = tf.argmax(pred, axis=-1).numpy()  # Get the token with the highest probability
        if targ_tokenizer.index_word.get(predicted_id, '') == " <end>":  # Stop if we reach the end token
            break
        predicted_word = targ_tokenizer.index_word.get(predicted_id, '')  # Convert token to word
        predicted_sequence.append(predicted_word)
    
    # Return the original input and the predicted output
    result = ' '.join(predicted_sequence)
    return remove_tags(result), sentence

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

# Create the RNN model with specified parameters
model = create_rnn_model(vocab_inp_size, embedding_dim, number_of_hidden_layer)

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
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(dataset):
            total_loss += train_step(model, optimizer, loss_object, inp, targ)


        if(epoch + 10 > number_of_epochs):
            epoch_info += 'Epoch:{:3d}, Loss:{:.4f} \n'.format(epoch,sum(total_loss)/len(total_loss))

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
    model.load_weights(load_file)
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
            result, sentence = ask(user_input, model, inp_lang, targ_lang, vocab_tar_size)
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
                result, sentence = ask(query, model, inp_lang, targ_lang, vocab_tar_size)
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
    
        model_path = f'RNN_MODEL_N{first_n_elements}_B{batch_size}_U{number_of_hidden_layer}_E{number_of_epochs}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.h5'
        model.save_weights(model_path)
        print("Succesfully Saved {}".format(model_path))

