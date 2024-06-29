# Written by: Hasan Suca Kayman
# The City College of New York, CUNY
# June 2024
# chatbot_RNNwAtt is a simple chatbot that uses a Recurrent Neural Network 
# with Attention mechanism to generate responses to user queries. Training 
# matrials is a set of dialogs taken from the text messages in the Internet.

# Importing necessary libraries
import tensorflow as tf
from datetime import datetime
import os
import time

# Setting a random seed for reproducibility
seed = 301
tf.keras.utils.set_random_seed(seed)

# Log file to store results
log_path = 'log_RNNwAtt_'
# Example queries file
queries_path = 'questions.txt'
# Data input file
data_path = 'cleaned_dialogs.txt'

noof_samples = 3000
batch_size = 16
noof_hidden_layers = 512
noof_epochs = 3000
embedding_dim = 256

# Tokenize a list of sentences and pad them to ensure equal length
# Convert words into numberic tokens and add padding based on max length
def tokenize(lang):
    # Create a tokenizer with no filters
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    
    # Fit tokenizer to the given text
    lang_tokenizer.fit_on_texts(lang)

    # Convert text to sequences of numbers
    tensor = lang_tokenizer.texts_to_sequences(lang)

    # Pad sequences
    tensor = \
          tf.keras.preprocessing.sequence.pad_sequences(tensor,padding='post')
    return tensor, lang_tokenizer

# Load a dataset and tokenize both input and target (expected) languages
def load_dataset(inp_lang, targ_lang):
    # Unpack the given data into target and input languages
    lang = inp_lang + targ_lang

    # Tokenize both the input and target languages
    data, lang_tokenizer = tokenize(lang)
    input_tensor, target_tensor = data[:len(inp_lang)], data[len(inp_lang):]

    # Return tokenized inputs, targets, and their tokenizers
    return input_tensor, target_tensor, lang_tokenizer

# Creates a simple RNNwAtt  model with specified parameters
def create_rnnwAtt_model(alph_size, embedding_dim, rnn_units):
    # Define a sequential model with embedding, RNNwAtt, and output layer
    inputs = tf.keras.Input(shape=(None,))
    # Embedding layer
    embedding = tf.keras.layers.Embedding(alph_size, embedding_dim)(inputs)
    # Simple RNN layer
    rnn_output = tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True)(embedding)
    # Attention layer
    attention_output = tf.keras.layers.Attention(use_scale=False, score_mode='dot')([rnn_output, rnn_output])
    # Dense output layer
    output = tf.keras.layers.Dense(alph_size, activation='softmax')(attention_output)
    # Build and compile the model
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model

# Given an input sentence, generate a response using the model
def predict(sentence, model, tokenizer, max_length=20):

    # Preprocess the input sentence
    preprocessed_sentence = '<start> ' + sentence.lower()  + ' <end>'

    # Tokenize the sentence and pad it to the max_length
    input_sequence = tokenizer.texts_to_sequences([preprocessed_sentence])
    input_tensor = \
          tf.keras.preprocessing.sequence.pad_sequences(input_sequence, \
            maxlen=max_length, padding='post')

    # Convert to a tensor and reshape for the RNN model
    input_tensor = tf.convert_to_tensor(input_tensor)

    # Generate predictions
    prediction = model.predict(input_tensor)
    
    # Extract the most likely tokens and convert them back to words
    predicted_sequence = ['<start> ']
    for pred in prediction[0]:
        # Get the token with the highest probability
        predicted_id = tf.argmax(pred, axis=-1).numpy()
        # Stop if we reach the end token
        if tokenizer.index_word.get(predicted_id, '') == " <end>":  
            break
        # Convert token to word
        predicted_word = tokenizer.index_word.get(predicted_id, '')  
        predicted_sequence.append(predicted_word)
    
    # Return the original input and the predicted output and removing tags
    result = ' '.join(predicted_sequence).split("<start>")[-1].split("<end>")[0]
    return result, sentence

# Read dialogs from a text file
file = open(data_path,'r').read()

# Split the dialog into questions and answers
qna_list = [f.split('\t') for f in file.split('\n')][:-1]
questions = [x[0] for x in qna_list][:noof_samples+1]
answers = [x[1] for x in qna_list][:noof_samples+1]

# Load and tokenize the dataset
input_tensor_train, target_tensor_train, lang_tokenizer = \
                                            load_dataset(questions, answers)

# Set the unit count for alphabet sizes
alph_size = len(lang_tokenizer.word_index)+1

# Create the RNN model with specified parameters
model = create_rnnwAtt_model(alph_size, embedding_dim, noof_hidden_layers)

# Define the optimizer and loss function for training
optimizer_a = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

model.compile(optimizer=optimizer_a, loss=loss_object)

# prompt user to train or load an RNN model
option_list = ['1','2']
option = ' '
while option not in option_list:  
    print('\nOPTIONS:')
    print('1 - Train a new RNN-Att model')
    print('2 - Load an existing model')
    
    option = input('\nSelect an option by entering a number: \n')
    if option not in option_list:
        message = 'Invalid input: Input must be one of the following - '
        print(message, option_list)
        time.sleep(2)

# Prepare information about the current run
training_info = 'N{}_B{}_U{}_EP{}_EM{}\n'.\
   format(noof_samples,
          batch_size,
          noof_hidden_layers,
          noof_epochs,
          embedding_dim)

training_info += "*"*20+"\n"
training_info += "Number of samples: "+str(noof_samples)+"\n"
training_info += "Batch size: "+str(batch_size)+"\n"
training_info += "Number of hidden layers: "+str(noof_hidden_layers)+"\n"
training_info += "Number of epochs: "+str(noof_epochs)+"\n"
training_info += "Embedding dimension: "+str(embedding_dim)+"\n"
training_info += "*"*20+"\n"
if option == '1':
    ## OPTION 1: TRAIN A NEW RNN MODEL
    
    print('\n********* NOW TRAINING A NEW RNN-Att MODEL *********')
    time.sleep(3)

    # Start timing the training process
    start_time = time.time()

    model.fit(input_tensor_train, target_tensor_train, epochs=noof_epochs, \
              batch_size = batch_size, shuffle=True)
    end_time = time.time()
    
    # Calculate the duration
    duration = end_time - start_time  # Duration in seconds
    training_info += "Duration:{:.4f} seconds\n".format(duration)

    # Evaluate model
    print('\n\n********** RNN-Att training complete **********\n\n') 

elif option == '2':
    ## OPTION 2: LOAD RNN MODEL FROM FILE
    
    message = 'Enter the file name of the RNN-Att Model you want to load: \n'
    load_file = input(message)
    
    ## if file name does not end with '.h5', add '.h5' to the file name
    if load_file[-3:] != '.h5':
        load_file += '.h5'

    ## load the RNN model from load_file
    model.load_weights(load_file)
    print('\n\n****** SUCCESSFULLY LOADED RNN-Att MODEL ', load_file,'******')   
else:
    print('ERROR: INVALID OPTION SELECTED')
    
    raise ValueError()

# prompt user to test or save an RNN model
option_list = ['1','2','3','4']
while option != '4':
    option = ' '
    while option not in option_list:  
        print('\nOPTIONS:')
        print('1 - Test this RNN-Att model with input')
        print('2 - Test queries given in a file')
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
            msg = 'Leave space before punctuation\n'
            msg += 'Make sure your words are in the dictionary\n'
            msg += 'Enter your input: '
            user_input = input(msg)
            result, sentence = \
                predict(user_input, model, lang_tokenizer, alph_size)
            training_info += 'Question: {} \n'.format(sentence)
            training_info += 'Predicted Output: {} \n'.format(result)
            print('Question: {} \n'.format(sentence))
            print('Predicted Output: {} \n'.format(result))
        except Exception as e:
            # Catch all exceptions and store the exception object
            print(f"An error occurred: {e}. \nplease check your queries.")
            print("\nEnsure that the words are in the dictionary,")
            print(" and write punctuation separately.")  
            continue

    elif option == '2':
    ## OPTION 2: TEST QUERIES
        try:
            # List of queries for the model to respond to
            file = open(queries_path,'r').read()
            queries = [question for question in file.split('\n')]
            for index,query in enumerate(queries):
                result, sentence = predict(query, model, lang_tokenizer, alph_size)
                training_info += '{}.Question: {} \n'.format(index+1,sentence)
                training_info += '{}.Predicted Output: {} \n'.format(index+1,result)
                print('{}.Question: {} \n'.format(index+1,sentence))
                print('{}.Predicted Output: {} \n'.format(index+1,result))
        
        except Exception as e:  
            # Catch all exceptions and store the exception object
            print(f"An error occurred: {e}. \nplease check your queries and files.")  
            print("\nEnsure that the words are in the dictionary, ")
            print("and write punctuation separately.")  
            continue
    
    elif option == '3':
    ## OPTION 3: SAVE LOGS AND MODEL
        times = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        
        with open(log_path+times+'.txt', 'w') as f:  
            f.write(training_info)   
        model_path = 'RNNwAtt_MODEL_N{}_B{}_U{}_EP{}_EM{}_{}.h5'.format(
            noof_samples,
            batch_size,
            noof_hidden_layers,
            noof_epochs,
            embedding_dim,
            times
        )
        model.save_weights(model_path)
        print("Succesfully Saved {}".format(model_path))

