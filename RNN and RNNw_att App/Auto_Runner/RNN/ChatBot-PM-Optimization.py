# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 18:32:49 2024

@author: HSK
"""
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import unicodedata
import re
import numpy as np
import os
import io
import time
import warnings
warnings.filterwarnings('ignore')

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')

def expand_contractions(text):
    def expand(match):
        contraction = match.group(0).lower()  
        if contraction in contraction_map:
            return contraction_map[contraction]
        return contraction  
    
    contraction_map = {
        "i'm": "I am",
        "i've": "I have",
        "it's": "it is",
        "how's": "how is",
        "everything's": "everything is",
        "haven't": "have not",
        "shouldn't": "should not",
        "wasn't": "was not",
        "can't": "cannot",
        "i'd": "I would",
        "doesn't": "does not",
        "you're": "you are",
        "wouldn't": "would not",
        "that's": "that is",
        "didn't": "did not",
        "isn't": "is not",
        "don't": "do not",
        "what's": "what is",
        "it'll": "it will",
        "what'll": "what will",
        "let's": "let us",
        "i'll": "I will",
        "she's": "she is",
        "there's": "there is",
        "might've": "might have",
        "you've": "you have",
        "weren't": "were not",
        "macy's": "Macy's (store)",
        "couldn't": "could not",
        "night's": "night is",
        "must've": "must have",
        "should've": "should have",
        "would've": "would have",
        "didn't?": "did not?",
        "mom's": "mom is",
        "they're": "they are",
        "where's": "where is",
        "here's": "here is",
        "we're": "we are",
        "you'll": "you will",
        "he'll": "he will",
        "he's": "he is",
        "won't": "will not",
        "mcdonald's": "McDonald's",
        "grandma's": "grandma is",
        "people's": "people are",
        "something's": "something is",
        "you'd": "you would",
        "aren't": "are not",
        "nothing's": "nothing is",
        "a's": "A is",
        "b's": "B is",
        "why's": "why is",
        "shoes—they're": "shoes — they are",
        "mother's": "mother is",
        "they'll": "they will",
        "when's": "when is",
        "dad's": "dad is",
        "driver's": "driver is",
        "o'clock": "of the clock",
    }
    
    pattern = re.compile(r"\b(" + "|".join(contraction_map.keys()) + r")\b", re.IGNORECASE)
    expanded_text = pattern.sub(expand, text)
    return expanded_text

def preprocess_sentence(w):
    w = expand_contractions(w.lower().strip())
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.strip()
    w = '<start> ' + w + ' <end>'
    return w

def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,padding='post')

    return tensor, lang_tokenizer

def load_dataset(data):
    # creating cleaned input, output pairs
    targ_lang, inp_lang, = data

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

def create_rnn_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True, return_state=False),
        tf.keras.layers.Dense(vocab_size)  # Output layer
    ])
    return model

def remove_tags(sentence):
    return sentence.split("<start>")[-1].split("<end>")[0]

# Training and evaluation
def train_step(model, optimizer, loss_function, inp, targ):
    loss = 0
    with tf.GradientTape() as tape:
        predictions = model(inp)
        for t in range(1, targ.shape[1]):
            loss += loss_function(targ[:, t], predictions[:, t, :])

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss / int(targ.shape[1])

def ask(sentence, model, inp_tokenizer, targ_tokenizer, max_length=20):
    """
    Given an input sentence, preprocess it, convert it to a tensor, and use the model to generate a response.
    """
    # Preprocess the input sentence
    preprocessed_sentence = preprocess_sentence(sentence)

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

file_path = 'log.txt'

if not os.path.exists(file_path):
    with open(file_path, 'w') as f:
        pass  # Creates an empty file


        
queries=["are you enjoying it there?",
            "i really wish it was not so hot every day.",
            "any rain right now would be pointless.",
            "how come?",
            "is it going to be perfect beach weather?",
            "is it hot lately?",
            "have you been pretty good?",
            "i am absolutely going to school, where are you going?",
            "where are going?",
            "it is the middle of school.",
            "that would be perfect.",
            'since it is hot outside.',
            "it would be perfect if it rained.",
            "why?",
            "the stars look perfect.",
            "do you think it is perfect? ",
            "you like the hot?",
            "the sky looks clean.",
            "rain does make it smell cleaner.",
            "i love rain at night."]
           
firstNElement = int(sys.argv[1])
BATCH_SIZE = int(sys.argv[2])
units = int(sys.argv[3])
EPOCHS = int(sys.argv[4])
file = open('dialogs.txt','r').read()
qna_list = [f.split('\t') for f in file.split('\n')][:-1]
questions = [x[0] for x in qna_list][:firstNElement+1]
answers = [x[1] for x in qna_list][:firstNElement+1]

pre_questions = [preprocess_sentence(w) for w in questions]
pre_answers = [preprocess_sentence(w) for w in answers]

data = pre_answers, pre_questions
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(data)
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

ALPHABET_SIZE = len(input_tensor_train)
#
#BATCH_SIZE = 64
#
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
#
#units = 1024
#
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(ALPHABET_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
example_input_batch, example_target_batch = next(iter(dataset))

# Create RNN model
model = create_rnn_model(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
#
# EPOCHS = 300
#
start_time = time.time()
epochInfo=""
epochInfo+="*"*20+"\n"
epochInfo+="Number of elements: "+str(firstNElement)+"\n"
epochInfo+="Batch size: "+str(BATCH_SIZE)+"\n"
epochInfo+="Unit size: "+str(units)+"\n"
epochInfo+="Number of epochs: "+str(EPOCHS)+"\n"
infos="\n"
for epoch in range(1, EPOCHS + 1):
    total_loss = 0
    for (batch, (inp, targ)) in enumerate(dataset):
        total_loss += train_step(model, optimizer, loss_object, inp, targ)


    if(epoch + 10 > EPOCHS):
        infos += 'Epoch:{:3d}, Loss:{:.4f} \n'.format(epoch,sum(total_loss)/len(total_loss))

end_time = time.time()   
 
# Calculate the duration
duration = end_time - start_time  # Duration in seconds
epochInfo += "Duration:{:.4f} seconds\n".format(duration)
epochInfo +=  infos
for query in queries:
    result, sentence = ask(query, model, inp_lang, targ_lang, vocab_tar_size)
    epochInfo+='Question: {} \n'.format(sentence)
    epochInfo+='Predicted Output: {} \n'.format(result)


print(epochInfo)
with open(file_path, 'a') as f:  
    f.write(epochInfo) 
    
model_path = f'RNN_MODEL_N{firstNElement}_B{BATCH_SIZE}_U{units}_E{EPOCHS}.h5'
model.save_weights(model_path)
