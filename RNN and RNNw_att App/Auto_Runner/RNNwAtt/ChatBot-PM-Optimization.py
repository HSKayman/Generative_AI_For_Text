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

def remove_tags(sentence):
    return sentence.split("<start>")[-1].split("<end>")[0]

def evaluate(sentence):
    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],maxlen=vocab_inp_size,padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(vocab_tar_size):
        predictions, dec_hidden, attention_weights = decoder(dec_input,dec_hidden,enc_out)
        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        predicted_id = tf.argmax(predictions[0]).numpy()
        result += targ_lang.index_word[predicted_id] + ' '
        if targ_lang.index_word[predicted_id] == '<end>':
            return remove_tags(result), remove_tags(sentence)
        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return remove_tags(result), remove_tags(sentence)

def ask(sentence):
    result, sentence = evaluate(sentence)
    return result, sentence
       

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

        for t in range(1, targ.shape[1]):

            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

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
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

# sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)

attention_layer = Attention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),sample_hidden, sample_output)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
#
# EPOCHS = 300
#
epochInfo=""
epochInfo+="*"*20+"\n"
epochInfo+="First # Element: "+str(firstNElement)+"\n"
epochInfo+="Batch Size: "+str(BATCH_SIZE)+"\n"
epochInfo+="Unit Size: "+str(units)+"\n"
epochInfo+="Epochs: "+str(EPOCHS)+"\n"
for epoch in range(1, EPOCHS + 1):
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

    if(epoch + 10 > EPOCHS):
        epochInfo+= 'Epoch:{:3d} Loss:{:.4f}\n'.format(epoch,total_loss / steps_per_epoch)
        
for query in queries:
    result, sentence = ask(query)
    epochInfo+='Question: {} \n'.format(sentence)
    epochInfo+='Predicted answer: {} \n'.format(result)


print(epochInfo)
with open(file_path, 'a') as f:  
    f.write(epochInfo) 
model_path = 'RNN_MODEL_N{}_B{}_U{}_E{}_'.format(firstNElement,BATCH_SIZE,units,EPOCHS)
# Save the entire model
encoder.save(os.path.join(model_path, 'encoder'))
decoder.save(os.path.join(model_path, 'decoder'))
