# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 09:50:16 2024

@author: HSK
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 10:02:48 2024

@author: HSK
"""

import numpy as np

np.random.seed(42)
# Define the vocabulary and create mappings
vocab = ['h', 'e', 'l', 'o']
vocab_size = len(vocab)
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for i, ch in enumerate(vocab)}

def char_to_one_hot(char, vocab_size=vocab_size):
    one_hot = np.zeros((vocab_size,))
    one_hot[char_to_idx[char]] = 1
    return one_hot

def one_hot_to_char(one_hot):
    return idx_to_char[np.argmax(one_hot)]

# Convert a string to one-hot encoding
def string_to_one_hot(string):
    return np.array([char_to_one_hot(char) for char in string])

# Convert one-hot encoding to a string
def one_hot_to_string(one_hot_matrix):
    return ''.join([one_hot_to_char(one_hot) for one_hot in one_hot_matrix])

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class SimpleEmbedding:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = np.random.randn(vocab_size, embedding_dim)

    def forward(self, x):
        self.input = x
        return self.embeddings[x]

    def backward(self, dL_dE):
        dE_dV = np.zeros_like(self.embeddings)
        np.add.at(dE_dV, self.input, dL_dE)
        return dE_dV

class SimpleRNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.Wx = np.random.randn(input_dim, hidden_dim)
        self.Wh = np.random.randn(hidden_dim, hidden_dim)
        self.Wy = np.random.randn(hidden_dim, output_dim)
        self.bh = np.zeros((1, hidden_dim))
        self.by = np.zeros((1, output_dim))

    def forward(self, x):
        self.h = np.zeros((x.shape[0], x.shape[1], self.hidden_dim))
        self.outputs = []
        h_t = np.zeros((x.shape[0], self.hidden_dim))
        for t in range(x.shape[1]):
            h_t = np.tanh(np.dot(x[:, t, :], self.Wx) + np.dot(h_t, self.Wh) + self.bh)
            print(h_t.shape)
            y_t = softmax(np.dot(h_t, self.Wy) + self.by)
            self.h[:, t, :] = h_t
            self.outputs.append(y_t)
        self.outputs = np.array(self.outputs)
        return self.outputs

    def backward(self, dL_dy, x):
        dL_dWx = np.zeros_like(self.Wx)
        dL_dWh = np.zeros_like(self.Wh)
        dL_dWy = np.zeros_like(self.Wy)
        dL_dbh = np.zeros_like(self.bh)
        dL_dby = np.zeros_like(self.by)
        dL_dh_next = np.zeros((x.shape[0], self.hidden_dim))
        dL_dE = np.zeros_like(x)

        for t in reversed(range(x.shape[1])):
            dL_dy_t = dL_dy[t]
            dL_dWy += np.dot(self.h[:, t, :].T, dL_dy_t)
            dL_dby += np.sum(dL_dy_t, axis=0, keepdims=True)
            dL_dh = np.dot(dL_dy_t, self.Wy.T) + dL_dh_next
            dh_raw = (1 - self.h[:, t, :] ** 2) * dL_dh
            dL_dbh += np.sum(dh_raw, axis=0, keepdims=True)
            dL_dWx += np.dot(x[:, t, :].T, dh_raw)
            if t > 0:
                dL_dWh += np.dot(self.h[:, t-1, :].T, dh_raw)
            dL_dh_next = np.dot(dh_raw, self.Wh.T)
            dL_dE[:, t, :] = np.dot(dh_raw, self.Wx.T)

        return dL_dE, dL_dWx, dL_dWh, dL_dWy, dL_dbh, dL_dby

class SimpleRNNWithEmbedding:
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        self.embedding = SimpleEmbedding(vocab_size, embedding_dim)
        self.rnn = SimpleRNN(embedding_dim, hidden_dim, output_dim)

    def forward(self, x):
        embedded_x = self.embedding.forward(x)
        return self.rnn.forward(embedded_x)

    def backward(self, dL_dy, x):
        embedded_x = self.embedding.forward(x)
        dL_dE_rnn, dL_dWx, dL_dWh, dL_dWy, dL_dbh, dL_dby = self.rnn.backward(dL_dy, embedded_x)
        dE_dV = self.embedding.backward(dL_dE_rnn)
        return dE_dV, dL_dWx, dL_dWh, dL_dWy, dL_dbh, dL_dby

    # def update_weights(self, dE_dV, dL_dWx, dL_dWh, dL_dWy, dL_dbh, dL_dby, learning_rate):
    #     self.embedding.embeddings -= learning_rate * dE_dV
    #     self.rnn.Wx -= learning_rate * dL_dWx
    #     self.rnn.Wh -= learning_rate * dL_dWh
    #     self.rnn.Wy -= learning_rate * dL_dWy
    #     self.rnn.bh -= learning_rate * dL_dbh
    #     self.rnn.by -= learning_rate * dL_dby
    def update_weights(self, dE_dV, dL_dWx, dL_dWh, dL_dWy, dL_dbh, dL_dby, learning_rate):
        # Update embeddings
        #print("Updating embeddings")
        #print(f"Before update: {self.embedding.embeddings}")
        #print(f"Gradient: {dE_dV}")
        self.embedding.embeddings -= learning_rate * dE_dV
        #print(f"After update: {self.embedding.embeddings}\n")
        
        # Update input-to-hidden weights
        #print("Updating input-to-hidden weights (Wx)")
        #print(f"Before update: {self.rnn.Wx}")
        #print(f"Gradient: {dL_dWx}")
        self.rnn.Wx -= learning_rate * dL_dWx
        #print(f"After update: {self.rnn.Wx}\n")
        
        # Update hidden-to-hidden weights
        # print("Updating hidden-to-hidden weights (Wh)")
        # print(f"Before update: {self.rnn.Wh}")
        # print(f"Gradient: {dL_dWh}")
        self.rnn.Wh -= learning_rate * dL_dWh
        # print(f"After update: {self.rnn.Wh}\n")
        
        # Update hidden-to-output weights
        # print("Updating hidden-to-output weights (Wy)")
        # print(f"Before update: {self.rnn.Wy}")
        # print(f"Gradient: {dL_dWy}")
        self.rnn.Wy -= learning_rate * dL_dWy
        #print(f"After update: {self.rnn.Wy}\n")
        
        # Update biases for hidden layer
        # print("Updating biases for hidden layer (bh)")
        # print(f"Before update: {self.rnn.bh}")
        # print(f"Gradient: {dL_dbh}")
        self.rnn.bh -= learning_rate * dL_dbh
        # print(f"After update: {self.rnn.bh}\n")
        
        # Update biases for output layer
        # print("Updating biases for output layer (by)")
        # print(f"Before update: {self.rnn.by}")
        # print(f"Gradient: {dL_dby}")
        self.rnn.by -= learning_rate * dL_dby
        # print(f"After update: {self.rnn.by}\n")

        
def print_char_details(char, embedding_layer):
    one_hot = char_to_one_hot(char)
    embedding_value = embedding_layer.embeddings[char_to_idx[char]]
    print(f"Character: {char}")
    #print(f"One-hot encoding: {one_hot}")
    print(f"Embedding value: {embedding_value}")
    
# Hyperparameters
embedding_dim = 2
hidden_dim = 1
output_dim = vocab_size
learning_rate = 0.1

# Instantiate the RNN with Embedding
rnn_with_embedding = SimpleRNNWithEmbedding(vocab_size, embedding_dim, hidden_dim, output_dim)

# Training data
input_seq = "hell"
target_seq = "ello"

# Convert the input and target sequences to indices
x_train = np.array([[char_to_idx[char] for char in input_seq]])
y_train_indices = np.array([[char_to_idx[char] for char in target_seq]])


# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    if epoch%20 == 0:
        print("----Epoch ",epoch+1)
    #print("Embeding Weights:")
    #print(rnn_with_embedding.embedding.embeddings)
    # Forward pass
    output = rnn_with_embedding.forward(x_train)
    if epoch%20 == 0:
        print("Embedded char")
        print_char_details('h', rnn_with_embedding.embedding)
        print_char_details('e', rnn_with_embedding.embedding)
        print_char_details('l', rnn_with_embedding.embedding)
        print_char_details('o', rnn_with_embedding.embedding)
    # Convert output to one-hot encoding
    y_train = np.zeros_like(output)
    for t in range(y_train_indices.shape[1]):
        y_train[t, 0, y_train_indices[0, t]] = 1
    
    #print("Results:",np.argmax(output,axis=2))
    
    # Compute loss (Cross-Entropy Loss)
    loss = -np.sum(y_train * np.log(output)) / y_train.shape[0]
    if epoch%20 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss}')
    # Compute gradients (Cross-Entropy loss gradient)
    dL_dy = output - y_train

    # Backward pass
    dE_dV, dL_dWx, dL_dWh, dL_dWy, dL_dbh, dL_dby = rnn_with_embedding.backward(dL_dy, x_train)
    #print("Gradients:",dE_dV, dL_dWx, dL_dWh, dL_dWy, dL_dbh, dL_dby)
    
    # Update weights
    rnn_with_embedding.update_weights(dE_dV, dL_dWx, dL_dWh, dL_dWy, dL_dbh, dL_dby, learning_rate)
    
    
    
    

# Test the RNN with the input sequence
test_output = rnn_with_embedding.forward(x_train)
predicted_indices = np.argmax(test_output, axis=-1)[0]
predicted_chars = ''.join([idx_to_char[idx] for idx in predicted_indices])
#print("Predicted sequence:", predicted_chars)
