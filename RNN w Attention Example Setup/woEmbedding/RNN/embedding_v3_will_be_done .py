# # -*- coding: utf-8 -*-
# """
# Created on Sat Jun 15 09:50:16 2024

# @author: HSK
# """
# # -*- coding: utf-8 -*-
# """
# Created on Sat Jun 15 10:02:48 2024

# @author: HSK
# """

# import numpy as np

# np.random.seed(42)
# # Define the vocabulary and create mappings
# vocab = ['h', 'e', 'l', 'o']
# vocab_size = len(vocab)
# char_to_idx = {ch: i for i, ch in enumerate(vocab)}
# idx_to_char = {i: ch for i, ch in enumerate(vocab)}

# def char_to_one_hot(char, vocab_size=vocab_size):
#     one_hot = np.zeros((vocab_size,))
#     one_hot[char_to_idx[char]] = 1
#     return one_hot

# def one_hot_to_char(one_hot):
#     return idx_to_char[np.argmax(one_hot)]

# # Convert a string to one-hot encoding
# def string_to_one_hot(string):
#     return np.array([char_to_one_hot(char) for char in string])

# # Convert one-hot encoding to a string
# def one_hot_to_string(one_hot_matrix):
#     return ''.join([one_hot_to_char(one_hot) for one_hot in one_hot_matrix])

# def softmax(x):
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum(axis=-1, keepdims=True)

# class SimpleEmbedding:
#     def __init__(self, vocab_size, embedding_dim):
#         self.vocab_size = vocab_size
#         self.embedding_dim = embedding_dim
#         self.embeddings = np.random.randn(vocab_size, embedding_dim)

#     def forward(self, x):
#         self.input = x
#         return self.embeddings[x]

#     def backward(self, dL_dE):
#         dE_dV = np.zeros_like(self.embeddings)
#         np.add.at(dE_dV, self.input, dL_dE)
#         return dE_dV

# class SimpleRNN:
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         self.Wx = np.random.randn(input_dim, hidden_dim)
#         self.Wh = np.random.randn(hidden_dim, hidden_dim)
#         self.Wy = np.random.randn(hidden_dim, output_dim)
#         self.bh = np.zeros((1, hidden_dim))
#         self.by = np.zeros((1, output_dim))

#     def forward(self, x):
#         self.h = np.zeros((x.shape[0], x.shape[1], self.hidden_dim))
#         self.outputs = []
#         h_t = np.zeros((x.shape[0], self.hidden_dim))
#         for t in range(x.shape[1]):
#             h_t = np.tanh(np.dot(x[:, t, :], self.Wx) + np.dot(h_t, self.Wh) + self.bh)
#             y_t = softmax(np.dot(h_t, self.Wy) + self.by)
#             self.h[:, t, :] = h_t
#             self.outputs.append(y_t)
#         self.outputs = np.array(self.outputs)
#         return self.outputs

#     def backward(self, dL_dy, x):
#         dL_dWx = np.zeros_like(self.Wx)
#         dL_dWh = np.zeros_like(self.Wh)
#         dL_dWy = np.zeros_like(self.Wy)
#         dL_dbh = np.zeros_like(self.bh)
#         dL_dby = np.zeros_like(self.by)
#         dL_dh_next = np.zeros((x.shape[0], self.hidden_dim))
#         dL_dE = np.zeros_like(x)

#         for t in reversed(range(x.shape[1])):
#             dL_dy_t = dL_dy[t]
#             dL_dWy += np.dot(self.h[:, t, :].T, dL_dy_t)
#             dL_dby += np.sum(dL_dy_t, axis=0, keepdims=True)
#             dL_dh = np.dot(dL_dy_t, self.Wy.T) + dL_dh_next
#             dh_raw = (1 - self.h[:, t, :] ** 2) * dL_dh
#             dL_dbh += np.sum(dh_raw, axis=0, keepdims=True)
#             dL_dWx += np.dot(x[:, t, :].T, dh_raw)
#             if t > 0:
#                 dL_dWh += np.dot(self.h[:, t-1, :].T, dh_raw)
#             dL_dh_next = np.dot(dh_raw, self.Wh.T)
#             dL_dE[:, t, :] = np.dot(dh_raw, self.Wx.T)

#         return dL_dE, dL_dWx, dL_dWh, dL_dWy, dL_dbh, dL_dby

# class SimpleRNNWithEmbedding:
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
#         self.embedding = SimpleEmbedding(vocab_size, embedding_dim)
#         self.rnn = SimpleRNN(embedding_dim, hidden_dim, output_dim)

#     def forward(self, x):
#         embedded_x = self.embedding.forward(x)
#         return self.rnn.forward(embedded_x)

#     def backward(self, dL_dy, x):
#         embedded_x = self.embedding.forward(x)
#         dL_dE_rnn, dL_dWx, dL_dWh, dL_dWy, dL_dbh, dL_dby = self.rnn.backward(dL_dy, embedded_x)
#         dE_dV = self.embedding.backward(dL_dE_rnn)
#         return dE_dV, dL_dWx, dL_dWh, dL_dWy, dL_dbh, dL_dby

#     # def update_weights(self, dE_dV, dL_dWx, dL_dWh, dL_dWy, dL_dbh, dL_dby, learning_rate):
#     #     self.embedding.embeddings -= learning_rate * dE_dV
#     #     self.rnn.Wx -= learning_rate * dL_dWx
#     #     self.rnn.Wh -= learning_rate * dL_dWh
#     #     self.rnn.Wy -= learning_rate * dL_dWy
#     #     self.rnn.bh -= learning_rate * dL_dbh
#     #     self.rnn.by -= learning_rate * dL_dby
#     def update_weights(self, dE_dV, dL_dWx, dL_dWh, dL_dWy, dL_dbh, dL_dby, learning_rate):
#         # Update embeddings
#         #print("Updating embeddings")
#         #print(f"Before update: {self.embedding.embeddings}")
#         #print(f"Gradient: {dE_dV}")
#         self.embedding.embeddings -= learning_rate * dE_dV
#         #print(f"After update: {self.embedding.embeddings}\n")
        
#         # Update input-to-hidden weights
#         #print("Updating input-to-hidden weights (Wx)")
#         #print(f"Before update: {self.rnn.Wx}")
#         #print(f"Gradient: {dL_dWx}")
#         self.rnn.Wx -= learning_rate * dL_dWx
#         #print(f"After update: {self.rnn.Wx}\n")
        
#         # Update hidden-to-hidden weights
#         # print("Updating hidden-to-hidden weights (Wh)")
#         # print(f"Before update: {self.rnn.Wh}")
#         # print(f"Gradient: {dL_dWh}")
#         self.rnn.Wh -= learning_rate * dL_dWh
#         # print(f"After update: {self.rnn.Wh}\n")
        
#         # Update hidden-to-output weights
#         # print("Updating hidden-to-output weights (Wy)")
#         # print(f"Before update: {self.rnn.Wy}")
#         # print(f"Gradient: {dL_dWy}")
#         self.rnn.Wy -= learning_rate * dL_dWy
#         #print(f"After update: {self.rnn.Wy}\n")
        
#         # Update biases for hidden layer
#         # print("Updating biases for hidden layer (bh)")
#         # print(f"Before update: {self.rnn.bh}")
#         # print(f"Gradient: {dL_dbh}")
#         self.rnn.bh -= learning_rate * dL_dbh
#         # print(f"After update: {self.rnn.bh}\n")
        
#         # Update biases for output layer
#         # print("Updating biases for output layer (by)")
#         # print(f"Before update: {self.rnn.by}")
#         # print(f"Gradient: {dL_dby}")
#         self.rnn.by -= learning_rate * dL_dby
#         # print(f"After update: {self.rnn.by}\n")

        
# def print_char_details(char, embedding_layer):
#     one_hot = char_to_one_hot(char)
#     embedding_value = embedding_layer.embeddings[char_to_idx[char]]
#     print(f"Character: {char}")
#     #print(f"One-hot encoding: {one_hot}")
#     print(f"Embedding value: {embedding_value}")
    
# # Hyperparameters
# embedding_dim = 2
# hidden_dim = 1
# output_dim = vocab_size
# learning_rate = 0.1

# # Instantiate the RNN with Embedding
# rnn_with_embedding = SimpleRNNWithEmbedding(vocab_size, embedding_dim, hidden_dim, output_dim)

# # Training data
# input_seq = "hell"
# target_seq = "ello"

# # Convert the input and target sequences to indices
# x_train = np.array([[char_to_idx[char] for char in input_seq]])
# y_train_indices = np.array([[char_to_idx[char] for char in target_seq]])


# # Training loop
# num_epochs = 100
# for epoch in range(num_epochs):
#     if epoch%20 == 0:
#         print("----Epoch ",epoch+1)
#     #print("Embeding Weights:")
#     #print(rnn_with_embedding.embedding.embeddings)
#     # Forward pass
#     output = rnn_with_embedding.forward(x_train)
#     if epoch%20 == 0:
#         print("Embedded char")
#         print_char_details('h', rnn_with_embedding.embedding)
#         print_char_details('e', rnn_with_embedding.embedding)
#         print_char_details('l', rnn_with_embedding.embedding)
#         print_char_details('o', rnn_with_embedding.embedding)
#     # Convert output to one-hot encoding
#     y_train = np.zeros_like(output)
#     for t in range(y_train_indices.shape[1]):
#         y_train[t, 0, y_train_indices[0, t]] = 1
    
#     #print("Results:",np.argmax(output,axis=2))
    
#     # Compute loss (Cross-Entropy Loss)
#     loss = -np.sum(y_train * np.log(output)) / y_train.shape[0]
#     if epoch%20 == 0:
#         print(f'Epoch {epoch+1}, Loss: {loss}')
#     # Compute gradients (Cross-Entropy loss gradient)
#     dL_dy = output - y_train

#     # Backward pass
#     dE_dV, dL_dWx, dL_dWh, dL_dWy, dL_dbh, dL_dby = rnn_with_embedding.backward(dL_dy, x_train)
#     #print("Gradients:",dE_dV, dL_dWx, dL_dWh, dL_dWy, dL_dbh, dL_dby)
    
#     # Update weights
#     rnn_with_embedding.update_weights(dE_dV, dL_dWx, dL_dWh, dL_dWy, dL_dbh, dL_dby, learning_rate)
    
    
    
    

# # Test the RNN with the input sequence
# test_output = rnn_with_embedding.forward(x_train)
# predicted_indices = np.argmax(test_output, axis=-1)[0]
# predicted_chars = ''.join([idx_to_char[idx] for idx in predicted_indices])
# #print("Predicted sequence:", predicted_chars)


import numpy as np
np.random.seed(124)
def print_colored(text, color):
    """Prints text in a specified color using ANSI escape codes."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }
    #print(f"{colors.get(color, colors['reset'])}{text}{colors['reset']}")

print_colored("="*60 + " INITIALIZATION " + "="*60, "cyan")

# Vocabulary setup
vocab = ['h', 'e', 'l','o']
vocab_size = len(vocab)
#print_colored(f"Vocabulary Size: {vocab_size}", "green")

# One-hot encoding setup
char_to_onehot = {ch: np.eye(vocab_size)[i] for i, ch in enumerate(vocab)}
onehot_to_char = {i: ch for i, ch in enumerate(vocab)}

#print_colored("\nCharacter to One-hot Encoding:", "yellow")
#for char, encoding in char_to_onehot.items():
#    print_colored(f"{char}: {encoding}", "yellow")

# Initialize parameters
hidden_size = 1  # Size of the hidden layer
U = np.random.randn(hidden_size, 2) * 0.5  # Input to hidden
W = np.random.randn(hidden_size, hidden_size) * 0.4  # Hidden to hidden
V = np.random.randn(vocab_size, hidden_size) * 0.3  # Hidden to output
b = np.zeros((hidden_size, 1))  # Hidden bias
c = np.zeros((vocab_size, 1))  # Output bias
E = np.random.randn(vocab_size, 2) * 0.5
#print_colored("\nParameters Initialized:", "magenta")

# Input ("hell") and target ("ello") sequences
x_sequence = 'hell'
y_sequence = 'ello'
#print_colored("\nInput and Target Sequences:", "green")
#for i in range(len(x_sequence)):
    #print_colored(f"x[{i}] = '{x_sequence[i]}', y[{i}] = '{y_sequence[i]}'", "green")

# Forward Pass function definition
def forward_pass(x_sequence, U, W, V, b, c,E):
    xs, hs, os, ys = {}, {}, {}, {}
    hs[-1] = np.zeros((hidden_size, 1))  # Initial hidden state
    loss = 0
    #print(1)
    for t in range(len(x_sequence)):
        #xs[t] = char_to_onehot[x_sequence[t]].reshape(-1, 1)  # One-hot
        xs[t] = np.dot(char_to_onehot[x_sequence[t]],E)   # One-hot
        hs[t] = np.tanh(np.dot(U, xs[t]) + np.dot(W, hs[t-1]) + b)  # Hidden state
        os[t] = np.dot(V, hs[t]) + c  # Raw output
        ys[t] = np.exp(os[t]) / np.sum(np.exp(os[t]))  # Softmax output
        correct_char_vec = char_to_onehot[y_sequence[t]].reshape(-1, 1)
        loss += -np.sum(correct_char_vec * np.log(ys[t]))  # Cross-entropy loss
        #print("y",ys[t])
        #print(-np.sum(correct_char_vec * np.log(ys[t])))
        print_colored(f"\nStep {t+1}:", "cyan")
        print_colored(f"x_{t+1} = One-hot encoding for '{x_sequence[t]}'", "yellow")
        print_colored(f"\nHidden State at t={t+1}:", "cyan")
        print_colored(f"x_{t} = x^o_t * E", "cyan")
        print_colored(f"x_{t} = {char_to_onehot[x_sequence[t]]} * {E}", "cyan")
        print_colored(f"x_{t} = {xs[t]}", "cyan")
        print_colored(f"h_{t+1} = tanh(U * x_{t+1} + W * h_{t} + b)", "cyan")
        print_colored(f"= tanh({U} * {xs[t]} + {W} * {hs[t-1]} + {b})","red")
        print_colored(f"h_{t+1} = {hs[t]}", "cyan")

        print_colored(f"o_{t+1} = V * h_{t+1} + c", "blue")
        print_colored(f"= {V} * {hs[t]} + {c}", "blue")
        print_colored(f"o_{t+1} = {os[t]}", "blue")
        print_colored(f"y_hat{t+1} = softmax(y_hat{t+1})", "magenta")
        print_colored(f"y_hat{t+1} = {ys[t]}", "magenta")

        print_colored(f"Loss{t+1} = -y * ln(y_hat{t+1})", "red")
        print_colored(f"Loss{t+1} = -{char_to_onehot[y_sequence[t]].reshape(-1, 1)} * ln({ys[t]})", "red")
        print_colored(f"Loss at step {t+1}: {-np.sum(correct_char_vec * np.log(ys[t]))}", "red")
    print("Total Loss:",loss)    
    return loss, xs, hs, ys

def bptt(xs, hs, ys, U, W, V, b, c):
    S=int(len(xs))
    dU2, dW2, dV2 = np.zeros_like(U), np.zeros_like(W), np.zeros_like(V)
    db2, dc2 = np.zeros_like(b), np.zeros_like(c)
    de2 = np.zeros_like(E)
    for t in reversed(range(len(x_sequence))):
        dy = np.copy(ys[t])
        correct_char_vec = char_to_onehot[y_sequence[t]].reshape(-1, 1)
        dy -= correct_char_vec  # Gradient of softmax (y - t)
        dV2 += np.dot(dy, hs[t].T)
        dc2 += dy
    # print("dc2:",dc2)
    # print("dV2:",dV2)
    for t in range(0, S):
        for k in range(0, t+1):
            dy = np.copy(ys[t])
            correct_char_vec = char_to_onehot[y_sequence[t]].reshape(-1, 1)
            dy -= correct_char_vec  # Gradient of softmax (y - t) wrt omega
            beforeProduct=np.dot(V.T, dy)  # Gradient of omega wrt h
            product_of_gradients =  1
            for j in range(k,t):
                product_of_gradients *= (1 - hs[j+1] ** 2) * W
                #print(j,product_of_gradients)
            product_of_gradients*= hs[k-1]*(1 - hs[k]**2) #if k-1 > -1 else  np.zeros_like(W) 
            dW2+= np.dot(product_of_gradients,beforeProduct)
    #print("dW2:",dW2)   
    for t in range(0, S):
        for k in range(0, t+1):
            dy = np.copy(ys[t])
            correct_char_vec = char_to_onehot[y_sequence[t]].reshape(-1, 1)
            dy -= correct_char_vec  # Gradient of softmax (y - t) wrt omega
            beforeProduct=np.dot(V.T, dy)  # Gradient of omega wrt h
            product_of_gradients =  np.ones_like(de2)
            for j in range(k,t):
                product_of_gradients *= W.T*(1 - hs[j+1] ** 2)
            product_of_gradients*= np.dot(char_to_onehot[x_sequence[t]].reshape(4, 1), U) * (1 - hs[k]**2) #if k-1 > -1 else  np.zeros_like(W) 
            #print(t,k,product_of_gradients)
            de2+= product_of_gradients*beforeProduct 
            
    for t in range(0, S):
        for k in range(0, t+1):
            dy = np.copy(ys[t])
            correct_char_vec = char_to_onehot[y_sequence[t]].reshape(-1, 1)
            dy -= correct_char_vec  # Gradient of softmax (y - t) wrt omega
            beforeProduct=np.dot(V.T, dy)  # Gradient of omega wrt h
            product_of_gradients =  np.ones_like(dU2)
            for j in range(k,t):
                product_of_gradients *= W.T*(1 - hs[j+1] ** 2)
            product_of_gradients*= xs[k].T * (1 - hs[k]**2) #if k-1 > -1 else  np.zeros_like(W) 
            dU2+= product_of_gradients*beforeProduct 
    # print("dU2:",dU2)   
    
    for t in range(0, S):
        for k in range(0, t+1):
            dy = np.copy(ys[t])
            correct_char_vec = char_to_onehot[y_sequence[t]].reshape(-1, 1)
            dy -= correct_char_vec  # Gradient of softmax (y - t) wrt omega
            beforeProduct=np.dot(V.T, dy)  # Gradient of omega wrt h
            product_of_gradients =  1
            for j in range(k,t):
                product_of_gradients *= (1 - hs[j+1] ** 2) * W
            product_of_gradients*= (1 - hs[k]**2) #if k-1 > -1 else  np.zeros_like(W) 
            db2+= product_of_gradients*beforeProduct
    # print("db2:",db2)   
    # dU, dW, dV = np.zeros_like(U), np.zeros_like(W), np.zeros_like(V)
    # db, dc = np.zeros_like(b), np.zeros_like(c)
    # dhnext = np.zeros_like(hs[0])
    # for t in reversed(range(len(x_sequence))):
    #     dy = np.copy(ys[t])
    #     correct_char_vec = char_to_onehot[y_sequence[t]].reshape(-1, 1)
    #     dy -= correct_char_vec  # Gradient of softmax (y - t)
    #     dV += np.dot(dy, hs[t].T)
    #     dc += dy
        
    #     dh = np.dot(V.T, dy) + dhnext  # Backprop into h
    #     dhraw = (1 - hs[t] ** 2) * dh  # Backprop through tanh nonlinearity
    #     db += dhraw
    #     dU += np.dot(dhraw, xs[t].T)
    #     dW += np.dot(dhraw, hs[t-1].T if t > 0 else np.zeros_like(hs[t-1].T))
    #     dhnext = np.dot(W.T, dhraw)
        
    for dparam in [dU2, dW2, dV2, db2, dc2]:
        np.clip(dparam, -5, 5, out=dparam)

    return dU2, dW2, dV2, db2, dc2, de2


a=[]
for epoch in range(1000):
    
    print_colored(("="*60)+"#"+str(epoch)+" EPOCH "+("="*60)+"\n\n", "cyan")
    print_colored("E =\n{}".format(E), "green")
    print_colored("U =\n{}".format(U), "green")
    print_colored("W =\n{}".format(W), "blue")
    print_colored("b =\n{}".format(b), "red")

    print_colored("V =\n{}".format(V), "magenta")
    print_colored("c =\n{}".format(c), "yellow")
    
    # print_colored(("="*60)+"FORWARD PASS"+("="*60)+"\n\n", "cyan")
    loss, xs, hs, ys = forward_pass(x_sequence, U, W, V, b, c,E)
    a.append(loss)
    # print_colored(("="*60)+"BACKPROP PASS"+("="*60)+"\n\n", "cyan")
    dU, dW, dV, db, dc,de2 = bptt(xs, hs, ys, U, W, V, b, c)
    
   
    # print_colored("\nGradients after BPTT:", "cyan")
    # print_colored(f"dU: \n{dU}", "green")
    print_colored(f"dW: \n{dW}", "blue")
    # print_colored(f"dV: \n{dV}", "magenta")
    # print_colored(f"db: \n{db}", "red")
    # print_colored(f"dc: \n{dc}", "yellow")

    # Update parameters
    learning_rate = 0.1
    U -= learning_rate * dU
    W -= learning_rate * dW
    V -= learning_rate * dV
    b -= learning_rate * db
    c -= learning_rate * dc
    E -= learning_rate * de2

    # print_colored("\nWeights after BPTT:", "cyan")
    # print_colored(f"U: \n{U}", "green")
    # print_colored(f"W: \n{W}", "blue")
    # print_colored(f"V: \n{V}", "magenta")
    # print_colored(f"b: \n{b}", "red")
    # print_colored(f"c: \n{c}", "yellow")

   
    for dparam in [dU, dW, dV, db, dc]:
        np.clip(dparam, -5, 5, out=dparam)
    
    

# The code provided assumes a terminal or console that supports ANSI escape codes for colored output.
# If you're using an environment that doesn't support ANSI codes, the colors may not appear as intended.
