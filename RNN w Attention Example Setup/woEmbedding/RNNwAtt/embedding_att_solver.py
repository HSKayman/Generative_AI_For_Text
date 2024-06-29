# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 21:27:24 2024

@author: HSK
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 01:16:52 2024

@author: HSK
"""

import numpy as np
np.random.seed(124)
# Define the function for colored printing
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
print_colored(f"Vocabulary Size: {vocab_size}", "green")

# One-hot encoding setup
char_to_onehot = {ch: np.eye(vocab_size)[i] for i, ch in enumerate(vocab)}
onehot_to_char = {i: ch for i, ch in enumerate(vocab)}

print_colored("\nCharacter to One-hot Encoding:", "yellow")
for char, encoding in char_to_onehot.items():
    print_colored(f"{char}: {encoding}", "yellow")

# Initialize parameters
hidden_size = 1  # Size of the hidden layer
U = np.random.randn(hidden_size, 2) * 0.5  # Input to hidden
W = np.random.randn(hidden_size, hidden_size) * 0.4  # Hidden to hidden
V = np.random.randn(vocab_size, hidden_size) * 0.3  # Hidden to output
b = np.zeros((hidden_size, 1))  # Hidden bias
c = np.zeros((vocab_size, 1))  # Output bias
E = np.random.randn(vocab_size, 2) * 0.5
print_colored("\nParameters Initialized:", "magenta")
print(E)
# Input ("hell") and target ("ello") sequences
x_sequence = 'hell'
y_sequence = 'ello'
print_colored("\nInput and Target Sequences:", "green")
for i in range(len(x_sequence)):
    print_colored(f"x[{i}] = '{x_sequence[i]}', y[{i}] = '{y_sequence[i]}'", "green")

# Forward Pass function definition
def forward_pass_with_attention(x_sequence, y_sequence, U, W, V, b, c,E):
    xs, hs, os, ys, attentions = {}, {}, {}, {}, {}
    hs[-1] = np.zeros((hidden_size, 1))  # Initial hidden state
    loss = 0

    # Compute hidden states for the entire sequence
    for t in range(len(x_sequence)):
        xs[t] = np.dot(char_to_onehot[x_sequence[t]],E)  # One-hot
        hs[t] = np.tanh(np.dot(U, xs[t]) + np.dot(W, hs[t-1]) + b)  # Hidden state
        print_colored(f"\nHidden State at t={t+1}:", "cyan")
        print_colored(f"h_{t+1} = tanh(U * x_{t+1} + W * h_{t} + b)", "cyan")
        print_colored(f"= tanh({U} * {xs[t]} + {W} * {hs[t-1]} + {b})","red")
        print_colored(f"h_{t+1} = {hs[t]}", "cyan")

    # Calculate attention weights and context vectors after computing all hidden states
    for t in range(len(x_sequence)):
        # Correct approach to compute attention scores: comparing hs[t] with each previous hs
        attention_scores = {tau: np.dot(hs[t].T, hs[tau]) for tau in range(t + 1)} # 2x1 * 1x2 = 2x2
        attention_weights = np.exp(list(attention_scores.values())) / np.sum(np.exp(list(attention_scores.values()))) # Softmax
        attentions[t] = attention_weights
        
        print_colored(f"Output Stage at t={t+1}:", "cyan")
        for tau in range(t + 1):
            print_colored(f"A_({t+1},{tau+1}) , tau+1= h_{t+1}* h_{tau+1}", "magenta")
            print_colored(f"A_({t+1},{tau+1})= {hs[t].T}*{hs[tau]}", "magenta")
        
        print_colored(f"softmax(A_({t+1})= attentions[t]", "magenta")
        print_colored(f"Z_{t+1})= \sum_(k=0)^{t+1} softmax(A_({t+1},(k)) \cdot h_k$$", "magenta")
        for tau in range(t + 1):
            print_colored(f"Z_{t+1} += {attentions[t][tau]}*{hs[tau]}", "magenta")
        # Compute context vector as weighted sum of hidden states
        context_vector = np.sum([attentions[t][tau] * hs[tau] for tau in range(t + 1)], axis=0)
        print_colored(f"Z_{t+1}= {context_vector}", "magenta")
        
        # Combine context vector with current hidden state to produce output
        os[t] = np.dot(V, context_vector) + c
        ys[t] = np.exp(os[t]) / np.sum(np.exp(os[t]))  # Softmax output
        print_colored(f"Omega_{t+1} = V * Z_{t+1} + c", "blue")  # Updated to reflect use of context_vector
        print_colored(f"Omega__{t+1} = {V} * {context_vector} + {c}", "blue")
        print_colored(f"Omega__{t+1} = {os[t]}", "blue")
        print_colored(f"hat(y)_{t+1} = softmax(Omega_{t})", "magenta")
        print_colored(f"hat(y)_{t+1} = {ys[t]}", "magenta")
        correct_char_vec = char_to_onehot[y_sequence[t]].reshape(-1, 1)
        print(ys[t])
        print(-np.sum(correct_char_vec * np.log(ys[t])))
        loss += -np.sum(correct_char_vec * np.log(ys[t]))  # Cross-entropy loss
        print_colored(f"Loss{t+1} = -y * ln(y_hat{t+1})", "red")
        print_colored(f"Loss{t+1}: {-np.sum(correct_char_vec * np.log(ys[t]))}", "red")
    print("Total Loss:",loss) 
    return loss, xs, hs, ys, attentions


def bptt(xs, hs, ys, attentions, correct_char_vecs, U, W, V, b, c):
    """
    Backward pass including attention mechanism.
    """
    dU, dW, dV = np.zeros_like(U), np.zeros_like(W), np.zeros_like(V)
    db, dc = np.zeros_like(b), np.zeros_like(c)
    S=int(len(xs))
    de2 = np.zeros_like(E)
    for t in reversed(range(len(x_sequence))):
        dy = np.copy(ys[t])
        correct_char_vec = char_to_onehot[y_sequence[t]].reshape(-1, 1)
        dy -= correct_char_vec  # Gradient of softmax (y - t)
        dV += np.dot(dy, hs[t].T)
        dc += dy
        
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
        for m in range(0,t+1):
            for k in range(0, m+1):
                dy = np.copy(ys[t])
                correct_char_vec = char_to_onehot[y_sequence[t]].reshape(-1, 1)
                dy -= correct_char_vec  # Gradient of softmax (y - t) wrt omega
                beforeProduct=np.dot(V.T, dy)  # Gradient of omega wrt h
                beforeProduct*= attentions[t][m]
                product_of_gradients =  1
                for j in range(k,m):
                    product_of_gradients *= (1 - hs[j+1] ** 2) * W
                product_of_gradients*= hs[k-1]*(1 - hs[k]**2) 
                dW+= np.dot(product_of_gradients,beforeProduct)
    
    for t in range(0, S):
        for m in range(0,t+1):
            for k in range(0, m+1):
                dy = np.copy(ys[t])
                correct_char_vec = char_to_onehot[y_sequence[t]].reshape(-1, 1)
                dy -= correct_char_vec  # Gradient of softmax (y - t) wrt omega
                beforeProduct=np.dot(V.T, dy)  # Gradient of omega wrt h
                beforeProduct*= attentions[t][m]
                product_of_gradients =  np.ones_like(dU)
                for j in range(k,m):
                    product_of_gradients *= W.T * (1 - hs[j+1] ** 2) 
                product_of_gradients*= xs[k].T *(1 - hs[k]**2) 
                dU+= product_of_gradients * beforeProduct
    
    for t in range(0, S):
        for m in range(0,t+1):
            for k in range(0, m+1):
                dy = np.copy(ys[t])
                correct_char_vec = char_to_onehot[y_sequence[t]].reshape(-1, 1)
                dy -= correct_char_vec  # Gradient of softmax (y - t) wrt omega
                beforeProduct=np.dot(V.T, dy)  # Gradient of omega wrt h
                beforeProduct*= attentions[t][m]
                product_of_gradients =  1
                for j in range(k,m):
                    product_of_gradients *= (1 - hs[j+1] ** 2) * W
                product_of_gradients*= (1 - hs[k]**2) 
                db+= np.dot(product_of_gradients,beforeProduct)
    
    return dU, dW, dV, db, dc, de2


a=[]
for epoch in range(501):
    print_colored(("="*60)+"#"+str(epoch)+" EPOCH "+("="*60)+"\n\n", "cyan")
    print_colored("U =\n{}".format(U), "green")
    print_colored("W =\n{}".format(W), "blue")
    print_colored("b =\n{}".format(b), "red")

    print_colored("V =\n{}".format(V), "magenta")
    print_colored("c =\n{}".format(c), "yellow")
    
    print_colored(("="*60)+"FORWARD PASS"+("="*60)+"\n\n", "cyan")
    loss, xs, hs, ys,attention = forward_pass_with_attention(x_sequence,y_sequence, U, W, V, b, c,E)
    print_colored(f"\nAttention Weights:{attention}", "magenta")
    print_colored(("="*60)+"BACKPROP PASS"+("="*60)+"\n\n", "cyan")

    correct_char_vecs = [char_to_onehot[y_sequence[t]].reshape(-1, 1) for t in range(len(y_sequence))]
    dU, dW, dV, db, dc,de2 = bptt(xs, hs, ys, attention, correct_char_vecs, U, W, V, b, c)
    print(loss)
    print_colored("\nGradients after BPTT:", "cyan")
    print_colored(f"dU: \n{dU}", "green")
    print_colored(f"dW: \n{dW}", "blue")
    print_colored(f"dV: \n{dV}", "magenta")
    print_colored(f"db: \n{db}", "red")
    print_colored(f"dc: \n{dc}", "yellow")
    a.append(loss)
    # Update parameters
    learning_rate = 0.1
    U -= learning_rate * dU
    W -= learning_rate * dW
    V -= learning_rate * dV
    b -= learning_rate * db
    c -= learning_rate * dc
    E -= learning_rate * de2
    
    print_colored("\nWeights after BPTT:", "cyan")
    print_colored(f"U: \n{U}", "green")
    print_colored(f"W: \n{W}", "blue")
    print_colored(f"V: \n{V}", "magenta")
    print_colored(f"b: \n{b}", "red")
    print_colored(f"c: \n{c}", "yellow")

   
    for dparam in [dU, dW, dV, db, dc]:
        np.clip(dparam, -5, 5, out=dparam)
    
    

