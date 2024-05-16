# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:04:27 2024

@author: HSK
"""

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
    print(f"{colors.get(color, colors['reset'])}{text}{colors['reset']}")

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
U = np.random.randn(hidden_size, vocab_size) * 0.5  # Input to hidden
W = np.random.randn(hidden_size, hidden_size) * 0.4  # Hidden to hidden
V = np.random.randn(vocab_size, hidden_size) * 0.3  # Hidden to output
b = np.zeros((hidden_size, 1))  # Hidden bias
c = np.zeros((vocab_size, 1))  # Output bias

print_colored("\nParameters Initialized:", "magenta")

# Input ("hell") and target ("ello") sequences
x_sequence = 'hell'
y_sequence = 'ello'
print_colored("\nInput and Target Sequences:", "green")
for i in range(len(x_sequence)):
    print_colored(f"x[{i}] = '{x_sequence[i]}', y[{i}] = '{y_sequence[i]}'", "green")

# Forward Pass function definition
def forward_pass(x_sequence, U, W, V, b, c):
    xs, hs, os, ys = {}, {}, {}, {}
    hs[-1] = np.zeros((hidden_size, 1))  # Initial hidden state
    loss = 0

    for t in range(len(x_sequence)):
        xs[t] = char_to_onehot[x_sequence[t]].reshape(-1, 1)  # One-hot
        hs[t] = np.tanh(np.dot(U, xs[t]) + np.dot(W, hs[t-1]) + b)  # Hidden state
        os[t] = np.dot(V, hs[t]) + c  # Raw output
        ys[t] = np.exp(os[t]) / np.sum(np.exp(os[t]))  # Softmax output
        correct_char_vec = char_to_onehot[y_sequence[t]].reshape(-1, 1)
        loss += -np.sum(correct_char_vec * np.log(ys[t]))  # Cross-entropy loss
        # print_colored(f"\nStep {t+1}:", "cyan")
        # print_colored(f"x_{t+1} = One-hot encoding for '{x_sequence[t]}'", "yellow")
        # print_colored(f"\nHidden State at t={t+1}:", "cyan")
        # print_colored(f"h_{t+1} = tanh(U * x_{t+1} + W * h_{t} + b)", "cyan")
        # print_colored(f"= tanh({U} * {xs[t]} + {W} * {hs[t-1]} + {b})","red")
        # print_colored(f"h_{t+1} = {hs[t]}", "cyan")

        # print_colored(f"o_{t+1} = V * h_{t+1} + c", "blue")
        # print_colored(f"= {V} * {hs[t]} + {c}", "blue")
        # print_colored(f"o_{t+1} = {os[t]}", "blue")
        # print_colored(f"y_hat{t+1} = softmax(y_hat{t+1})", "magenta")
        # print_colored(f"y_hat{t+1} = {ys[t]}", "magenta")

        # print_colored(f"Loss{t+1} = -y * ln(y_hat{t+1})", "red")
        # print_colored(f"Loss{t+1} = -{char_to_onehot[y_sequence[t]].reshape(-1, 1)} * ln({ys[t]})", "red")
        # print_colored(f"Loss at step {t+1}: {-np.sum(correct_char_vec * np.log(ys[t]))}", "red")
    #print("Total Loss:",loss)    
    return loss, xs, hs, ys

def bptt(xs, hs, ys, U, W, V, b, c):
    S=int(len(xs[0]))
    dU2, dW2, dV2 = np.zeros_like(U), np.zeros_like(W), np.zeros_like(V)
    db2, dc2 = np.zeros_like(b), np.zeros_like(c)
    
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
            product_of_gradients*= hs[k-1]*(1 - hs[k]**2) #if k-1 > -1 else  np.zeros_like(W) 
            dW2+= np.dot(product_of_gradients,beforeProduct)
    print("dW2:",dW2)   
    
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

    return dU2, dW2, dV2, db2, dc2



for epoch in range(1):
    print_colored(("="*60)+"#"+str(epoch)+" EPOCH "+("="*60)+"\n\n", "cyan")
    print_colored("U =\n{}".format(U), "green")
    print_colored("W =\n{}".format(W), "blue")
    print_colored("b =\n{}".format(b), "red")

    print_colored("V =\n{}".format(V), "magenta")
    print_colored("c =\n{}".format(c), "yellow")

    # print_colored(("="*60)+"FORWARD PASS"+("="*60)+"\n\n", "cyan")
    loss, xs, hs, ys = forward_pass(x_sequence, U, W, V, b, c)
    # print_colored(("="*60)+"BACKPROP PASS"+("="*60)+"\n\n", "cyan")
    dU, dW, dV, db, dc = bptt(xs, hs, ys, U, W, V, b, c)

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
