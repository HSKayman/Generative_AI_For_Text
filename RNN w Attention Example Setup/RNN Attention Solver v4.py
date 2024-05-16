# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 19:28:57 2024

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
    print(f"{colors.get(color, colors['reset'])}{text}{colors['reset']}")

print_colored("="*60 + " INITIALIZATION " + "="*60, "cyan")

# Vocabulary setup
vocab = ['h', 'e', 'l', 'o']
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
def forward_pass_with_attention(x_sequence, y_sequence, U, W, V, b, c):
    xs, hs, os, ys, attentions = {}, {}, {}, {}, {}
    hs[-1] = np.zeros((hidden_size, 1))  # Initial hidden state
    loss = 0

    # Compute hidden states for the entire sequence
    for t in range(len(x_sequence)):
        xs[t] = char_to_onehot[x_sequence[t]].reshape(-1, 1)  # One-hot
        hs[t] = np.tanh(np.dot(U, xs[t]) + np.dot(W, hs[t-1]) + b)  # Hidden state
        print_colored(f"\nHidden State at t={t}:", "cyan")
        print_colored(f"h_{t} = tanh(U * x_{t} + W * h_{t-1} + b)", "cyan")
        print_colored(f"= tanh({U} * {xs[t]} + {W} * {hs[t-1]} + {b})","red")
        print_colored(f"h_{t} = {hs[t]}", "cyan")

    # Calculate attention weights and context vectors after computing all hidden states
    for t in range(len(x_sequence)):
        # Correct approach to compute attention scores: comparing hs[t] with each previous hs
        attention_scores = {tau: np.dot(hs[t].T, hs[tau]) for tau in range(t + 1)} # 2x1 * 1x2 = 2x2
        attention_weights = np.exp(list(attention_scores.values())) / np.sum(np.exp(list(attention_scores.values()))) # Softmax
        attentions[t] = attention_weights

        # Compute context vector as weighted sum of hidden states
        context_vector = np.sum([attentions[t][tau] * hs[tau] for tau in range(t + 1)], axis=0)
        print_colored(f"Output Stage at t={t}:", "cyan")
        print_colored(f"Context Vector= sum(h_{t} * softmax( h_{t}^T*h_tau (tau<t) ))","magenta")
        for tau in range(t + 1):
            print_colored(f"= {hs[t].T}*{hs[tau]}", "magenta")
        print_colored(f"softmax()={attention_weights}", "magenta")
        for tau in range(t + 1):
            print_colored(f"= {attentions[t][tau]}*{hs[tau]}", "magenta")
        print_colored(f"= {context_vector}", "magenta")
        
        # Combine context vector with current hidden state to produce output
        os[t] = np.dot(V, context_vector) + c
        ys[t] = np.exp(os[t]) / np.sum(np.exp(os[t]))  # Softmax output
        
        correct_char_vec = char_to_onehot[y_sequence[t]].reshape(-1, 1)
        loss += -np.sum(correct_char_vec * np.log(ys[t]))  # Cross-entropy loss
        print_colored(f"o_{t} = V * context_vector + c", "blue")  # Updated to reflect use of context_vector
        print_colored(f"o_{t} = {V} * {context_vector} + {c}", "blue")
        print_colored(f"o_{t} = {os[t]}", "blue")
        print_colored(f"y_{t} = softmax(o_{t})", "magenta")
        print_colored(f"y_{t} = {ys[t]}", "magenta")

        print_colored("Loss = -t * log(y)", "red")
        print_colored(f"Loss at step {t}: {loss}", "red")

    print_colored(f"\nTotal Loss: {loss}", "red")
    return loss, xs, hs, ys, attentions


def bptt(xs, hs, ys, attentions, correct_char_vecs, U, W, V, b, c):
    """
    Backward pass including attention mechanism.
    """
    dU, dW, dV = np.zeros_like(U), np.zeros_like(W), np.zeros_like(V)
    db, dc = np.zeros_like(b), np.zeros_like(c)
    dhnext = np.zeros_like(hs[0])
    
    for t in reversed(range(len(xs))):
        dy = np.copy(ys[t])
        dy -= correct_char_vecs[t]  # Gradient of softmax (y - t)
        dV += np.dot(dy, hs[t].T)
        dc += dy
        
        dh = np.dot(V.T, dy) + dhnext  # Backprop into h through output layer
        dhraw = (1 - hs[t] ** 2) * dh  # Backprop through tanh nonlinearity
        db += dhraw
        dU += np.dot(dhraw, xs[t].T)
        dW += np.dot(dhraw, hs[t-1].T if t > 0 else np.zeros_like(hs[t-1].T))
        dhnext = np.dot(W.T, dhraw)
        
        # Print formulas with colored output
        print_colored(f"\nBackprop Step {t}:", "cyan")
        print_colored("Gradient of loss w.r.t. softmax output (dy):", "yellow")
        print_colored(f"dy = y_hat - y ", "yellow")
        print_colored(f"dy = {ys[t]} - {correct_char_vecs[t]}", "yellow")
        print_colored(f"dy = {dy}", "yellow")
        
        print_colored("Update dV (Gradient of loss w.r.t. V):", "green")
        print_colored(f"dV += dy * hs_t.T ", "green")
        print_colored(f"dV += {dy} * {hs[t].T}", "green")
        print_colored(f"dV += {dV}", "green")
        
        print_colored("Update dc (Gradient of loss w.r.t. c):", "blue")
        print_colored(f"dc += dy", "blue")
        print_colored(f"dc += {dy}", "blue")
        print_colored(f"dc += {dc}", "blue")
        
        print_colored("Backprop into h (dh):", "magenta")
        print_colored(f"dh = V.T * dy + dh_{t-1} ", "magenta")
        print_colored(f"dh = {V.T} * {dy} + {dhnext} \n dh={dh}", "magenta")
        
        print_colored("Backprop through tanh nonlinearity (dhraw):", "red")
        print_colored(f"dhraw = (1 - hs_t^2) * dh", "red")
        print_colored(f"dhraw = (1 - {hs[t]}^2) * {dh}", "red")
        print_colored(f"dhraw = {dhraw}", "red")

        
        print_colored("Update dU (Gradient of loss w.r.t. U):", "green")
        print_colored(f"dU += dhraw * x_t.T", "green")
        print_colored(f"dU += {dhraw} * {xs[t].T}", "green")
        print_colored(f"dU += {dU}", "green")

        print_colored("Update dW (Gradient of loss w.r.t. W):", "blue")
        print_colored(f"dW += dhraw * h_(t-1).T \n= {dhraw} * {hs[t-1].T} \n= {dW}", "blue")


        # Backprop through attention mechanism
        dcontext_vector = np.dot(V.T, dy)
        print_colored(f" dcontext_vector = V.T * dy =\n = {V.T} * {dy} \n ={dcontext_vector} ", "magenta")
        dattention_weights = np.zeros_like(attentions[t])
        for tau in range(t + 1):
            dattention_weights[tau] = np.dot(dcontext_vector.T, hs[tau])
            print_colored(f" dattention_weights_{tau} = dcontext_vector^T * h_{tau} =\n = {dcontext_vector.T} * {hs[tau]} \n ={dattention_weights[tau]} ", "magenta")
            dh_tau = dcontext_vector * attentions[t][tau]
            print_colored(f" dh_{tau} = dcontext_vector * attention_weights_{tau} =\n = {dcontext_vector} * {attentions[t][tau]} \n ={dh_tau} ", "magenta")
            dhraw_tau = (1 - hs[tau] ** 2) * dh_tau
            print_colored(f" dhraw_{tau} = (1 - hs_{tau}^2) * dh_{tau} =\n = (1 - {hs[tau]}^2) * {dh_tau} \n ={dhraw_tau} ", "magenta")
            db += dhraw_tau
            print_colored(f" db += dhraw_{tau} =\n = {db} ", "magenta")
            dU += np.dot(dhraw_tau, xs[tau].T)
            print_colored(f" dU += dhraw_{tau} * x_{tau}.T =\n = {dU} ", "magenta")
            if tau > 0:
                dW += np.dot(dhraw_tau, hs[tau-1].T)
                print_colored(f" dW += dhraw_{tau} * h_{tau-1}.T =\n = {dW} ", "magenta")
            dhnext += np.dot(W.T, dhraw_tau) if tau < t else 0
            print_colored(f" dhnext += W.T * dhraw_{tau} =\n ={W.T} * dh_(t+1)\n = {dhnext} ", "magenta")

        
    return dU, dW, dV, db, dc



for epoch in range(1):
    print_colored(("="*60)+"#"+str(epoch)+" EPOCH "+("="*60)+"\n\n", "cyan")
    print_colored("U =\n{}".format(U), "green")
    print_colored("W =\n{}".format(W), "blue")
    print_colored("b =\n{}".format(b), "red")

    print_colored("V =\n{}".format(V), "magenta")
    print_colored("c =\n{}".format(c), "yellow")

    print_colored(("="*60)+"FORWARD PASS"+("="*60)+"\n\n", "cyan")
    loss, xs, hs, ys,attention = forward_pass_with_attention(x_sequence,y_sequence, U, W, V, b, c)
    print_colored(f"\nAttention Weights:{attention}", "magenta")
    print_colored(("="*60)+"BACKPROP PASS"+("="*60)+"\n\n", "cyan")

    correct_char_vecs = [char_to_onehot[y_sequence[t]].reshape(-1, 1) for t in range(len(y_sequence))]
    dU, dW, dV, db, dc = bptt(xs, hs, ys, attention, correct_char_vecs, U, W, V, b, c)

    print_colored("\nGradients after BPTT:", "cyan")
    print_colored(f"dU: \n{dU}", "green")
    print_colored(f"dW: \n{dW}", "blue")
    print_colored(f"dV: \n{dV}", "magenta")
    print_colored(f"db: \n{db}", "red")
    print_colored(f"dc: \n{dc}", "yellow")

    # Update parameters
    learning_rate = 0.1
    U -= learning_rate * dU
    W -= learning_rate * dW
    V -= learning_rate * dV
    b -= learning_rate * db
    c -= learning_rate * dc

    print_colored("\nWeights after BPTT:", "cyan")
    print_colored(f"U: \n{U}", "green")
    print_colored(f"W: \n{W}", "blue")
    print_colored(f"V: \n{V}", "magenta")
    print_colored(f"b: \n{b}", "red")
    print_colored(f"c: \n{c}", "yellow")

   
    for dparam in [dU, dW, dV, db, dc]:
        np.clip(dparam, -5, 5, out=dparam)
    
    
