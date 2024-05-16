# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:56:21 2024

@author: HSK
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('log_RNN.csv')
pivot_data = df.pivot_table(values='Training Time', index='Number of Epochs', columns='Batch Size', aggfunc=np.mean)

total_width = 0.8
number_of_bars = len(pivot_data.columns)
single_bar_width = total_width / number_of_bars
bar_positions = np.arange(len(pivot_data))


fig, ax = plt.subplots(figsize=(12, 8))
for i, batch_size in enumerate(pivot_data.columns):
    ax.bar(bar_positions + i * single_bar_width, pivot_data[batch_size], width=single_bar_width, label=f'Batch Size {batch_size}')


ax.set_xlabel('Number of Epochs')
ax.set_ylabel('Training Time')
ax.set_title('Training Time by Number of Epochs and Batch Sizes')
ax.set_xticks(bar_positions + total_width / 2 - single_bar_width / 2)
ax.set_xticklabels(pivot_data.index)
ax.legend(title='Batch Size')


plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('clustered_bar_chart 1.png')
plt.show()



pivot_data = df.pivot_table(values='Training Time', index='Number of Epochs', columns='Number of Hidden Layers', aggfunc=np.mean)

total_width = 0.8
number_of_bars = len(pivot_data.columns)
single_bar_width = total_width / number_of_bars
bar_positions = np.arange(len(pivot_data))


fig, ax = plt.subplots(figsize=(12, 8))
for i, batch_size in enumerate(pivot_data.columns):
    ax.bar(bar_positions + i * single_bar_width, pivot_data[batch_size], width=single_bar_width, label=f'Number of Hidden Layers {batch_size}')


ax.set_xlabel('Number of Epochs')
ax.set_ylabel('Training Time')
ax.set_title('Training Time by Number of Epochs and Number of Hidden Layers')
ax.set_xticks(bar_positions + total_width / 2 - single_bar_width / 2)
ax.set_xticklabels(pivot_data.index)
ax.legend(title='Number of Hidden Layers')


plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('clustered_bar_chart2.png')
plt.show()



pivot_data = df.pivot_table(values='Training Time', index='Number of Epochs', columns='Number of Samples', aggfunc=np.mean)

total_width = 0.8
number_of_bars = len(pivot_data.columns)
single_bar_width = total_width / number_of_bars
bar_positions = np.arange(len(pivot_data))


fig, ax = plt.subplots(figsize=(12, 8))
for i, batch_size in enumerate(pivot_data.columns):
    ax.bar(bar_positions + i * single_bar_width, pivot_data[batch_size], width=single_bar_width, label=f'Number of Samples {batch_size}')


ax.set_xlabel('Number of Epochs')
ax.set_ylabel('Training Time')
ax.set_title('Training Time by Number of Epochs and Number of Samples')
ax.set_xticks(bar_positions + total_width / 2 - single_bar_width / 2)
ax.set_xticklabels(pivot_data.index)
ax.legend(title='Number of Samples')


plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('clustered_bar_chart2.png')
plt.show()
