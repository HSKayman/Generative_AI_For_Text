import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from Excel
df = pd.read_csv('log_RNN.csv')

# Overview of the data
print(df.head())

sns.set_style("whitegrid")

# 1. Pairplot to visualize the relationships between all numerical variables
pairplot_fig = sns.pairplot(df, kind='scatter')
plt.suptitle('Pairplot of All Variables', verticalalignment='top')
# Save the pairplot
pairplot_fig.savefig('RNN_pairplot_all_variables.png')
plt.show()

# 2. Heatmap of the correlation matrix to see how each variable correlates with Duration
plt.figure(figsize=(8, 6))
correlation_matrix = df.corr()
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=2, linecolor='black')
plt.title('Correlation Matrix')
# Save the heatmap
plt.savefig('RNN_correlation_matrix.png')
plt.show()

# 3. Individual scatter plots for each parameter vs Duration
parameters = ['Number of elements', 'Batch size', 'Unit size', 'Number of epochs']
for param in parameters:
    plot = sns.lmplot(x=param, y='Duration', data=df, aspect=1.5)
    plt.title(f'Relationship between {param} and Duration')
    # Save each scatter plot
    plot.savefig(f'RNN_scatter_{param}_vs_duration.png')
    plt.show()

# Load data from Excel
df = pd.read_csv('log_RNNwAtt.csv')

# Overview of the data
print(df.head())

sns.set_style("whitegrid")

# 1. Pairplot to visualize the relationships between all numerical variables
pairplot_fig = sns.pairplot(df, kind='scatter')
plt.suptitle('Pairplot of All Variables', verticalalignment='top')
# Save the pairplot
pairplot_fig.savefig('RNNwAtt_pairplot_all_variables.png')
plt.show()

# 2. Heatmap of the correlation matrix to see how each variable correlates with Duration
plt.figure(figsize=(8, 6))
correlation_matrix = df.corr()
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=2, linecolor='black')
plt.title('Correlation Matrix')
# Save the heatmap
plt.savefig('RNNwAtt_correlation_matrix.png')
plt.show()

# 3. Individual scatter plots for each parameter vs Duration
parameters = ['Number of elements', 'Batch size', 'Unit size', 'Number of epochs']
for param in parameters:
    plot = sns.lmplot(x=param, y='Duration', data=df, aspect=1.5)
    plt.title(f'Relationship between {param} and Duration')
    # Save each scatter plot
    plot.savefig(f'RNNwAtt_scatter_{param}_vs_duration.png')
    plt.show()