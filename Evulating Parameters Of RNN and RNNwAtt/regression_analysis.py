import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_csv('log_RNN.csv')

# Define the independent variables and add a constant term for the intercept
X = df[['Number of elements', 'Batch size', 'Unit size', 'Number of epochs']]
X = sm.add_constant(X)

# Define the dependent variable
y = df['Duration']

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Save the summary of the model to a text file
with open('RNN_regression_analysis_summary.txt', 'w') as f:
    f.write(model.summary().as_text())

# Optionally, print the summary to the console
print(model.summary())


# Load data
df = pd.read_csv('log_RNNwAtt.csv')

# Define the independent variables and add a constant term for the intercept
X = df[['Number of elements', 'Batch size', 'Unit size', 'Number of epochs']]
X = sm.add_constant(X)

# Define the dependent variable
y = df['Duration']

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Save the summary of the model to a text file
with open('RNNwAtt_regression_analysis_summary.txt', 'w') as f:
    f.write(model.summary().as_text())

# Optionally, print the summary to the console
print(model.summary())