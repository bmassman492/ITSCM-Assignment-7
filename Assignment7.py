import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data from the Excel spreadsheet
file_path = 'C:/Users/bmass/Downloads/baseball.xlsx'
data = pd.read_excel(file_path)

# Define the independent variables
X = data[['Runs Scored', 'Runs Allowed', 'Wins', 'OBP', 'SLG', 'Team Batting Average']]

# Define the dependent variable
y = data['Playoffs']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a logistic regression model
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict the playoffs for the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print(f'Accuracy of the model: {accuracy:.2f}')

# Create values for the independent variables to test
test_values = [[800, 600, 90, 0.350, 0.500, 0.280],
               [750, 650, 85, 0.330, 0.470, 0.270],
               [700, 700, 80, 0.320, 0.450, 0.260]]

# Predict the playoffs for the test values
predictions = model.predict_proba(test_values)

# Print the likelihood of making the playoffs for each test value
for i, pred in enumerate(predictions):
    print(f'For team {i+1}, the likelihood of making the playoffs is: {pred[1]*100:.2f}%')

# Print data on the strength of the correlation
coefficients = model.coef_[0]
print('\nStrength of the correlation:')
for i, col in enumerate(X.columns):
    print(f'{col}: {coefficients[i]:.2f}')
