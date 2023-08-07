import numpy as np
from sklearn.linear_model import LogisticRegression

# Sample data for demonstration purposes (you can replace this with your own dataset)
# The first column represents the usage minutes, and the second column represents the contract duration.
# The third column represents the label '1' for churned customers and '0' for non-churned customers.
data = np.array([
    [200, 12, 1],
    [150, 6, 0],
    [300, 24, 1],
    [100, 3, 0],
    [400, 36, 1]
])

# Splitting the data into features (X) and target (y)
X = data[:, :-1]  # All columns except the last one (features: usage minutes, contract duration)
y = data[:, -1]   # Last column (target: churn status)

# Creating the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Function to get user inputs for new customer features
def get_new_customer_features():
    usage_minutes = float(input("Enter the usage minutes of the new customer: "))
    contract_duration = int(input("Enter the contract duration of the new customer: "))
    return np.array([[usage_minutes, contract_duration]])

# Predicting whether the new customer will churn or not based on user input
def predict_churn_status(model, features):
    return model.predict(features)

if __name__ == "__main__":
    # Getting new customer features from the user
    new_customer_features = get_new_customer_features()

    # Predicting whether the new customer will churn or not
    predicted_churn_status = predict_churn_status(model, new_customer_features)

    if predicted_churn_status[0] == 1:
        print("The new customer is predicted to churn.")
    else:
        print("The new customer is predicted not to churn.")
