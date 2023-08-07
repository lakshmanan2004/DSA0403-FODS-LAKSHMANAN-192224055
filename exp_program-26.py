import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data for demonstration purposes (you can replace this with your own dataset)
# The first column represents the area of the house, and the second column represents the number of bedrooms.
# The third column represents the price of the house.
data = np.array([
    [1000, 2, 200000],
    [1500, 3, 300000],
    [1200, 2, 250000],
    [1800, 3, 350000],
    [2000, 4, 400000]
])

# Splitting the data into features (X) and target (y)
X = data[:, :-1]  # All columns except the last one (features: area, number of bedrooms)
y = data[:, -1]   # Last column (target: price)

# Creating the linear regression model
model = LinearRegression()
model.fit(X, y)

# Function to get user inputs for new house features
def get_new_house_features():
    area = float(input("Enter the area of the new house: "))
    num_bedrooms = int(input("Enter the number of bedrooms in the new house: "))
    return np.array([[area, num_bedrooms]])

# Predicting the price of the new house based on user input
def predict_price(model, features):
    return model.predict(features)

if __name__ == "__main__":
    # Getting new house features from the user
    new_house_features = get_new_house_features()

    # Predicting the price of the new house
    predicted_price = predict_price(model, new_house_features)

    print(f"Predicted price for the new house: ${predicted_price[0]:.2f}")
