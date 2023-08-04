from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

def main():
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Get input features from the user
    sepal_length = float(input("Enter sepal length (cm): "))
    sepal_width = float(input("Enter sepal width (cm): "))
    petal_length = float(input("Enter petal length (cm): "))
    petal_width = float(input("Enter petal width (cm): "))

    # Combine features into a single array for prediction
    new_flower = [[sepal_length, sepal_width, petal_length, petal_width]]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    new_flower = scaler.transform(new_flower)

    # Create a Decision Tree classifier and fit it to the training data
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Predict the species of the new flower
    prediction = clf.predict(new_flower)

    # Get the species name based on the prediction
    species_names = iris.target_names
    predicted_species = species_names[prediction[0]]

    print(f"The predicted species of the new flower is: {predicted_species}")

if __name__ == "__main__":
    main()
