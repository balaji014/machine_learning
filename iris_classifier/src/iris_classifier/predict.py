import joblib # joblib is used to load the trained machine learning model from disk
import pandas as pd # pandas is used to create a DataFrame for model input


def main():
    model = joblib.load("models/iris_model.joblib") # Load the trained Iris classification model from disk
    target_names = joblib.load("models/target_names.joblib")  # Load the target class names (setosa, versicolor, virginica)

    # Sample input data for prediction
    # Keys must exactly match the feature names used during training
    sample = {
        "sepal length (cm)": 3.1,
        "sepal width (cm)": 6.8,
        "petal length (cm)": 4.7,
        "petal width (cm)": 4.2
    }

    # Convert the sample dictionary into a pandas DataFrame
    # Model expects a 2D structure even for a single record
    X = pd.DataFrame([sample])

    # Predict the class index for the input sample
    # [0] is used because predict() returns an array
    pred = model.predict(X)[0]

    # Print the predicted species name using the class index
    print("Predicted species:", target_names[pred])

# Entry point of the script
# Ensures main() runs only when this file is executed directly
if __name__ == "__main__":
    main()
