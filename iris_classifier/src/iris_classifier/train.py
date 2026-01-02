from pathlib import Path # Import Path to handle file system paths in a cross-platform way
import joblib # joblib is used to save and load trained ML models efficiently

from sklearn.datasets import load_iris # Load the built-in Iris dataset
from sklearn.model_selection import train_test_split # Utility to split dataset into training and testing sets
from sklearn.pipeline import Pipeline # Pipeline allows chaining preprocessing + model into a single object
from sklearn.preprocessing import StandardScaler # StandardScaler normalizes feature values (mean=0, std=1)
from sklearn.linear_model import LogisticRegression # Logistic Regression model for classification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # Metrics used to evaluate model performance


def main():
    iris = load_iris(as_frame=True) # Load Iris dataset as pandas DataFrame
    X = iris.data # Feature matrix (sepal length, sepal width, petal length, petal width)
    y = iris.target # Target labels (0=setosa, 1=versicolor, 2=virginica)
    target_names = iris.target_names # Human-readable class names

    # Split data into training (80%) and testing (20%)
    # stratify=y ensures class balance in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Create a machine learning pipeline:
    model = Pipeline([
        ("scaler", StandardScaler()), # 1. Scale input features
        ("clf", LogisticRegression(max_iter=200)) # 2. Train Logistic Regression classifier
    ])

    model.fit(X_train, y_train) # Train the model using training data

    preds = model.predict(X_test)  # Generate predictions on test data

    print("Accuracy:", accuracy_score(y_test, preds)) # Print overall accuracy metric
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds)) # Print confusion matrix to analyze prediction errors
    print("\nReport:\n", classification_report(y_test, preds, target_names=target_names)) # Print detailed precision, recall, and F1-score

    Path("models").mkdir(exist_ok=True) # Create 'models' directory if it doesn't exist
    joblib.dump(model, "models/iris_model.joblib") # Save trained model to disk
    joblib.dump(target_names, "models/target_names.joblib") # Save target names for later use
    print("\n✅ Saved model to models/iris_model.joblib") # Confirmation message

# Entry point of the script
# Ensures main() runs only when file is executed directly
if __name__ == "__main__":
    main()
