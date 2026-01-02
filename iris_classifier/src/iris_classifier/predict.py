import joblib
import pandas as pd


def main():
    model = joblib.load("models/iris_model.joblib")
    target_names = joblib.load("models/target_names.joblib")

    sample = {
        "sepal length (cm)": 3.1,
        "sepal width (cm)": 6.8,
        "petal length (cm)": 4.7,
        "petal width (cm)": 4.2
    }

    X = pd.DataFrame([sample])
    pred = model.predict(X)[0]
    print("Predicted species:", target_names[pred])


if __name__ == "__main__":
    main()
