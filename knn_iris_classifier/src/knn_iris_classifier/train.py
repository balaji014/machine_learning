"""
Train a KNN model on the Iris dataset using:
- Pipeline (StandardScaler + KNeighborsClassifier)
- StratifiedKFold cross-validation
- GridSearchCV hyperparameter tuning
Saves the best model to models/knn_iris_model.joblib
"""

from pathlib import Path
import joblib

from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix


def main() -> None:
    # Load Iris dataset as pandas DataFrame
    iris = load_iris(as_frame=True)
    X = iris.data                     # Features: 4 numeric columns
    y = iris.target                   # Labels: 0/1/2
    target_names = iris.target_names  # Class names

    df = iris.frame
    print("Dataset shape:", df.shape)

    # Build pipeline: scaling is important for distance-based KNN
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ])

    # StratifiedKFold keeps class distribution equal across folds
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Hyperparameter grid (expanded for better chances of best accuracy)
    param_grid = {
        "knn__n_neighbors": list(range(1, 31)),          # try k = 1..30
        "knn__weights": ["uniform", "distance"],         # voting strategy
        "knn__metric": ["minkowski"],                    # minkowski + p controls distance
        "knn__p": [1, 2],                                # p=1 -> manhattan, p=2 -> euclidean
        "knn__leaf_size": [10, 20, 30, 40, 50],          # tree leaf size (performance tuning)
    }

    # GridSearchCV will try all combinations using StratifiedKFold CV
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
        verbose=0
    )

    # Train on the FULL dataset using CV to select best hyperparameters
    grid.fit(X, y)

    # Best model after tuning
    best_model = grid.best_estimator_
    best_score = grid.best_score_

    print("Best CV Accuracy:", best_score)
    print("Best Parameters:", grid.best_params_)

    # Optional: Evaluate the tuned model on the same data (not a true test)
    # This is mainly for seeing a confusion matrix + report quickly.
    preds = best_model.predict(X)

    print("\nConfusion Matrix (on full dataset):\n", confusion_matrix(y, preds))
    print("\nClassification Report (on full dataset):\n")
    print(classification_report(y, preds, target_names=target_names))

    # Save model artifacts
    Path("models").mkdir(exist_ok=True)
    joblib.dump(best_model, "models/knn_iris_model.joblib")
    joblib.dump(target_names, "models/target_names.joblib")

    print("\n✅ Saved model to models/knn_iris_model.joblib")


if __name__ == "__main__":
    main()
