# KNN Iris Classifier

A compact scikit-learn project that trains a k-nearest neighbors (KNN) classifier on the classic Iris dataset. The training script tunes hyperparameters with cross-validation, evaluates the fitted model, and saves reusable artifacts for prediction.

## Project layout
- src/knn_iris_classifier/train.py: build pipeline (StandardScaler + KNN), run StratifiedKFold + GridSearchCV, report metrics, and save artifacts.
- src/knn_iris_classifier/predict.py: load saved model/labels and run a sample prediction; edit the `sample` dict to test your own inputs.
- models/: serialized `knn_iris_model.joblib` and `target_names.joblib` created by the training step.

## Quickstart
1) Use Python 3.11+. Create and activate a virtual environment:
   - Windows PowerShell: `py -3.11 -m venv .venv; .\\.venv\\Scripts\\Activate.ps1`
2) Install dependencies: `pip install scikit-learn pandas joblib`
3) Train the model (downloads Iris from scikit-learn automatically):  
   `python src/knn_iris_classifier/train.py`
4) Run a prediction using the saved model:  
   `python src/knn_iris_classifier/predict.py`

## What training does
- Loads Iris as a pandas DataFrame, separating features and labels.
- Builds a pipeline with StandardScaler (feature scaling) and KNeighborsClassifier.
- Tunes hyperparameters via GridSearchCV over 10-fold StratifiedKFold splits:
  - n_neighbors: 1..30
  - weights: uniform, distance
  - metric: minkowski with p in {1, 2}
  - leaf_size: 10, 20, 30, 40, 50
- Prints best accuracy/params, confusion matrix, and classification report.
- Saves best estimator and class names to `models/`.

## Notes
- Ensure `models/knn_iris_model.joblib` exists (from training) before running predictions.
- When crafting custom predictions, keep feature names identical to the Iris dataset columns: `sepal length (cm)`, `sepal width (cm)`, `petal length (cm)`, `petal width (cm)`.
