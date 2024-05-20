from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectPercentile, chi2

"""
This module provides a function to create machine learning models based on the specified type.

It currently supports two models:

- Logistic Regression
- Poly SVC
- Linear SVC
- RBF SVC
- Random Forest
- Neural Network

All models use a random state for reproducibility.

**Error Handling:**

- Raises a ValueError if an invalid model_type is provided.

**Example Usage:**

```python
from model_creation import create_model

try:
    model = create_model("Logistic Regression")
    print(f"Created model: {model}")
except ValueError as e:
    print(f"Error creating model: {e}")
"""


# Probably should make categorical_cols an optional parameter. Doesn't matter rn, maybe not worth.


def create_model(
    model_type,
    numeric_cols,
    categorical_cols,
    preprocess=True,
    model_state=123,
):
    """Loads configuration data from a JSON file.

    Args:
        model_type (str): Model type to train.
        numeric_cols (list): Column names of numeric input data.
        categorical_cols (list): Column names of categorical input data.
        preprocess (bool): Declare if data preprocessing should be applied. Defaults to True.
        model_state (int): Random seed for model training.

    Returns:
        sklearn.pipeline.Pipeline: The configuration dictionary parsed from the JSON file.

    Raises:
        ValueError: If the specified model string is not valid.

    """

    model_options = {
        "Logistic Regression": {
            "model": LogisticRegressionCV,
            "args": {"Cs": 10, "fit_intercept": True, "random_state": model_state},
        },
        "Poly SVC": {
            "model": SVC,
            "args": {
                "kernel": "poly",
                "class_weight": "balanced",
                "probability": False,
                "random_state": model_state,
            },
        },
        "Linear SVC": {
            "model": SVC,
            "args": {
                "kernel": "linear",
                "class_weight": "balanced",
                "probability": False,
                "random_state": model_state,
            },
        },
        "RBF SVC": {
            "model": SVC,
            "args": {
                "kernel": "rbf",
                "class_weight": "balanced",
                "probability": False,
                "random_state": model_state,
            },
        },
        "Random Forest": {
            "model": RandomForestClassifier,
            "args": {
                "n_estimators": 100,
                "criterion": "gini",
                "class_weight": "balanced",
                "random_state": model_state,
            },
        },
        "Neural Network": {
            "model": MLPClassifier,
            "args": {
                "hidden_layer_sizes": [27, 7, 10],  # Umm.... we can do better here haha
                "activation": "relu",
                "random_state": model_state,
            },
        },
    }

    if model_type not in model_options:
        raise ValueError(f"Invalid model_type: {model_type}")

    # If preprocess is True let's make a pipeline, else let's just declare the model

    model = model_options[model_type]["model"](**model_options[model_type]["args"])
    if preprocess:
        numeric_transformer = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                # Might want a different scaler here
                ("scale", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("encode", OneHotEncoder(handle_unknown="ignore")),
                ("select", SelectPercentile(chi2, percentile=50)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
    else:
        clf = Pipeline(steps=[("classifier", model)])

    return clf
