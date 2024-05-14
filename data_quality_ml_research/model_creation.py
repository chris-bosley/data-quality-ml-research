from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


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


def create_model(model_type="Logistic Regression", model_state=123):

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

    clf = model_options[model_type]["model"](**model_options[model_type]["args"])

    return clf
