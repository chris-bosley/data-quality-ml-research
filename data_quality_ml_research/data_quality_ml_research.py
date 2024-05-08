"""Module for training a classifier and evaluating performance under data mutations.

This script performs the following steps:

1. Loads configuration variables.
2. Defines and trains a model.
3. Splits data into training and some number of equally sized training sets.
4. Mutates the scoring data.
5. Evaluates model performance on the mutated data.
6. Saves results to a CSV file.

**Usage:**

# Example: Run on online shoppers data, evaluate change to zero mutations, write output file.
project > python data_quality_ml_research
"""

import pandas as pd
import os
from sklearn.linear_model import LogisticRegressionCV

import helpers as dq


def main():
    """Ingests data, trains a model and evaluates it's performance against mutated data.

    Args:
        None.

    Returns:
        None.
    """

    # Declare configuration variables
    config = {
        "model": "Logistic Regression",
        "data_location": "../data/online_shoppers_intention.csv",
        "output_path": "../output_data",  # Path to store output files
        "output_filename": "output.csv",  # Filename for the results
        "preprocess_data": False,
        "training_percent": 0.5,
        "mutator": "change_to_zero",
        "num_scoring_groups": 5,
        "seeds": {"outer_seed": 123, "inner_seed": 456},
    }

    # Handle potential output file conflicts
    output_path = config["output_path"]
    output_filename = config["output_filename"]
    output_file = os.path.join(output_path, output_filename)
    if os.path.exists(output_file):
        print(f"Warning: Output file already exists: {output_file}")
        print(f"Deleting file...")
        os.remove(output_file)

    # Declare required classifier
    # Look at default parameters OR quick grid search
    clf = LogisticRegressionCV(Cs=[0.1, 1, 10], fit_intercept=True)

    # Ingest dataset passed from command line
    data = dq.read_data(config["data_location"])

    # We'll also have to handle data more carefully than this... this is a bit lazy
    data.drop(["Month", "VisitorType", "Weekend"], axis=1, inplace=True)

    # identify features and targets. We'll have to specify or assume last or something here
    feature_cols = data.columns[:-1]
    target_col = data.columns[-1]

    # Split data
    # Build Scoring Groups
    # Determine the size each scoring group by evenly splitting non-training data
    scoring_group_size = (1 - config["training_percent"]) / config["num_scoring_groups"]
    scoring_proportions = [config["training_percent"]]
    # Build splits
    for i in range(config["num_scoring_groups"]):
        scoring_proportions.append(scoring_group_size)

    # Split data into training and specified number of non-training datasets
    data_splits = dq.random_nonintersecting_subsets(
        data, proportions=scoring_proportions, seed=config["seeds"]["inner_seed"]
    )

    # Fit declared model
    clf.fit(data_splits[0][feature_cols], data_splits[0][target_col])

    # Discard training data for ease of handling mutations
    data_splits.pop(0)

    # print(confusion_matrix(test[target_col], clf_list[0][0].predict(test[feature_cols])))

    # Mutate all datasets
    # Apply a specified mutation and examine model performance under various severities of data failure.
    # BUG REVIEW Check that mutator is mutating correct proportions...
    mutation_results = dq.collect_mutation_results(
        data_splits, config, target_col, feature_cols, clf
    )

    pd.DataFrame(mutation_results).to_csv(config["result_output_location"])


if __name__ == "__main__":
    main()
