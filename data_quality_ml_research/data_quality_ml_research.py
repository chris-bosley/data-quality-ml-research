"""Module for training a classifier and evaluating performance under data mutations.

This script performs the following steps:

1. Loads configuration variables.
2. Defines and trains a model.
3. Splits data into training and some number of equally sized training sets.
4. Mutates the scoring data.
5. Evaluates model performance on the mutated data.
6. Saves results and configuration to a CSV file.

**Usage:**

# Example: Run on online shoppers data, evaluate change to zero mutations, write output file.
project > python data_quality_ml_research ./sample_config.json
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegressionCV

import helpers as dq


def main():
    """Ingests data, trains a model and evaluates it's performance against mutated data.

    Args:
        None.

    Returns:
        None.
    """
    # Parse configuration file from command line
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        # Could just provide dummy config too
        print("Error: Must Provide Run Configuration")
        sys.exit(1)

    config = dq.load_config(config_file)
    """
    EXAMPLE CONFIGURATION
    config = {
        "model": "Logistic Regression",
        "data_location": "../data/online_shoppers_intention.csv",
        "preprocess_data": 0,
        "training_percent": 0.5,
        "mutator": "change_to_zero",
        "num_scoring_groups": 5,
        "num_mutation_checks": 100,
        "seeds": {"outer_seed": 123, "inner_seed": 456},
    }
    """
    # Determine output location
    run_timestamp = str(int(time.time()))
    # Grab name of data file to be used
    data_file_name = config["data_location"].split("/")[-1].split(".")[0]
    output_dir = Path("../output_data/" + "_".join([data_file_name, run_timestamp]))
    output_dir.mkdir(parents=True, exist_ok=True)
    # Output files
    output_file = "results.csv"
    output_config = "config.json"

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
        data_splits,
        config,
        target_col,
        feature_cols,
        clf,
        num_experiments=config["num_mutation_checks"],
    )

    # Write results and config to disk
    pd.DataFrame(mutation_results).to_csv(output_dir / output_file)
    with open(output_dir / output_config, "w") as f:
        json.dump(config, f)


if __name__ == "__main__":
    main()
