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

import tools as data_tools
from model_creation import create_model
from itertools import chain


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

        config_file = "./sample_config.json"
        print("WARNING: sample configuration used")


    config = data_tools.load_config(config_file)
    """
    EXAMPLE CONFIGURATION
    config = {
        "model": "Logistic Regression",
        "data_location": "../data/online_shoppers_intention.csv",
        "preprocess_data": 0, <-----I think this should be drop?
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

    # Ingest dataset passed from command line
    data = data_tools.read_data(config["data_location"])

    # Specific handling ONLY FOR ONLINE SHOPPERS
    categorical_cols = ["Month", "VisitorType", "TrafficType", "Weekend", "Region"]
    drop_cols = ["OperatingSystems", "Browser", "index"]
    target_col = "Revenue"
    numeric_cols = [
        i for i in data.columns if i not in categorical_cols + drop_cols + [target_col]
    ]
    feature_cols = categorical_cols + numeric_cols
    mutation_results = []
    # Loop over mutations first then models
    for mutation_name in config["mutator"]:
        for model_name in config["model"]:
            # Declare required classifiers
            clf = create_model(
                model_type=model_name,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                preprocess=True,
                model_state=123,
            )

            # Split data
            # Build Scoring Groups
            # Determine the size each scoring group by evenly splitting non-training data
            scoring_group_size = (1 - config["training_percent"]) / config["num_scoring_groups"]
            # This is a little unclear -- what I'm doing is making the scoring proportions array early then padding.
            scoring_proportions = [config["training_percent"]]
            # Build splits
            for i in range(config["num_scoring_groups"]):
                scoring_proportions.append(scoring_group_size)

            # Split data into training and specified number of non-training datasets
            data_splits = data_tools.random_nonintersecting_subsets(
                data, proportions=scoring_proportions, seed=config["seeds"]["inner_seed"]
            )

            training_data = data_splits[0]
            data_splits.pop(0)

            # Fit declared model
            clf.fit(training_data[feature_cols], training_data[target_col])

            # Discard training data for ease of handling mutations

            # print(confusion_matrix(test[target_col], clf_list[0][0].predict(test[feature_cols])))

            # Mutate all datasets for the given model
            # Apply a specified mutation and examine model performance under various severities of data failure.
            model_mutation_results = data_tools.collect_mutation_results(
                data_splits,
                config,
                target_col,
                feature_cols,
                model_name,
                mutation_name,
                clf,
                num_experiments=config["num_mutation_checks"],
            )
            # Append the mutation results from the model and end the for loop
            mutation_results.append(model_mutation_results)

    mutation_results = list(chain.from_iterable(mutation_results))
    # Write results and config to disk
    pd.DataFrame(mutation_results).to_csv(output_dir / output_file, index=False)
    with open(output_dir / output_config, "w") as f:
        json.dump(config, f)


if __name__ == "__main__":
    main()
