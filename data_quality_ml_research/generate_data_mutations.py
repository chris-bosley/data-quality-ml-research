"""
Module for generating mutated datasets

This module provides functionalities to split a dataset into a "pure" set
and multiple "mutated" sets. Each mutated set has the specified column
altered using a configured method.

**Features:**

* Reads CSV data using the `data_tools` module.
* Splits data into training and mutated sets with user-defined proportions.
* Applies mutations to specific columns with adjustable severity.
* Saves all generated datasets as CSV files.

**Usage:**

This module is intended to be run as a script. Simply execute the script,
and it will process the data based on the provided configuration.
"""

import tools as data_tools
import numpy as np


def define_mutation(config):
    """
    Extracts the mutation method and target column from the configuration.

    Args:
        config: A dictionary containing configuration parameters.

    Returns:
        A tuple containing the mutation method name (string) and the target column name (string).
    """
    mutation_name = list(config["mutator"].keys())[0]
    column_name = list(config["mutator"].values())[0][0]
    return mutation_name, column_name


def main():
    """
    Main function for processing data and generating mutated datasets.

    This function reads the configuration, defines the mutation, splits the data,
    applies mutations to individual datasets, and saves the results.
    """

    # Example config for development (modify as needed)
    config = {
        "data_location": "../data/bank-additional-full.csv",
        "training_percent": 0.35,
        "num_mutated_sets": 15,
        "output_path": "../mutated_data/",
        "mutator": {"change_to_zero": ["euribor3m"]},
        "seeds": {"outer_seed": 123, "inner_seed": 456},
        "noise_arguments": {
            "distribution": "normal",
            "low": 0,
            "high": 100,
            "scale": 2,
        },
    }

    # Define mutation based on config
    mutation_name, column_name = define_mutation(config)

    # Read data using data_tools module (adapt based on data format)
    if "shoppers" in config["data_location"]:
        data = data_tools.read_data(config["data_location"])

    elif "bank-additional" in config["data_location"]:
        data = data_tools.read_data(config["data_location"], separator=";")
        data = data.replace("unknown", np.NaN)

    # Split data into training and mutated sets
    mutate_group_size = (1 - config["training_percent"]) / config["num_mutated_sets"]
    scoring_proportions = [config["training_percent"]] + [mutate_group_size] * config[
        "num_mutated_sets"
    ]

    data_splits = data_tools.random_nonintersecting_subsets(
        data, proportions=scoring_proportions, seed=config["seeds"]["inner_seed"]
    )

    training_data = data_splits.pop(0)
    training_data.to_csv(f"{config['output_path']}mutated_0_{column_name}.csv")

    # Apply mutations to individual datasets with varying severity
    mutation_severities = [round((j + 1) * 1 / 15, 2) for j in range(15)]
    for n, severity in enumerate(mutation_severities):
        mutated_data = data_tools.get_mutations(
            [data_splits[n]], severity, mutation_name, config
        )
        mutated_data[0].to_csv(
            f"{config['output_path']}mutated_{int(round(severity * 100, 0))}_{column_name}.csv"
        )


if __name__ == "__main__":
    main()
