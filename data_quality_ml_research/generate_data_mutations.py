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

from pathlib import Path

import tools as data_tools
import numpy as np
import sys


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
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "C:/Users/chris/PycharmProjects/data-quality-ml-research/data_quality_ml_research/sample_file_mutation_config.json"
        print("WARNING: sample configuration used")

    config = data_tools.load_config(config_file)

    # Ensure output path exists
    output_dir = Path(f"{config['output_path']}").absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define mutation based on config
    mutation_name, column_name = define_mutation(config)
    num_mutation_steps = config["num_mutated_sets"]

    # Read data using data_tools module (adapt based on data format)
    try:
        if "shoppers" in config["data_location"]:
            data = data_tools.read_data(config["data_location"])
            dataset_name = "shoppers"

        elif "bank-additional" in config["data_location"]:
            data = data_tools.read_data(config["data_location"], separator=";")
            data = data.replace("unknown", np.NaN)
            dataset_name = "banking"
        else:
            raise ValueError("Unsupported data format based on file location")
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    # Split data into training and mutated sets
    mutate_group_size = (1 - config["training_percent"]) / config["num_mutated_sets"]
    scoring_proportions = [config["training_percent"]] + [mutate_group_size] * config[
        "num_mutated_sets"
    ]

    data_splits = data_tools.random_nonintersecting_subsets(
        data, proportions=scoring_proportions, seed=config["seeds"]["inner_seed"]
    )

    training_data = data_splits.pop(0)
    training_data.to_csv(
        f"{config['output_path']}/{dataset_name}_mutated_0_{column_name}.csv"
    )

    # Apply mutations to individual datasets with varying severity
    mutation_severities = [
        round((j + 1) * 1 / num_mutation_steps, 2) for j in range(num_mutation_steps)
    ]
    for n, severity in enumerate(mutation_severities):
        try:
            mutated_data = data_tools.get_mutations(
                [data_splits[n]], severity, mutation_name, config
            )
            mutated_data[0].to_csv(
                f"{config['output_path']}/{dataset_name}_mutated_{int(round(severity * 100, 0))}_{column_name}.csv"
            )
        except Exception as e:
            print(f"Error applying mutation to dataset {n}: {e}")


if __name__ == "__main__":
    main()
