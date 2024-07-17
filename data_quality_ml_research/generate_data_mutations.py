import tools as data_tools
import numpy as np

"""
Module for generating mutated datasets

This script performs the following functions:
1. Picks up a csv
2. Splits the csv into the specified number of datasets with specified sizes - one for "pure" and n for "mutated"
3. Mutates the indicated column using the configured method in each of the "mutated" datasets
4. Writes all files back as csv to the specified directory in the format <filename>_<mutation>_<column>.csv

** Usage **

> Do the thing

"""


def main():
    # Example config for development
    # Need to force-create full path for output
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

    # Define mutation
    mutation_name = list(config["mutator"].keys())[0]
    column_name = list(config["mutator"].values())[0][0]

    # pick up file
    if "shoppers" in config["data_location"]:
        data = data_tools.read_data(config["data_location"])

    elif "bank-additional" in config["data_location"]:
        data = data_tools.read_data(config["data_location"], separator=";")
        data = data.replace("unknown", np.NaN)

    # Split data
    # Determine the size each mutated group by evenly splitting non-training data
    mutate_group_size = (1 - config["training_percent"]) / config["num_mutated_sets"]
    # Build splits
    scoring_proportions = [config["training_percent"]]
    for i in range(config["num_mutated_sets"]):
        scoring_proportions.append(mutate_group_size)

    # Split data into pure set and specified number of non-training datasets
    data_splits = data_tools.random_nonintersecting_subsets(
        data,
        proportions=scoring_proportions,
        seed=config["seeds"]["inner_seed"],
    )

    training_data = data_splits[0]
    training_data.to_csv(f"{config["output_path"]}mutated_0_{column_name}.csv")

    data_splits.pop(0)

    # Figure out how much to mutate each dataset
    mutation_severities = [round((j + 1) * 1 / 15, 2) for j in range(15)]
    for n, severity in enumerate(mutation_severities):
        mutated_data = data_tools.get_mutations(
            [data_splits[n]], severity, mutation_name, config
        )
        mutated_data[0].to_csv(f"{config["output_path"]}mutated_{int(round(severity*100,0))}_{column_name}.csv")


if __name__ == "__main__":
    main()
