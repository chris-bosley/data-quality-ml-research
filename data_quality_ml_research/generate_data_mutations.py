import tools as data_tools

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

def main()
    #Example config for development
    config = {'file': 'foo.csv',
              'mutator': 'change_to_zero',
              'column': 'bar',
              'pure_size': 35,
              'num_mutated_sets': 10,
              'output_path': 'baz/'}

    #pick up file
    data = data_tools.read_data(config["file"])

    # Split data
    # Determine the size each mutated group by evenly splitting non-training data
    mutate_group_size = (1 - config["pure_size"]) / config["num_mutated_sets"]
    # Build splits
    scoring_proportions = [config["training_percent"]]
    for i in range(config["num_mutated_sets"]):
        scoring_proportions.append(mutate_group_size)

    # Split data into pure set and specified number of non-training datasets
    data_splits = data_tools.random_nonintersecting_subsets(
        data,
        proportions=scoring_proportions,
        seed=789,
    )

    training_data = data_splits[0]
    data_splits.pop(0)
