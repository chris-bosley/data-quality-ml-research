# This filename needs changed
import json

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)

from mutation import Mutator


def read_data(filepath, separator=",", inplace=False):
    """
    Reads a CSV file from a relative path within the project and returns a pandas DataFrame.

    Args:
        filepath (str): The relative path to the CSV file within the project.
        separator (str): String to use as field delimiter for file ingest.
        inplace (bool, optional): If True, modifies the existing DataFrame object passed to the function
                                 instead of creating a new one. This can be useful for memory efficiency
                                 when working with very large CSV files, but it can also lead to unexpected
                                 behavior if the DataFrame is used elsewhere in your code. Defaults to False.
                                 (creates a new DataFrame).

    Returns:
        pandas.DataFrame: The DataFrame containing the data from the CSV file.
                          (if inplace=False) or None (if inplace=True).
    """

    # Read the CSV file using the relative path
    if inplace:
        pd.read_csv(filepath, sep=separator, inplace=True)
    else:
        df = pd.read_csv(filepath, sep=separator)
        return df


def random_nonintersecting_subsets(df, proportions, seed=None):
    """
    Generates non-intersecting random subsets from a pandas DataFrame based on specified proportions.
    Sets are shuffled before extraction to ensure randomness.

    Args:
        df (pandas.DataFrame): The DataFrame to split into subsets.
        proportions (list): A list of floats representing the proportion of rows for each subset.
        seed (int, optional): An optional seed for the random number generator. Defaults to None.

    Returns:
        list: A list of non-intersecting DataFrames representing the random subsets.

    Raises:
        ValueError: If the total size of the subsets exceeds the number of rows in the DataFrame.
    """

    np.random.seed(seed)

    # Calculate the total number of rows in the DataFrame
    total_rows = len(df)

    # Calculate the number of rows for each subset based on proportions
    subset_sizes = [int(prop * total_rows) for prop in proportions]

    # Create an empty list to store the non-intersecting random subsets
    these_subsets = []

    # Ensure the total number of rows is at least as large as the total required for all subsets
    if sum(subset_sizes) <= total_rows:
        # Create a copy of the DataFrame to work with
        df_copy = df.copy()

        # Shuffle the rows of the copy DataFrame
        df_copy = df_copy.sample(frac=1).reset_index(drop=True)

        # Extract non-intersecting random subsets based on the subset sizes
        start_idx = 0
        for size in subset_sizes:
            if size > 0:
                subset = df_copy.iloc[start_idx : start_idx + size]
                these_subsets.append(
                    subset.reset_index()
                )  # Reset index for compatibility
                start_idx += size
    else:
        raise ValueError(
            "Total subset size exceeds the total number of rows in the DataFrame."
        )

    return these_subsets


def get_random_subsets_from_file(
    filepath, proportions, seed=None, separator=",", inplace=False
):
    """
    Reads a CSV file, generates non-intersecting random subsets based on proportions, and returns them.

    Args:
        filepath (str): The relative path to the CSV file within the project.
        proportions (list): A list of floats representing the proportion of rows for each subset.
        seed (int, optional): An optional seed for the random number generator. Defaults to None.
        separator (str): String to use as field delimiter for file ingest.
        inplace (bool, optional): If True, modifies the existing DataFrame object passed to the function
                                 instead of creating a new one. This can be useful for memory efficiency
                                 when working with very large CSV files, but it can also lead to unexpected
                                 behavior if the DataFrame is used elsewhere in your code. Defaults to False.
                                 (creates a new DataFrame).

    Returns:
        list: A list of non-intersecting DataFrames representing the random subsets.

    Raises:
          ValueError: If the total size of the subsets exceeds the number of rows in the DataFrame.
    """

    # Read the DataFrame from the file
    df = read_data(filepath, separator=separator, inplace=inplace)

    # Generate random subsets based on proportions
    subsets = random_nonintersecting_subsets(df, proportions, seed)

    return subsets


def get_mutation_metrics(datasets, target_col, feature_cols, clf):
    """
    Calculates a set of classification metrics for each dataset in a list.

    This function ingests a list of DataFrames, the target column name,
    the feature column names, and a trained classifier, and calculates a set of classification
    metrics (precision, recall, accuracy, F1-score, balanced accuracy) for each dataset in the list.

    Args:
        datasets (list): A list of pandas DataFrames.
        target_col (str): The name of the column containing the target variable in the DataFrames.
        feature_cols (list): A list of strings representing the names of the feature columns in the DataFrames.
        clf (sklearn.classifier): A trained sklearn classifier object.

    Returns:
        dict: A dictionary where keys are classification metric names (strings) and values are lists containing
              the corresponding metric scores for each dataset in the `datasets` list.
    """
    # Declare metrics to be calculated
    # Each metric is defined with three required elements - name, function & arguments.
    classification_metrics = [
        {
            "name": "precision",
            "function": precision_score,
            "args": {"zero_division": np.nan},
        },
        {"name": "recall", "function": recall_score, "args": {"zero_division": np.nan}},
        {"name": "accuracy", "function": accuracy_score, "args": {}},
        {"name": "f1", "function": f1_score, "args": {"zero_division": np.nan}},
        {"name": "balanced-accuracy", "function": balanced_accuracy_score, "args": {}},
    ]

    # Create results object
    mutation_results = {}
    # Iterate through required metrics
    # Ideally I'd like to set zero_division=np.nan for precision and f1, but handle that later
    # As is it'll return 0, which is less than ideal. Probably use a dict of lambdas.
    for metFunc in classification_metrics:
        this_metric = []
        for i in datasets:
            # Calculate metric for each dataset
            this_metric.append(
                metFunc["function"](
                    i[target_col], clf.predict(i[feature_cols]), **metFunc["args"]
                )
            )
        mutation_results[metFunc["name"]] = this_metric
    return mutation_results


def get_mutations(data, severity, config):
    """
    Applies a data mutation to each DataFrame in a list with a specified severity level.

    This function takes a list of DataFrames (`data`), a mutation severity level (`severity`),
    and a configuration dictionary (`config`) containing random number generator seeds and other collateral.
    It iterates through each DataFrame in `data`, applies the mutation  with the specified severity to the
    named columns, and appends the mutated DataFrame to a new list.

    Args:
        data (list): A list of pandas DataFrames.
        severity (float): A value between 0 and 1 representing the proportion of elements in the
                          specified columns to be mutated.
        config (dict): A dictionary containing configuration parameters, including seeds for the random
                       number generator. The dictionary should have keys "seeds" and "outer_seed" and
                       "inner_seed" nested within "seeds". This is going to be needed for other stuff too.

    Returns:
        list: A list of pandas DataFrames containing the mutated versions of the input DataFrames.
    """
    # Declare mutator
    mutator = Mutator()
    mutated_data = []
    for i in data:
        # induce mutation on each dataset
        mutated = mutator.change_to_zero(
            i,
            # NEED TO CONFIG COLUMN IN FUNCTION LOL
            columns=["PageValues"],
            percentage=severity,
            seed=config["seeds"]["outer_seed"],
            seed_modifier=config["seeds"]["inner_seed"],
        )
        mutated_data.append(mutated)
    return mutated_data


def collect_mutation_results(
    data_to_mutate, config, target_col, feature_cols, clf, num_experiments=100
):
    """
    Evaluates a classifier's performance under a series of data mutations.

    This function iterates through a range of data mutation severities (from 0 to 1 in steps of 0.1 should be config).
    For each severity level, it performs the following steps:
        1. Applies the mutation (defined in `Mutator`) to the input data with the specified severity.
        2. Calculates a set of classification metrics (precision, recall, accuracy, F1-score) on the mutated data.
        3. Creates a dictionary containing information about the mutation, severity level,
           and the calculated metrics.
        4. Appends this dictionary to a list that stores results for all severities.

    Args:
        data_to_mutate (list): A list of pandas DataFrames.
        config (dict): A dictionary containing configuration parameters.
        target_col (str): The name of the column containing the target variable in the DataFrames.
        feature_cols (list): A list of strings representing the names of the feature columns in the DataFrames.
        clf (sklearn.classifier): A trained sklearn classifier object.
        num_experiments (int): The number of experiments to run. This will generate evenly spaced
            mutation severities in [0,1]

    Returns:
        list: A list of dictionaries containing information about each mutation run, including mutation type,
              severity, and a dictionary of classification metrics for the mutated data.
    """
    all_runs = []
    for mutation_severity in [j * 1 / num_experiments for j in range(num_experiments)]:
        # Round here because np float stuff
        mutation_severity = round(mutation_severity, 2)
        # Mutate the data
        mutated_data = get_mutations(data_to_mutate, mutation_severity, config)
        # Calculate results at current severity
        run_results = {
            "mutation": "change_to_zero",
            "severity": mutation_severity,
            "metrics": get_mutation_metrics(
                mutated_data, target_col, feature_cols, clf
            ),
        }
        all_runs.append(run_results)
    return all_runs


def load_config(configuration_file):
    """Loads configuration data from a JSON file.

    Args:
        configuration_file (str): The path to the configuration JSON file.

    Returns:
        dict: The configuration dictionary parsed from the JSON file.

    Raises:
        FileNotFoundError: If the specified JSON file cannot be found.
        json.JSONDecodeError: If there's an error parsing the JSON data.

    """
    try:
        with open(configuration_file, "r") as config_file:
            config = json.loads(config_file.read())
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File Not Found")
    except json.JSONDecodeError as e:
        # raise json.JSONDecodeError(f"JSON Parsing Error: {e}", e.doc, e.pos)
        print(f"JSON Parsing Error: {e} at {e.pos} in {e.doc}")
