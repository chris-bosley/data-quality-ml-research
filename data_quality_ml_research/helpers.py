# This name needs changed
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data_quality_ml_research import mutation


# noinspection PyArgumentList
def read_csv_to_dataframe(filepath, inplace=False):
    """
    Reads a CSV file from a relative path within the project and returns a pandas DataFrame.

    Args:
        filepath (str): The relative path to the CSV file within the project.
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
        pd.read_csv(filepath, inplace=True)
    else:
        df = pd.read_csv(filepath)
        return df


def random_nonintersecting_subsets(df, proportions, seed=None):
    """
    Generates non-intersecting random subsets from a pandas DataFrame based on specified proportions.

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
            "Total subset size exceeds the number of rows in the DataFrame."
        )

    return these_subsets


def get_random_subsets_from_csv(filepath, proportions, seed=None):
    """
    Reads a CSV file, generates non-intersecting random subsets based on proportions, and returns them.

    Args:
        filepath (str): The relative path to the CSV file within the project.
        proportions (list): A list of floats representing the proportion of rows for each subset.
        seed (int, optional): An optional seed for the random number generator. Defaults to None.

    Returns:
        list: A list of non-intersecting DataFrames representing the random subsets.

    Raises:
          ValueError: If the total size of the subsets exceeds the number of rows in the DataFrame.
    """

    # Read the DataFrame from the CSV file (using inplace=False)
    df = read_csv_to_dataframe(filepath, inplace=False)

    # Generate random subsets based on proportions
    subsets = random_nonintersecting_subsets(df, proportions, seed)

    return subsets


def get_mutation_metrics(datasets, target_col, feature_cols, clf):
    classification_metrics = {
        "precision": precision_score,
        "recall": recall_score,
        "accuracy": accuracy_score,
        "f1": f1_score,
    }

    mutation_results = {}
    for metricName, metFunc in classification_metrics.items():
        this_metric = []
        for i in datasets:
            this_metric.append(metFunc(i[target_col], clf.predict(i[feature_cols])))
        mutation_results[metricName] = this_metric
    return mutation_results


def get_mutations(data, severity, config):
    mutator = mutation.Mutator()
    mutated_data = []
    for i in data:
        mutated = mutator.change_to_zero(
            i,
            columns=["PageValues"],
            percentage=severity,
            seed=config["seeds"]["outer_seed"],
            seed_modifier=config["seeds"]["inner_seed"],
        )
        mutated_data.append(mutated)
    return mutated_data
