# Perhaps refactor to split modeling tools from general tools
import ast
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


def get_mutation_metrics(datasets, target_col, feature_cols, clf, cost_weights=None):
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
        cost_weights (dict): A dict containing the values used for weighting the penalty matrix.
            Must include the keys "tp", "fp", "tn", "fn". Weights must be numeric.

    Returns:
        dict: A dictionary where keys are classification metric names (strings) and values are lists containing
              the corresponding metric scores for each dataset in the `datasets` list.
    """
    # Declare metrics to be calculated
    # Each metric is defined with three required elements - name, function & arguments.

    # Force a default penalty weighting if none is provided
    if cost_weights is None:
        cost_weights = {"tp": 10, "tn": 0, "fp": -3, "fn": -8}

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
        {
            "name": "positive-retrieval",
            "function": retrieval_percent,
            "args": {"signal_type": "pos"},
        },
        {
            "name": "negative-retrieval",
            "function": retrieval_percent,
            "args": {"signal_type": "neg"},
        },
        {
            "name": "cost-penalty",
            "function": cost_penalty,
            "args": {"weight_dict": cost_weights},
        },
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
    valid_mutators = [
        "change_to_zero",
        "transpose_columns",
        "add_categorical_class",
        "remove_class",
        "add_noise",
    ]
    # Declare mutator
    mutator = Mutator()

    if config["mutator"] not in valid_mutators:
        raise ValueError(
            f"Error: {config['mutator']} invalid. Must be one of {valid_mutators}"
        )
    mutated_data = []
    for i in data:
        if config["mutator"] == "change_to_zero":
            mutator_method = mutator.change_to_zero
            mutator_args = {
                "df": i,
                "columns": config["mutation_target"],
                "percentage": severity,
                "seed": config["seeds"]["outer_seed"],
                "seed_modifier": config["seeds"]["inner_seed"],
            }
        elif config["mutator"] == "transpose_columns":
            mutator_method = mutator.transpose_columns
            mutator_args = {
                "df": i,
                "col1": config["mutation_target"][0],
                "col2": config["mutation_target"][1],
            }
        elif config["mutator"] == "add_categorical_class":
            mutator_method = mutator.add_categorical_class
            mutator_args = {
                "df": i,
                "column_name": config["mutation_target"],
                "percentage": severity,
                "seed": config["seeds"]["outer_seed"],
            }
        elif config["mutator"] == "remove_class":
            mutator_method = mutator.remove_class
            mutator_args = {
                "df": i,
                "column_name": config["mutation_target"],
                "class_seed": config["seeds"]["outer_seed"],
                "mutator_seed": config["seeds"]["inner_seed"],
            }
        elif config["mutator"] == "add_noise":
            mutator_method = mutator.add_noise
            mutator_args = {"df": i, "column_name": config["mutation_target"]}
            mutator_args.update(config["noise_arguments"])
        else:
            raise ValueError(
                f"Error: {config['mutator']} invalid. Must be one of {valid_mutators}"
            )

        # induce mutation on each dataset
        mutated = mutator_method(**mutator_args)
        mutated_data.append(mutated)
    return mutated_data


def collect_mutation_results(
    data_to_mutate, config, target_col, feature_cols, model_name, clf, num_experiments=100
):
    # Does this really need to hold the mutation?? IT's also in the config file...
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
            "model" : model_name,
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


def pick_up_run_data(run_path: str) -> pd.DataFrame:
    """
    This function reads and processes data from a specified output path.

    Args:
        run_path (str): The path to the directory containing the run data.
            This directory is expected to have a 'results.csv' file and a 'config.json' file.

    Returns:
        pd.DataFrame: A DataFrame containing the processed data.

    Raises:
        FileNotFoundError: If either 'results.csv' or 'config.json' is not found in the run path.
    """

    try:
        # Read results data
        df = pd.read_csv(f"{run_path}/results.csv")

        # Load model name from config
        config = load_config(f"{run_path}/config.json")
        df["model"] = config["model"]

        # Process metrics column
        # This is janky. Need to define metrics for this AND the metrics calculator outside functions
        for met in ["precision", "recall", "accuracy", "f1", "balanced-accuracy","positive-retrieval", "negative-retrieval"]:
            # Extract metric values
            df[met] = df["metrics"].apply(lambda x: np.array(ast.literal_eval(x)[met]))
            # Remove NaN
            df[met] = df[met].apply(lambda x: x[~np.isnan(x)])
            # Calculate mean, standard deviation, and median of each metric
            df[f"mean_{met}"] = df[met].apply(np.mean)
            df[f"sd_{met}"] = df[met].apply(np.std)
            df[f"median_{met}"] = df[met].apply(np.median)

        # Drop the original metrics column
        df = df.drop(columns=["metrics"])

        return df

    except FileNotFoundError as e:
        print(f"Error: Data files not found in run path: {run_path}")
        raise e  # Re-raise the exception for further handling


def retrieval_percent(y_true, y_pred, signal_type):
    """
    Calculates the proportion of retrieved positives (or negatives)

    Args:
        y_true: iterable of true labels (1 or 0)
        y_pred: iterable of predicted labels
        signal_type: 'pos' for positive class or 'neg' for negative class

    Returns:
        float: proportion of retrieved positives (or negatives)
    """

    if signal_type not in ["pos", "neg"]:
        raise ValueError("Type must be 'pos' or 'neg'")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    if signal_type == "pos":
        # True positives divided by total actual positives
        return np.sum(y_true & y_pred) / np.sum(y_true)
    else:
        # True negatives divided by total actual negatives
        return np.sum(~y_true & ~y_pred) / np.sum(~y_true)


def cost_penalty(y_true, y_pred, weight_dict):
    """
    Calculates a custom metric based on true labels, predicted labels and a weight dictionary

    Args:
        y_true: iterable of true labels (1 or 0)
        y_pred: iterable of predicted labels
        weight_dict: dictionary with keys 'tp', 'tn', 'fp', 'fn' for weights

    Returns:
        float: custom metric score
    """
    required_keys = ["tp", "tn", "fp", "fn"]
    if not all(key in weight_dict for key in required_keys):
        raise ValueError(f"weight_dict must contain keys: {', '.join(required_keys)}")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    # True positive, true negative, false positive, false negative
    tp = np.sum(y_true & y_pred)
    tn = np.sum(~y_true & ~y_pred)
    fp = np.sum(y_pred & ~y_true)
    fn = np.sum(~y_pred & y_true)

    # Calculate score using weights
    return (
        tp * weight_dict["tp"]
        + tn * weight_dict["tn"]
        + fp * weight_dict["fp"]
        + fn * weight_dict["fn"]
    )
