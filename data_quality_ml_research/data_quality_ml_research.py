"""Main module."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV

import helpers as dq


def main():
    # Declare configuration variables
    config = {
        "model": "Logistic Regression",
        "data_location": "../data/online_shoppers_intention.csv",
        "preprocess_data": False,
        "training_percent": 0.5,
        "mutator": "change_to_zero",
        "num_scoring_groups": 5,
        "seeds": {"outer_seed": 123, "inner_seed": 456},
    }

    # Declare required classifier
    # Look at default parameters OR quick grid search
    clf = LogisticRegressionCV(Cs=[0.1, 1, 10], fit_intercept=True)

    # Ingest dataset passed from command line
    data = dq.read_csv_to_dataframe("../data/online_shoppers_intention.csv")

    # We'll also have to handle data more carefully than this...
    data.drop(["Month", "VisitorType", "Weekend"], axis=1, inplace=True)

    # identify features and targets. We'll have to specify or assume last or something here
    feature_cols = data.columns.to_list()[:-1]
    target_col = data.columns.to_list()[-1]

    # Split data
    # Build Scoring Groups
    scoring_group_size = (1 - config["training_percent"]) / config["num_scoring_groups"]
    scoring_proportions = [config["training_percent"]]
    for i in range(config["num_scoring_groups"]):
        scoring_proportions.append(scoring_group_size)

    # Build splits
    datasets = dq.random_nonintersecting_subsets(
        data, proportions=scoring_proportions, seed=config["seeds"]["inner_seed"]
    )

    # Fit declared model
    clf.fit(datasets[0][feature_cols], datasets[0][target_col])

    # Discard training data
    datasets.pop(0)

    # print(confusion_matrix(test[target_col], clf_list[0][0].predict(test[feature_cols])))

    # Mutate all datasets
    # Check that mutator is mutating correct proportions...
    all_runs = []
    for sev in np.arange(0, 1.01, 0.1):
        sev = round(sev, 2)
        mutated_data = dq.get_mutations(datasets, sev, config)
        run_results = {
            "mutation": "change_to_zero",
            "severity": sev,
            "metrics": dq.get_mutation_metrics(
                mutated_data, target_col, feature_cols, clf
            ),
        }
        all_runs.append(run_results)

    pd.DataFrame(all_runs).to_csv("../output_data/output.csv")


if __name__ == "__main__":
    main()
