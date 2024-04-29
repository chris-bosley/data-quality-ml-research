"""Main module."""

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix

import helpers as dq
import mutation


def main():
    # Declare all potential classifiers
    lr = LogisticRegressionCV(Cs=[0.1, 1, 10], fit_intercept=True)

    clf_list = [(lr, "Logistic Regression")]

    # Ingest dataset passed from command line
    data = dq.read_csv_to_dataframe("../data/online_shoppers_intention.csv")
    # We'll also have to handle data more carefully than this...
    data.drop(["Month", "VisitorType", "Weekend"], axis=1, inplace=True)

    # identify features and targets. We'll have to specify or assume last or something here
    feature_cols = data.columns.to_list()[:-1]
    target_col = data.columns.to_list()[-1]

    # Split data
    train, test, validate = dq.random_nonintersecting_subsets(
        data, proportions=[0.5, 0.3, 0.2], seed=123
    )

    # Fit declared models
    for clf, name in clf_list:
        print("... " + name)
        clf.fit(train[feature_cols], train[target_col])

    # print(confusion_matrix(test[target_col], clf_list[0][0].predict(test[feature_cols])))

    # Test a mutation
    mutator = mutation.Mutator()
    validate_mutated = mutator.change_to_zero(
        validate,
        columns=["OperatingSystems", "PageValues"],
        percentage=0.35,
        seed=123,
        seed_modifier=123,
    )

    print(
        confusion_matrix(
            validate[target_col], clf_list[0][0].predict(validate[feature_cols])
        )
    )
    print(
        confusion_matrix(
            validate_mutated[target_col],
            clf_list[0][0].predict(validate_mutated[feature_cols]),
        )
    )


if __name__ == "__main__":
    main()
