"""Main module."""

from sklearn.linear_model import LogisticRegressionCV

import helpers as dq
import mutation

# Declare all potential classifiers
lr = LogisticRegressionCV(Cs=[0.1, 1, 10], fit_intercept=True)

clf_list = [(lr, "Logistic Regression")]

# Ingest dataset passed from command line
data = dq.read_csv_to_dataframe("../data/online_shoppers_intention.csv")
# We'll also have to handle data more carefully than this...
data.drop(["Month", "VisitorType", "Weekend"], axis=1, inplace=True)

# identify features and targets. We'll have to specify or assume last or something here
featureCols = data.columns.to_list()[:-1]
targetCol = data.columns.to_list()[-1]

# Split data
train, test, validate = dq.random_nonintersecting_subsets(
    data, proportions=[0.5, 0.3, 0.2], seed=123
)

# Fit declared models
for clf, name in clf_list:
    print("... " + name)
    clf.fit(train[featureCols], train[targetCol])


# print(confusion_matrix(test[targetCol], clf_list[0][0].predict(test[featureCols])))

# Test a mutation
mutator = mutation.Mutator()
validate_mutated = mutator.change_to_zero(
    validate,
    columns=["OperatingSystems", "PageValues"],
    percentage=0.35,
    seed=123,
    seed_modifier=123,
)


# confusion_matrix(validate[targetCol], clf_list[0][0].predict(validate[featureCols]))
# confusion_matrix(validate_mutated[targetCol], clf_list[0][0].predict(validate_mutated[featureCols]))
