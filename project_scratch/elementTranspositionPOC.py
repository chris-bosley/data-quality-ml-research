from sklearn import datasets
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def random_nonintersecting_subsets(df, proportions, seed=None):
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
                subset = df_copy.iloc[start_idx:start_idx + size]
                these_subsets.append(subset)
                start_idx += size
    else:
        raise ValueError("Total subset size exceeds the number of rows in the DataFrame.")

    return these_subsets


# Load Iris dataset
iris = datasets.load_iris()
irisPure = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                        columns=iris['feature_names'] + ['target'])
# irisPure.head()
species = []

for i in range(len(irisPure['target'])):
    if irisPure['target'][i] == 0:
        species.append("setosa")
    elif irisPure['target'][i] == 1:
        species.append('versicolor')
    else:
        species.append('virginica')

irisPure['species'] = species

setosa = irisPure[irisPure.species == "setosa"]
versicolor = irisPure[irisPure.species == 'versicolor']
virginica = irisPure[irisPure.species == 'virginica']

'''
Scatter plots of petal and sepal dimensions

fig, ax = plt.subplots()
fig.set_size_inches(13, 7) # adjusting the length and width of plot

ax.scatter(setosa['petal length (cm)'], setosa['petal width (cm)'], label="Setosa", facecolor="blue")
ax.scatter(versicolor['petal length (cm)'], versicolor['petal width (cm)'], label="Versicolor", facecolor="green")
ax.scatter(virginica['petal length (cm)'], virginica['petal width (cm)'], label="Virginica", facecolor="red")


ax.set_xlabel("petal length (cm)")
ax.set_ylabel("petal width (cm)")
ax.grid()
ax.set_title("Iris petals")
ax.legend()

plt.show()

fig, ax = plt.subplots()
fig.set_size_inches(13, 7) # adjusting the length and width of plot

ax.scatter(setosa['sepal length (cm)'], setosa['sepal width (cm)'], label="Setosa", facecolor="blue")
ax.scatter(versicolor['sepal length (cm)'], versicolor['sepal width (cm)'], label="Versicolor", facecolor="green")
ax.scatter(virginica['sepal length (cm)'], virginica['sepal width (cm)'], label="Virginica", facecolor="red")


ax.set_xlabel("sepal length (cm)")
ax.set_ylabel("sepal width (cm)")
ax.grid()
ax.set_title("Iris sepals")
ax.legend()

plt.show()
'''

# Split data into seeded random subsets
train, test, validate = random_nonintersecting_subsets(irisPure, [0.5, 0.3, 0.2], 1718)

# Define and fit classifier
clf = LogisticRegression()
clf_fitted = clf.fit(train[iris['feature_names']], train['target'])

# Check confusion matrices on original data
confusion_matrix(train['target'], clf_fitted.predict(train[iris['feature_names']]))
confusion_matrix(test['target'], clf_fitted.predict(test[iris['feature_names']]))
confusion_matrix(validate['target'], clf_fitted.predict(validate[iris['feature_names']]))

# Transposing columns in the validate dataset
validate_transposed_columns = validate.rename(columns={'sepal length (cm)': 'petal length (cm)',
                                                       'sepal width (cm)': 'petal width (cm)',
                                                       'petal length (cm)': 'sepal length (cm)',
                                                       'petal width (cm)': 'sepal width (cm)'})

# Confusion matrix for erroneous dataset
confusion_matrix(
    validate_transposed_columns['target'],
    clf_fitted.predict(validate_transposed_columns[iris['feature_names']]))
