import random

import numpy as np
import pandas as pd

# Note: This shuffles the DataFrame upon return. Perhaps not ideal, but I don't think it matters

"""
Defines a Mutator class with methods for data manipulation tasks.

**Methods:**

1. `return_original(self, df)`: Returns data without mutations applied
2. `change_to_zero(self, df, columns, percentage=0.0, seed=None, seed_modifier=53)`: Randomly changes a specified
   percentage of numerical values in each specified column of a DataFrame to 0
3. `transpose_columns(self, df, col1, col2)`: Transposes two columns of a dataframe.
4. `add_categorical_class(self, df, column_name, percentage=0.0, seed=None)`: Randomly changes entries in a
    specified categorical column to a new, unseen category.

**Usage:**

# Example 1: Return original data
data = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8]})
mutator = Mutator()
mutatedData = mutator.return_original(data)
print(mutatedData)

# Example 2: Change 30% of values in columns 'col1' and 'col2' to 0
data = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8]})
mutator = Mutator()
mutatedData = mutator.change_to_zero(data, columns=['col1', 'col2'], percentage=0.3, seed=42)
print(mutatedData)

# Example 3: Transpose two data columns
data = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8]})
mutator = Mutator()
mutatedData = mutator.transpose_columns(data, 'col1', 'col2')
print(mutatedData)

# Example 4: Introduce a new categorical class and randomly assign 50% of entries to the new class
data = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8]})
mutator = Mutator()
mutatedData = mutator.add_categorical_class(data, 'col2'', percentage=0.5, seed=314)
print(mutatedData)
"""


# noinspection PyUnboundLocalVariable,PyMethodMayBeStatic
class Mutator:
    """
    A class for data manipulation tasks on DataFrames.
    """

    def return_original(self, df):
        """
        Returns the original data. This is the identity mutation.

        Args:
            df (pandas.DataFrame): The DataFrame to mutate.
        """
        return df

    def change_to_zero(self, df, columns, percentage=0.0, seed=789, seed_modifier=53):
        """
        Randomly changes a specified percentage of numerical values in each column of a DataFrame to 0.

        Args:
            df (pandas.DataFrame): The DataFrame to mutate.
            columns (list): A list of column names to mutate.
            percentage (float, optional): The percentage of values to change to 0 in each column. Defaults to 0.0.
            seed (int, optional): The random seed for randomizing the DataFrame. Defaults to None (non-deterministic).
            seed_modifier (int, optional): The value used to salt the inner seed for each column. Defaults to 53.

        Raises:
            ValueError: If the percentage is not between 0 and 1.
            TypeError: If any specified column does not exist in the DataFrame or
                       if any column data type is not numerical.
        """

        # Check if percentage is valid
        if percentage < 0 or percentage > 1:
            raise ValueError("Percentage must be between 0 and 1.")

        # Check if all columns exist and are numerical
        for col in columns:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                raise TypeError(
                    f"Column '{col}' must exist and contain numerical data."
                )

        # Set the seed
        np.random.seed(seed)

        # Iterate through columns
        for col in columns:
            # Create an inner seed to improve randomization in subsequent columns
            inner_seed = seed * seed_modifier + (hash(col) % 100000)
            np.random.seed(inner_seed)

            # Randomize the row order for this column
            df_shuffled = df.sample(frac=1, random_state=inner_seed)

            # Calculate the number of values to change
            num_values_to_change = int(percentage * len(df_shuffled))

            # Modify values in the current column until the target number of changes is reached
            for j in range(len(df_shuffled)):
                if pd.api.types.is_numeric_dtype(df_shuffled.loc[j, col]):
                    df_shuffled.loc[j, col] = 0
                    num_values_to_change -= 1
                    if num_values_to_change == 0:
                        break  # Early termination if we're done with changes

        return df_shuffled  # Return the mutated DataFrame

    def transpose_columns(self, df, col1, col2):
        """Transposes two columns in the dataset to simulate a key transposition.

        Args:
            df (pandas.DataFrame): The DataFrame to mutate.
            col1: First column to transpose.
            col2: Second column to transpose.

        Raises:
            TypeError: If either column does not exist in the DataFrame.
        """

        # Check that both columns exist
        for col in [col1, col2]:
            if col not in df.columns:
                raise TypeError(f"Column {col} must exist in DataFrame.")

        # Transpose columns
        df_transposed = df.rename({col1: col2, col2: col1})

        return df_transposed

    def add_categorical_class(self, df, column_name, percentage=0.0, seed=None):
        """
        Modifies a categorical column in a DataFrame, inferring a new class and randomly changing entries.

        Args:
            df (pandas.DataFrame): The DataFrame to modify.
            column_name (str): The name of the column to modify.
            percentage (float, optional): The percentage of entries to change to the new
                class (between 0 and 1). Defaults to 0.0.
            seed (int, optional): The random seed for repeatability. Defaults to None.

        Raises:
            ValueError: If the column does not exist in the DataFrame,
                        or if the percentage is not between 0 and 1.
        """

        # Check if the column exists
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

        # Check if the column is numeric
        if not pd.api.types.is_numeric_dtype(df[column_name]):
            raise ValueError(f"Column '{column_name}' is not numeric.")

        # Check if the column is of integer type
        if not pd.api.types.is_integer_dtype(df[column_name]):
            raise ValueError(f"Column '{column_name}' is not of integer type.")

        # Set random seed (if provided)
        if seed is not None:
            random.seed(seed)

        # Get existing distinct values
        existing_values = df[column_name].unique()

        # Infer a new class label (assuming integer increment)
        new_class_label = max(existing_values) + 1

        # Randomly modify entries to the new class (if applicable)
        if percentage > 0:
            if percentage < 0 or percentage > 1:
                raise ValueError("Percentage must be between 0 and 1.")
            num_rows = len(df)
            num_to_change = int(percentage * num_rows)

            # Sample indices without replacement using random.sample
            indices_to_change = random.sample(range(num_rows), num_to_change)
            for i in indices_to_change:
                df.loc[i, column_name] = new_class_label

        return df
