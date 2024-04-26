import numpy as np
import pandas as pd

# Note: This shuffles the DataFrame upon return. Perhaps not ideal, but I don't think it matters

"""
Defines a Mutator class with methods for data manipulation tasks.

**Methods:**

1. `change_to_zero(self, df, columns, percentage=0.0, seed=None, seed_modifier=53)`: Randomly changes a specified
   percentage of numerical values in each column of a DataFrame to 0

**Usage:**

# Example 1: Change 10% of values in columns 'col1' and 'col2' to 0
data = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8]})
mutator = Mutator()
mutatedData = mutator.change_to_zero(data, columns=['col1', 'col2'], percentage=0.1, seed=42)
print(mutatedData)
"""


class Mutator:
    """
    A class for data manipulation tasks on DataFrames.
    """

    def change_to_zero(self, df, columns, percentage=0.0, seed=None, seed_modifier=53):
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
