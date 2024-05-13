import random
import numpy as np
import pandas as pd

# Note: This shuffles the DataFrame upon return. Perhaps not ideal, but I don't think it matters

"""
Defines a Mutator class with methods for data mutation tasks.

**Methods:**

1. `return_original(self, df)`: Returns data without mutations applied
2. `change_to_zero(self, df, columns, percentage=0.0, seed=None, seed_modifier=53)`: Randomly changes a specified
   percentage of numerical values in each specified column of a DataFrame to 0
3. `transpose_columns(self, df, col1, col2)`: Transposes two columns of a dataframe.
4. `add_categorical_class(self, df, column_name, percentage=0.0, seed=None)`: Randomly changes entries in a
    specified categorical column to a new, unseen category.
5. `remove_class(self, df, column_name, class_seed=None, mutator_seed=None)`: Randomly selects a class from a
    categorical column and removes it, replacing all members with existing classes.
5. `add_noise(self, df, column_name, distribution="normal", mean=0, std=0.1, low=None, high=None, scale=1)`:
    Adds normal, uniform or exponential noise to a column.

**Usage:**

# Example 1: Return original data
data = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8]})
mutator = Mutator()
mutatedData = mutator.return_original(data)
print(mutatedData)

# Example 2: Change 30% of values in columns 'col1' and 'col2' to 0
data = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8]})
mutator = Mutator()
mutatedData = mutator.change_to_zero(data, columns=['col1', 'col2'], percentage=0.3, seed=42, seed_modifier=12)
print(mutatedData)

# Example 3: Transpose two data columns
data = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8]})
mutator = Mutator()
mutatedData = mutator.transpose_columns(data, 'col1', 'col2')
print(mutatedData)

# Example 4: Introduce a new categorical class and randomly assign 50% of entries to the new class
data = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8]})
mutator = Mutator()
mutatedData = mutator.add_categorical_class(data, 'col2', percentage=0.5, seed=314)
print(mutatedData)

# Example 5: Randomly lose a class and replace all members with other classes
data = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8]})
mutator = Mutator()
mutatedData = mutator.remove_class(data, 'col2', class_seed=314, mutator_seed=1718)
print(mutatedData)

# Example 6: Add normal(0, 0.1) noise
data = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8]})
mutator = Mutator()
df_with_normal_noise = add_noise(data, 'col2', distribution="normal", mean=0, std=0.1)

# Example 7: Add uniform(0,1) noise
data = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8]})
mutator = Mutator()
df_with_uniform_noise = add_noise(data, "col1", distribution="uniform", low=0, high=1)

# Example 8: Add exponential(1) noise
data = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8]})
mutator = Mutator()
df_with_exponential_noise = add_noise(data, "col1", distribution="exponential", scale=1)
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

    def change_to_zero(self, df, columns, percentage, seed=None, seed_modifier=None):
        """
        Randomly changes a specified percentage of numerical values in each column of a DataFrame to 0. Note: This will
        alter the index for the returned DataFrame

        Args:
            df (pandas.DataFrame): The DataFrame to mutate.
            columns (list): A list of column names to mutate.
            percentage (float, optional): The percentage of values to change to 0 in each column.
            seed (int, optional): The random seed for randomizing the DataFrame. If None, a random seed will be
                generated. Defaults to None.
            seed_modifier (int, optional): The random seed for randomizing each column during mutation.
                If None, a random seed will be generated. Defaults to None.

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
        # Make  copy for changes
        df_internal = df.copy(deep=True)
        # Generate a random seed if none is provided
        if seed is None:
            seed = np.random.randint(100000)

            # Iterate through columns
        for col in columns:
            # Set inner seed using random component and column hash
            if seed_modifier is None:
                seed_modifier = np.random.randint(100000)

            inner_seed = seed * seed_modifier + (hash(col) % 100000)

            # Randomly select indices to modify
            num_values_to_change = int(percentage * len(df_internal))
            random_indices = df_internal.sample(
                num_values_to_change, random_state=inner_seed
            ).index

            # Modify values at random indices to 0
            df_internal.loc[random_indices, col] = 0

        return df_internal  # Return the mutated DataFrame

    def transpose_columns(self, df, col1, col2):
        """Transposes two columns in the dataset to simulate a key transposition.

        Args:
            df (pandas.DataFrame): The DataFrame to mutate.
            col1: First column to transpose.
            col2: Second column to transpose.

        Raises:
            TypeError: If either column does not exist in the DataFrame.
        """
        df_internal = df.copy(deep=True)
        # Check that both columns exist
        for col in [col1, col2]:
            if col not in df_internal.columns:
                raise TypeError(f"Column {col} must exist in DataFrame.")

        # Transpose columns
        df_transposed = df_internal.rename({col1: col2, col2: col1})

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

        df_internal = df.copy(deep=True)
        # Check if the column exists
        if column_name not in df_internal.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

        # Check if the column is numeric
        if not pd.api.types.is_numeric_dtype(df_internal[column_name]):
            raise ValueError(f"Column '{column_name}' is not numeric.")

        # Check if the column is of integer type
        if not pd.api.types.is_integer_dtype(df_internal[column_name]):
            raise ValueError(f"Column '{column_name}' is not of integer type.")

        # Set random seed (if provided)
        if seed is not None:
            random.seed(seed)

        # Get existing distinct values
        existing_values = df_internal[column_name].unique()

        # Infer a new class label (assuming integer increment)
        new_class_label = max(existing_values) + 1

        # Randomly modify entries to the new class (if applicable)
        if percentage > 0:
            if percentage < 0 or percentage > 1:
                raise ValueError("Percentage must be between 0 and 1.")
            num_rows = len(df_internal)
            num_to_change = int(percentage * num_rows)

            # Sample indices without replacement using random.sample
            indices_to_change = random.sample(range(num_rows), num_to_change)
            for i in indices_to_change:
                df_internal.loc[i, column_name] = new_class_label

        return df_internal

    def remove_class(self, df, column_name, class_seed=None, mutator_seed=None):
        """
        Modifies a specified column in a DataFrame, randomly selecting a class to remove and redistributing
            members to other classes in a seeded fashion.

        Args:
            df (pandas.DataFrame): The DataFrame to modify.
            column_name (str): The name of the column to modify.
            class_seed (int, optional): The random seed for selecting the class to remove. Defaults to None.
            mutator_seed (int, optional): The random seed for redistributing members. Defaults to None.

        Raises:
            ValueError: If the column does not exist in the DataFrame,
                        or if there are less than two unique classes in the column.
        """

        df_internal = df.copy(deep=True)
        # Check if the column exists
        if column_name not in df_internal.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

        # Check if there are at least two unique classes
        if len(df_internal[column_name].unique()) < 2:
            raise ValueError("There must be at least two unique classes in the column.")

        # Set random seeds (if provided)
        if class_seed is not None:
            random.seed(class_seed)

        # Get all classes
        all_classes = df_internal[column_name].unique()

        # Randomly select a class to remove using the selection seed
        class_to_remove = random.choice(all_classes)

        # Get all classes (excluding the one to remove)
        remaining_classes = all_classes[all_classes != class_to_remove]

        # Get indices of entries with the class to remove
        indices_to_change = df_internal[
            df_internal[column_name] == class_to_remove
        ].index

        # Set seed for member redistribution (if provided)
        if mutator_seed is not None:
            random.seed(mutator_seed)

        # Randomly select a new class for each entry (excluding the original class)
        new_classes = random.choices(remaining_classes, k=len(indices_to_change))

        # Update the DataFrame with the new classes
        df_internal.loc[indices_to_change, column_name] = new_classes

        return df_internal

    def add_noise(
        self,
        df,
        column_name,
        distribution="normal",
        mean=0,
        std=0.1,
        low=None,
        high=None,
        scale=1,
    ):
        """
        Adds random noise to a column in a pandas dataframe.

        Args:
            df: A Pandas DataFrame.
            column_name: Name of the column to add noise to.
            distribution: The type of noise distribution to use (default: "normal").
                Supported distributions: "normal", "uniform", "exponential".
            mean: The mean of the noise (default: 0).
            std: The standard deviation of the noise (default: 0.1).
                Used for normal distribution only.
            low: The lower bound of the noise distribution (default: None).
                Used for uniform distribution only.
            high: The upper bound of the noise distribution (default: None).
                Used for uniform distribution only.
            scale: The scale parameter of the exponential distribution (default: 1).
                Controls the spread of the distribution.

        Returns:
            A new pandas dataframe with the noise added to the specified column.
        """
        df_internal = df.copy(deep=True)
        if distribution == "normal":
            noise = np.random.normal(loc=mean, scale=std, size=len(df))
        elif distribution == "uniform":
            if low is None or high is None:
                raise ValueError(
                    "low and high must be specified for uniform distribution"
                )
            noise = np.random.uniform(low=low, high=high, size=len(df))
        elif distribution == "exponential":
            noise = np.random.exponential(scale=scale, size=len(df))
        else:
            raise ValueError("Unsupported noise distribution:", distribution)

        df_internal[column_name] = df_internal[column_name] + noise
        return df_internal
