import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tools as dq

"""
Defines a Visualization class with methods for data visualization tasks.

**Methods:**

1. count_column_nulls(self, df): returns an array with key value pairs of each column and the number of nulls in that column
2. column_data_type(self, df): returns an array with the data type of each column in a df
3. hist_plot(self, df): creates a historgram or plot with the instances of each value in a plot




"""
class Visualizations:
    def __init__(self, df):
        if not isinstance(df, pd.DataFrame): # Enforce dataframe
            raise TypeError("Input must be a pandas DataFrame")
        self.df = df.copy()  # Avoid modifying the original dataframe

    """
    This method calculates the count of null values in each column of the dataframe
    and returns a dictionary with column names as keys and null value counts as values.
    """
    @property
    def null_count(self):
        null_value_counts = self.df.isnull().sum()
        null_value_counts_dict = null_value_counts.to_dict()
        return null_value_counts_dict

    """
    Calculates the number of rows and columns for a square subplot grid,
    ensuring more rows than columns.
    """

    def calculate_subplot_grid(self, num_df_columns):

        num_subplot_rows = int(np.round(num_df_columns**0.5))
        num_subplot_columns = int(np.ceil(num_df_columns**0.5))
        return num_subplot_rows, num_subplot_columns

    def plot_column(self, col, ax):
        """
        Analyzes a column of data and creates a corresponding visualization on the provided axis.
        """
        # Print the number of nulls in the column
        # print(f"Number of null values in '{col}': {self.df[col].isnull().sum()}")

        # Check data type and create plot accordingly
        if pd.api.types.is_bool_dtype(self.df[col]):
            sns.histplot(self.df[col].astype(int), ax=ax)
            ax.set_title(f"Histogram of Boolean {col}")
        elif pd.api.types.is_numeric_dtype(self.df[col]):
            sns.histplot(self.df[col], ax=ax)
            ax.set_title(f"Histogram of {col}")
        elif pd.api.types.is_string_dtype(self.df[col]):
            sns.countplot(x=self.df[col], ax=ax)
            ax.set_title(f"Frequency of Values in {col}")
        else:
            print(f"{col} has an error and was skipped")

        # Set common styling options
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.grid(True)
    def create_and_display_plots(self):
        """
        Creates a square subplot grid, analyzes and visualizes each column of data,
        adjusts layout, and displays the final plot.
        """
        num_df_columns = len(self.df.columns)
        num_rows, num_columns = self.calculate_subplot_grid(num_df_columns)
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 10))

        # Iterate through columns and create plots
        for i, col in enumerate(self.df.columns):
            chart_row = i // num_columns
            chart_col = i % num_columns
            if chart_row < num_rows and chart_col < num_columns:
                ax = axes[chart_row, chart_col]
                self.plot_column(col, ax)  # Avoid modifying original data

        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()




# Defines a Mutator class with methods for data mutation tasks.
#
# **Methods:**
#
# 1. `return_original(self, df)`: Returns data without mutations applied
# 2. `change_to_zero(self, df, columns, percentage=0.0, seed=None, seed_modifier=53)`: Randomly changes a specified
#    percentage of numerical values in each specified column of a DataFrame to 0
# 3. `transpose_columns(self, df, col1, col2)`: Transposes two columns of a dataframe.
# 4. `add_categorical_class(self, df, column_name, percentage=0.0, seed=None)`: Randomly changes entries in a
#     specified categorical column to a new, unseen category.
# 5. `remove_class(self, df, column_name, class_seed=None, mutator_seed=None)`: Randomly selects a class from a
#     categorical column and removes it, replacing all members with existing classes.
