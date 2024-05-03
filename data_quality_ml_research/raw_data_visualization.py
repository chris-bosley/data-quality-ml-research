import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import helpers as dq

def main():
    shopper_data = dq.read_csv_to_dataframe("../data/online_shoppers_intention.csv")
    bank_data = dq.read_csv_to_dataframe("../data/bank-additional-full.csv")

    # Create rows and columns of the subplots to create a square
    # With more rows than columns if necessary
    # Calculate number of columns and rows based on the number of columns
    sd_cols = len(shopper_data.columns)
    num_columns = int(np.ceil(sd_cols ** 0.5))
    num_rows = int(np.round(sd_cols ** 0.5))  # Ensure fewer rows
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 10))

    # Iterate through columns in shopper data
    for i, col in enumerate(shopper_data.columns):
        chart_row = i // num_columns
        chart_col = i % num_columns
        if chart_row < num_rows and chart_col < num_columns:
            ax = axes[chart_row, chart_col]
        # First print the number of nulls in the column
        print(f"Number of null values in '{col}': {shopper_data[col].isnull().sum()}")
        # Check if the column is boolean first.
        # This MUST come before the numeric call
        if pd.api.types.is_bool_dtype(shopper_data[col]):
            ax.hist(shopper_data[col].astype(int))
            ax.set_title(f"Histogram of Boolean {col}")
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.grid(True)
            print(f"{col} is boolean")

       #Check if the column is numeric EXCEPT boolean.
        elif pd.api.types.is_numeric_dtype(shopper_data[col]):
            #create a histogram for numerical columns
            ax.hist(shopper_data[col])
            ax.set_title(f"Histogram of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.grid(True)
            print(f"{col} is numeric")
        elif pd.api.types.is_string_dtype(shopper_data[col]):
            # Create value count chart for string columns
            value_counts = shopper_data[col].value_counts()
            ax.bar(value_counts.index, value_counts.values)
            ax.set_title(f"Frequency of Values in {col}")
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.grid(True)
            print(f"{col} is a string")
        else:
            print(f"{col} has an error and was skipped")
    # Adjust layout
    plt.tight_layout()

    # Display the plot
    plt.show()




if __name__ == "__main__":
    main()
