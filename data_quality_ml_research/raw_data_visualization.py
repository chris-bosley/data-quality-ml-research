import tools as dq
from dfvisualization import Visualizations

def main():

    # bank_data = dq.read_data("../data/bank-additional-full.csv", separator=";")
    shopper_data = Visualizations(dq.read_data("../data/online_shoppers_intention.csv"))
    shopper_data.create_and_display_plots()

if __name__ == "__main__":
    main()
