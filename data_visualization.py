import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

"""
This is very old. It was used for visualizing the old auto data.
"""
AUTO_FINANCIAL_DATA = "../data/supply_data/auto_financial.csv"
# FEATURE_OF_INTEREST = "EBIDTA/ToRev.(operating profitability)"
# FEATURE_OF_INTEREST = "Trnovr"
# FEATURE_OF_INTEREST = "Tobin's Q"
FEATURE_OF_INTEREST = "Total Rev"

def filter_data():
    """
    Filter features and profit data
    """
    raw = pd.read_csv(AUTO_FINANCIAL_DATA)
    profit = raw[FEATURE_OF_INTEREST]

    # Get rid of columns that are not viable features for predicting profitability
    drop_cols = ['Id', 'SICa', 'S<20','M21-79','L>80', FEATURE_OF_INTEREST]
    raw.drop(drop_cols, axis=1, inplace=True)

    return raw, profit

def visualize_relationships():
    features, profit = filter_data()

    cat_headers = ["SIC", "Country", "sizex"]
    cat_df = pd.DataFrame([features[i] for i in cat_headers])
    

    num_df = features.drop(cat_headers, axis=1)
    # num_df[(np.abs(stats.zscore(num_df)) < 3).all(axis=1)]

    for col in num_df:
        fig = plt.figure()
        ax = plt.gca()
        ax.scatter(num_df[col].to_numpy(), profit.to_numpy())
        # plt.ylim(-1,1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(col)
        ax.set_ylabel(FEATURE_OF_INTEREST)
        # plt.ylim(0,50)
        # plt.xlim(0,50)
        # ax.ylim(0,10)
        plt.show()

    
    # for col in cat_headers:
    #     plt.hist(cat_df, profit.to_numpy)
    

def measure_corrs():

    features, profit = filter_data()

    # Separate categorical data from numerical data
    cat_headers = ["SIC", "Country", "sizex"]
    cat_df = pd.pd.concat(features, axis=1, keys=cat_headers)
    num_df = features.drop(cat_headers, axis=1)

    # corr_coefs = [np.corrcoef(i.to_array())]

def make_production_graph():

    features, profit = filter_data()

    adj = np.load("../data/supply_data/adjacency.npy")

    print(profit.shape)
    print(adj.shape)

def make_adj_mat():
    edges = pd.read_excel("../data/supply_data/auto.xlsx")
    known_nodes = {}
    
    print("Adding edges to known edges...")
    for i, row in edges.iterrows():
        if not edges['Source'].loc[i] in known_nodes:
            known_nodes[edges['Source'].loc[i]] = len(known_nodes)
            # print("Added key ", edges['Source'].loc[i], " with value ", len(known_nodes))
    
    print("Adding edges to adj...")
    adj = np.zeros((len(known_nodes), len(known_nodes)))
    for i, row in edges.iterrows():
        id1 = edges['Source'].loc[i]
        id2 = edges['Target'].loc[i]
        # print("Adding edge from ", id1, " to ", id2)
        idx1 = known_nodes.get(id1)
        idx2 = known_nodes.get(id2)
        adj[idx1, idx2] = 1

    np.savetxt("../data/supply_data/adj2.csv", adj, delimiter=",")
    


if(__name__ == "__main__"):
    visualize_relationships()
    # make_production_graph()
    # make_adj_mat()