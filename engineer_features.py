import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

RAW_DATA_PATH = "./Autodatacomplete.xlsx"
EDGE_LIST_PATH = "./edge_list.csv"

def construct_edge_list(data_path=RAW_DATA_PATH, save_path=EDGE_LIST_PATH):
    """
    Create an edge list csv with source, target columns from the raw data.
    """
    raw_data = pd.read_excel(data_path)
    edge_list = []
    print("Creating edge_list. This may take a while...")

    # The data has the source as the first element in the row and the targets as the rest of the columns.
    row = 0
    for i in range(len(raw_data)):
        for j in range(2,len(raw_data.iloc[i])):
            # Make sure the target in not NaN
            if not pd.isna(raw_data.iloc[i,j]):
                edge_list.append([raw_data.iloc[i,1], raw_data.iloc[i,j], 1])
                row += 1

    print("Done! Saving to " + save_path)
    edge_df = pd.DataFrame(edge_list, columns=["source", "target", "weight"]) 
    edge_df.to_csv(save_path, index=False)

class SupplyGraph():
    """
    This class is used to create the feature vectors that will be passed into a 
    GNN classifier for determining if a pair of nodes are connected (or not connected).
    It is an implementation of the paper 'A machine learning approach for predicting hidden
    links in supply chain data with graph neural networks' 
    https://doi.org/10.1080/00207543.2021.1956697
    """
    
    def __init__(self, edge_list_path=EDGE_LIST_PATH):

        self.edge_list = pd.read_csv(edge_list_path)
        self.graph = nx.from_pandas_edgelist(self.edge_list, "source", "target")

    def color_hash(self, d_x, d_y):
        """
        Returns hashes for the colors of the nodes based on topological importance in the
        subgraph surrounding x and y. It may be of interest to add financial features 
        into this hashing function so that color is based on both distance and financial data.
        """

        # Initialize different distance vectors
        d_xy = np.vstack((d_x, d_y))
        d = d_x + d_y
        min_d = np.min(d_xy, axis=0)

        # Hash function
        colors = np.ones(len(d_x)) + min_d + d / 2 * (d / 2 + d % 2 - 1)
        max_color = np.max(colors)

        # Create one-hot matrix
        one_hot_colors = np.zeros((len(colors), 1 + int(max_color)))
        one_hot_colors[np.arange(len(colors)), colors.astype(int)] = 1
        return one_hot_colors

    def get_color_encoding(self, x, y, k_hop=1):
        """
        Get the color encodings for the nodes in the k_hop subgraph surrounding x and y.
        """

        # Get the subgraph
        x_subgraph = nx.generators.ego_graph(self.graph, x, radius=k_hop)
        y_subgraph = nx.generators.ego_graph(self.graph, y, radius=k_hop)
        G = nx.compose(x_subgraph, y_subgraph)

        # Find the distances from each node to x and y
        d_x = np.array(list(nx.algorithms.shortest_paths.generic.shortest_path_length(G, target=x).values()))
        d_y = np.array(list(nx.algorithms.shortest_paths.generic.shortest_path_length(G, target=y).values()))

        # Get the color hash for each node
        return self.color_hash(d_x, d_y)

    
if __name__ == "__main__":
    sg = SupplyGraph()
    # This graph is suprisingly well connected. Using a 1 hop neighborhood, all nodes were directly connected.
    one_hot_colors = sg.get_color_encoding(x="NXPI US Equity", y="BRSS US Equity", k_hop=2)
    print(one_hot_colors.shape)
    plt.imshow(one_hot_colors, aspect='auto', cmap='viridis')
    plt.show()