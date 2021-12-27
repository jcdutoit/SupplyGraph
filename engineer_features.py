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
        self.graph = nx.from_pandas_edgelist(self.edge_list, "source", "target").to_undirected()

        # Get the maximum color value

        # Interestingly the graph is not fully connected there are 4 connected components

        max_distance = 20 # This is a magic number Our graph is not connected but no length should be larger than 20
        self.max_color = None
        self.color_hash(np.array([max_distance]), np.array([max_distance]))

    def color_hash(self, d_x, d_y):
        """
        Returns hashes for the colors of the nodes based on topological importance in the
        subgraph surrounding x and y. It may be of interest to add financial X 
        into this hashing function so that color is based on both distance and financial data.
        """

        # Initialize different distance vectors
        d_xy = np.vstack((d_x, d_y))
        d = d_x + d_y
        min_d = np.min(d_xy, axis=0)

        # Hash function
        colors = np.ones(len(d_x)) + min_d + d / 2 * (d / 2 + d % 2 - 1)

        # Create one-hot matrix

        if self.max_color is None:
            # If this is the first time this function is called, set the max color
            self.max_color = colors
            return

        one_hot_colors = np.zeros((len(colors), 1 + int(self.max_color)))
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
        print(G.number_of_nodes())

        # Find the distances from each node to x and y
        d_x = np.array(list(nx.algorithms.shortest_paths.generic.shortest_path_length(G, target=x).values()))
        d_y = np.array(list(nx.algorithms.shortest_paths.generic.shortest_path_length(G, target=y).values()))

        print(d_x, d_y)
        # Get the color hash for each node
        return self.color_hash(d_x, d_y)

    def generate_training_data(self, k_hop=1, save=True):
        """
        Generate a training set of one_hot vectors
        """

        node_list = list(self.graph.nodes)
        X = []
        Y = []

        for i in range(len(node_list)):
            for j in range(i, len(node_list)):
                print("Generating training data for " + str(i) + " and " + str(j))
                v = self.get_color_encoding(node_list[i], node_list[j], k_hop)
                X.append(v)
                Y.append(int(self.graph.has_edge(node_list[i], node_list[j])))
        
        max_len = max([v.shape[0] for v in node_list])

        # Pad the X to the maximum length
        padded_X = []
        for v in node_list:
            if v.shape[0] < max_len:
                v = np.pad(v, (0, max_len - v.shape[0]), 'constant')
            padded_X.append(v)

        if save:
            np.save("X.npy", np.array(padded_X))
            np.save("Y.npy", np.array(Y))
        
        return padded_X, Y

    
if __name__ == "__main__":
    sg = SupplyGraph()
    
    # print(one_hot_colors.shape)
    # plt.imshow(one_hot_colors, aspect='auto', cmap='viridis')
    # plt.show()

    sg.generate_training_data()
    # This seems to have problems with the graph being disconnected.