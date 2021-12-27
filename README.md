# SupplyGraph

## Summary

This repo contains preliminary data analysis and a partial implementation of a link prediction network as described in the paper link_prediction.pdf in the repository. The file Autodatacomplete.xlsx contains supply chain data for the automotive industry pulled from Bloomberg's terminal. This data still needs some cleaning, since the Bloomberg terminal threw errors a few times that we still have not been able to debug. Further research could clean this data and try to predict where the Bloomberg terminal data is lacking. For the purposes of this research, we assumed the data was complete. The file edgelist.csv contains the complete edge list of the supply chain graph built from Autodatacomplete.xlsx. The code for building the edgelist is in engineer_features.py.

The engineer_features.py file also contains color encoding for a k_hop subgraph as described in the link_prediction paper. I did not finish the next steps of the project which include training a Graph Convolutional Network (GCN) and binary classification network for determining if a link exists or not. 

The data_visualization.py file was applied to the data used in https://link.springer.com/article/10.1007/s42943-021-00026-8 to get a sense of what trends we could expect in supply chain centrality, and ultimately train a network to predict firm importance based on graph position and characteristics. However, both our observations and the observations in the paper seem to suggest that no characteristics are correlated enough with centrality to warrant a predictive model.
