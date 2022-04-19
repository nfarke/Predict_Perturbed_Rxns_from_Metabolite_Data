# Predict_Perturbed_Rxns_from_Metabolite_Data

The idea behind this project was to train a model that predicts perturbed locations in metabolism from metabolome data.
As dataset we take amino acid profiles from a dataset from Ralser lab

The idea was pretty simple:
- Embedd a graph in 2D space (node2vec or graphsage in this iteration). As objective function we preserved the distance in the original graph.
- We then used amino acid data and trained a neural network to predict the 2D-coordinates of the embedding. 

- We then compared our results against conventional clustering approaches and several different distance functions
