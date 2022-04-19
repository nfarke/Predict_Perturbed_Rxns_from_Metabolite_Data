import torch
import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, GraphConv, SAGEConv, ChebConv, SGConv, RGCNConv, TopKPooling,global_max_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import dropout_adj
import networkx as nx
from scipy.spatial.distance import pdist,squareform

import pdb

seed_num = 1
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
np.random.seed(seed_num)
random.seed(seed_num)
torch.backends.cudnn.deterministic= True
torch.cuda.empty_cache()


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SAGEConv(data.num_features, 50, normalize = True)
        #self.conv2 = SAGEConv(500, 500, normalize = True)
        #self.bn         = torch.nn.BatchNorm1d(100)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        #if x.shape[0] > 1:
        #   x = self.bn(x)
       # x = self.conv2(x, edge_index)
        return x

def cosine_sim(out):
    pred               = out.to(torch.double)
        
    a_norm             = (pred / pred.norm(dim = 1)[:, None])
    b_norm             = (pred / pred.norm(dim = 1)[:, None])
    cosine_sim_mat     = torch.mm(a_norm, b_norm.transpose(0,1)) #cosine similarity
    cosine_dist_mat    = 1 - cosine_sim_mat   
    
    #normalize so values lie between 0 and 1
    cosine_dist_mat = cosine_dist_mat/torch.max(cosine_dist_mat)
    return cosine_dist_mat

def graph_distance(N):
    
    N = (N != 0)*1
    Nnew = np.matmul(np.transpose(N),N)
    Nnew = (Nnew != 0)*1
    G = nx.from_numpy_matrix(Nnew)
    
    graph_dist = get_distances(G)
    
    #normalize so values lie between 0 and 1
    graph_dist = graph_dist/np.max(graph_dist)
    
    return graph_dist

def get_distances(G):
    
    path_length = nx.all_pairs_shortest_path_length(G)
    distances  = np.zeros((len(G), len(G)))
    for u,p in path_length:
        for v,d in p.items():
            distances[u][v]=d
    return distances


edges      = np.array(pd.read_excel (r'Graph2.xlsx','Edges'))
features   = pd.read_excel (r'Graph2.xlsx','Features').to_numpy() #NewSS
N   = pd.read_excel (r'Graph2.xlsx','stoichiometry').to_numpy()

edges = np.transpose(edges)

graph_dist = graph_distance(N)
edge_index = torch.tensor(edges)
edge_index.type(torch.LongTensor)
x           = torch.tensor(features, dtype = torch.float) 
graph_dist  = torch.tensor(graph_dist,dtype = torch.float)
data       = Data(x=x, edge_index= edge_index, graph_dist = graph_dist)
print('is the graph directed?:  '+str(data.is_directed()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)


Results = {'epoch':[], 'loss':[], 'out':[]}

model.train()
for epoch in range(800):
    print(epoch)
    optimizer.zero_grad()
    out = model(data)
    out1 = cosine_sim(out)
    #loss = F.smooth_l1_loss(out1.float(),data.graph_dist)
    loss = F.mse_loss(out1.float(), data.graph_dist)
    
    
    Results['epoch'].append(epoch)
    Results['loss'].append(np.array(loss.detach().cpu()))
    Results['out'].append(out.detach().cpu())
    
    loss.backward()
    optimizer.step()
    
    
plt.figure(0)
plt.plot(Results['epoch'],Results['loss'])
plt.xlabel('epochs')
plt.ylabel('smooth_l1_loss')
plt.show()  
    
out = out.detach().cpu()
cosine = squareform(pdist(out,'cosine'))     
plt.figure(1)
plt.scatter(graph_dist[:,:], cosine[:,:])
plt.xlabel('graph distance')
plt.ylabel('cosine distance of the Embedding')
plt.show()    


xval = np.transpose(np.reshape(graph_dist,(len(graph_dist)*len(graph_dist),1)))
yval = np.transpose(np.reshape(cosine,((len(cosine)*len(cosine)),1)))
corr = np.corrcoef(xval,yval)


    
    
    
    
    
    