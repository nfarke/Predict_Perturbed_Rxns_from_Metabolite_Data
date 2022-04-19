import numpy as np
import torch
import pandas as pd
from numpy import random
from torch.utils.data import Dataset, DataLoader
import pdb
import torch.nn.functional as F
from  scipy.spatial.distance import cosine as cos_dist
import time
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist,squareform
from sklearn.metrics import mean_squared_error as mse
from sklearn.decomposition import PCA
import networkx as nx
import itertools
from sklearn.preprocessing import MinMaxScaler as mms
from scipy.spatial.distance import cdist
from sklearn.feature_selection import VarianceThreshold as VT
import matplotlib.pyplot as plt
import itertools
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler as SS

class MetData(Dataset):
    # Constructor
    def __init__(self, X,Y,G,R,device):
        self.x = torch.tensor(X).to(device)
        self.y = torch.tensor(Y).to(device)
        self.G_id = G
        self.R_id = R
        self.len = self.x.shape[0]
        self.num_features = self.x.shape[1]
    # Getter
    def __getitem__(self, index):    
        return self.x[index], self.y[index], self.G_id[index], self.R_id[index]
    # Get length
    def __len__(self):
        return self.len
    
    def __num_features__(self):
        return self.num_features
        
        
class Net(torch.nn.Module):
    def __init__(self, p=0, input_size=1, output_size=1):
        super(Net,self).__init__()
        self.p          = p
        self.drop1      = torch.nn.Dropout(p = p)
        self.drop2      = torch.nn.Dropout(p = p)
        self.linear1    = torch.nn.Linear(input_size,int(output_size),bias = True)
        self.linear2    = torch.nn.Linear(int(output_size),int(output_size))
        
        self.bn         = torch.nn.BatchNorm1d(int(output_size))
        
    def forward(self, data):
        
        x = data
        x = self.linear1(x.float()) 
        if x.shape[0] > 1:
           x = self.bn(x)
        x = self.drop1(x)
        x = F.leaky_relu(x)
        #x = torch.sigmoid(x)
 
        x = self.linear2(x.float())
        return x


def make_some_plots(TrainAccLoss, TestAccLoss, cluster_hist, all_dist_test, epochs,split, distance_data,cluster_euc1, cluster_seuc1, cluster_cblock1, cluster_canb1):
    
    chunkids = [0]
    for kk in range(split-1):
        chunksize = np.round(epochs*(kk+1))
        chunkids.append(chunksize)
    
    chunkids.append(epochs*split)


    for k in range(split-1):
        plt.figure(11)
        plt.plot(TrainAccLoss['epoch'][chunkids[k]:chunkids[k+1]],TrainAccLoss['loss'][chunkids[k]:chunkids[k+1]],c='blue', label = 'train loss')
        plt.show()
        plt.plot(TestAccLoss['epoch'][chunkids[k]:chunkids[k+1]],TestAccLoss['loss'][chunkids[k]:chunkids[k+1]],c='orange', label = 'test loss')#
        plt.show()

        plt.figure(12)
        plt.plot(TrainAccLoss['epoch'][chunkids[k]:chunkids[k+1]],TrainAccLoss['metric'][chunkids[k]:chunkids[k+1]],c='blue', label = 'train metric')
        plt.show()
        plt.plot(TestAccLoss['epoch'][chunkids[k]:chunkids[k+1]],TestAccLoss['metric'][chunkids[k]:chunkids[k+1]],c='orange', label = 'test_metric')
        plt.show()

        plt.figure(13)
        plt.plot(TrainAccLoss['epoch'][chunkids[k]:chunkids[k+1]],TrainAccLoss['ACC'][chunkids[k]:chunkids[k+1]],c='blue', label = 'mean_cosine_similarity_train')
        plt.show()
        plt.plot(TestAccLoss['epoch'][chunkids[k]:chunkids[k+1]],TestAccLoss['ACC'][chunkids[k]:chunkids[k+1]],c='orange', label = 'mean_cosine_similarity_test')
        plt.show()    
    
    plt.figure(11)
    plt.xlabel('epochs')
    plt.ylabel('smooth_l1_loss')
    plt.legend(['train','test'])

    plt.figure(12)
    plt.xlabel('Epochs')
    plt.ylabel('Average Path Length Difference')
    plt.legend(['train','test'])
     
    plt.figure(13)
    plt.xlabel('Epochs')
    plt.ylabel('average cosine similarity')
    plt.legend(['train','test'])
    
    
    plt.figure(14)
    plt.hist([cluster_hist, all_dist_test], bins = 30,label = ['clustering','Neural Network'], align = 'mid')
    #plt.hist(all_dist_test, bins = 40,label = 'Neural Network')
    plt.xlabel('Path Length Difference')
    plt.ylabel('Counts')
    plt.legend()
    plt.show()
    
    plt.figure(15)
    plt.hist(distance_data['rxn_pred_id'],621)
    plt.ylabel('counts')
    plt.xlabel('rxn_ID')
    plt.legend()
    plt.show()
    
   
    import seaborn as sns
    plt.figure(16)
    
    sns.displot(np.array(cluster_hist), bins = 50,color = 'red', label = 'cosine')
    plt.axvline(np.mean(cluster_hist), color='red', linestyle='dashed', linewidth=1)
    sns.displot(np.array(cluster_euc1), bins = 50,color = 'blue', label = 'euclidean' )
    plt.axvline(np.mean(cluster_euc1), color='blue', linestyle='dashed', linewidth=1)

    sns.displot(np.array(cluster_seuc1), bins = 50,color = 'orange',label = 'seuclidean' )#
    plt.axvline(np.mean(cluster_seuc1), color='orange', linestyle='dashed', linewidth=1)

    sns.displot(np.array(cluster_cblock1), bins = 50,color = 'green', label = 'cityblock' )
    plt.axvline(np.mean(cluster_cblock1), color='green', linestyle='dashed', linewidth=1)

    sns.displot(np.array(cluster_canb1), bins = 50, color = 'black', label = 'canberra' )
    plt.axvline(np.mean(cluster_canb1), color='black', linestyle='dashed', linewidth=1)

    plt.xlabel('Path Length Difference')
    plt.ylabel('Density')
    plt.legend()


#----------------------------------------------------------------------------------------------------------------------------------
def plot_embeddings(embeddings_2d, rxn_list_unique, cluster_hist1, cluster_rxns1, all_dist_test1, all_test_rxns1, distance_data,chunk):
    rxn_selection = np.array(cluster_rxns1)
    test_rxn_names      = np.intersect1d( rxn_list_unique, rxn_selection, return_indices = bool)[0]
    test_rxn_ids        = np.intersect1d(rxn_list_unique, rxn_selection, return_indices = bool)[1]
    
    
    train_idx =  int(distance_data['train_id'][0].sum())
    train_rxn = np.unique(distance_data['rxn_target'][0][:train_idx])
    train_rxn_ids = np.intersect1d(rxn_list_unique, train_rxn, return_indices = bool)[1]
    
    x = []
    x1 = []
    distdiff = []
    for k,val in enumerate(all_dist_test1):
        x.append((val <= cluster_hist1[k])*1) # in which cases is the NN better than or as good as clustering
        x1.append((val < cluster_hist1[k])*1) # in which cases is the NN better
        distdiff.append(val - cluster_hist1[k])


    distdiff = np.array(distdiff)
    loc_id = np.where(distdiff > 2)
    
    
    all_test_rxns1 = np.array(all_test_rxns1)
    bad_rxns = all_test_rxns1[loc_id[0]]
    bad_dists = distdiff[loc_id[0]]
    
    bad_rxn_ids = np.intersect1d( rxn_list_unique, bad_rxns, return_indices = bool)[1]
    
    
    plt.figure(chunk)
    plt.scatter(embeddings_2d[:,0],embeddings_2d[:,1], alpha = 0.01)
    plt.scatter(embeddings_2d[train_rxn_ids,0],embeddings_2d[train_rxn_ids,1], alpha = 1, c = 'blue')
    plt.scatter(embeddings_2d[test_rxn_ids,0],embeddings_2d[test_rxn_ids,1], alpha = 1, c = 'orange')
    plt.show()
    plt.scatter(embeddings_2d[bad_rxn_ids,0],embeddings_2d[bad_rxn_ids,1], alpha = 1, c = 'orange', edgecolors = 'black')
# =============================================================================
        


def train_test_split(rxn_list, gene_list, embeddings, features, split):
    genes = list(set(gene_list))
    embeddings = np.transpose(embeddings)
    features = np.transpose(features)
    
    ids =     np.array([x for x in range(0,len(genes))])
    random.shuffle(ids)
    
    train_ids = ids[:int(len(genes)*split)]
    test_ids = ids[int(len(genes)*split):]
    
    train_genes     =  [genes[i] for i in train_ids]
    test_genes      =  [genes[i] for i in test_ids]
    
    gene_list = pd.DataFrame(gene_list)
    rxn_list = pd.DataFrame(rxn_list)
    train_genes2        = gene_list.loc[gene_list[0].isin(train_genes)][0].to_list()
    test_genes2         = gene_list.loc[gene_list[0].isin(test_genes)][0].to_list()
    train_embeddings    = embeddings[gene_list[0].isin(train_genes),:]
    test_embeddings     = embeddings[gene_list[0].isin(test_genes),:]
    train_features      = features[gene_list[0].isin(train_genes),:]
    test_features       = features[gene_list[0].isin(test_genes),:]
    train_rxn_list      = rxn_list.loc[gene_list[0].isin(train_genes)][0].to_list()
    test_rxn_list       = rxn_list.loc[gene_list[0].isin(test_genes)][0].to_list()
    
    train_x = pd.concat([pd.DataFrame(train_genes2, columns = ['genes']), pd.DataFrame(train_rxn_list, columns = ['reactions']), pd.DataFrame(train_features)], axis=1)
    train_y = pd.concat([pd.DataFrame(train_genes2, columns = ['genes']), pd.DataFrame(train_rxn_list, columns = ['reactions']),pd.DataFrame(train_embeddings)], axis=1)
    
    test_x = pd.concat([pd.DataFrame(test_genes2, columns = ['genes']), pd.DataFrame(test_rxn_list, columns = ['reactions']),pd.DataFrame(test_features)], axis=1)
    test_y = pd.concat([pd.DataFrame(test_genes2, columns = ['genes']), pd.DataFrame(test_rxn_list, columns = ['reactions']),pd.DataFrame(test_embeddings)], axis=1)
    
    return train_x, train_y, test_x, test_y

#===================================================================================================
def train_test_split_CV(rxn_list, gene_list, embeddings, features, split,chunkids,chunk,rxn,ids):
    
    embeddings = np.transpose(embeddings)
    features = np.transpose(features)
    
    test_ids = ids[int(chunkids[chunk]):int(chunkids[chunk+1])]
    train_ids = np.array(list(set(ids).symmetric_difference(test_ids)))
    test_ids = np.array(test_ids)
    
    train_rxns     =  [rxn[i] for i in train_ids]
    test_rxns      =  [rxn[i] for i in test_ids]
    
    gene_list = pd.DataFrame(gene_list)
    rxn_list = pd.DataFrame(rxn_list)

    train_rxn_list      = rxn_list.loc[rxn_list[0].isin(train_rxns)][0].to_list()
    test_rxn_list       = rxn_list.loc[rxn_list[0].isin(test_rxns)][0].to_list()
    
    train_genes2        = gene_list.loc[rxn_list[0].isin(train_rxns)][0].to_list()
    test_genes2         = gene_list.loc[rxn_list[0].isin(test_rxns)][0].to_list()
    train_embeddings    = embeddings[rxn_list[0].isin(train_rxns),:]
    test_embeddings     = embeddings[rxn_list[0].isin(test_rxns),:]
    train_features      = features[rxn_list[0].isin(train_rxns),:]
    test_features       = features[rxn_list[0].isin(test_rxns),:]

    train_x = pd.concat([pd.DataFrame(train_genes2, columns = ['genes']), pd.DataFrame(train_rxn_list, columns = ['reactions']), pd.DataFrame(train_features)], axis=1)
    train_y = pd.concat([pd.DataFrame(train_genes2, columns = ['genes']), pd.DataFrame(train_rxn_list, columns = ['reactions']),pd.DataFrame(train_embeddings)], axis=1)
    
    test_x = pd.concat([pd.DataFrame(test_genes2, columns = ['genes']), pd.DataFrame(test_rxn_list, columns = ['reactions']),pd.DataFrame(test_features)], axis=1)
    test_y = pd.concat([pd.DataFrame(test_genes2, columns = ['genes']), pd.DataFrame(test_rxn_list, columns = ['reactions']),pd.DataFrame(test_embeddings)], axis=1)
    
    return train_x, train_y, test_x, test_y


#==========================================================================================================================================================
def load_and_rearrange_data_old():
    embedding   = np.array(pd.read_csv('Embedding.csv'))
    RxG         = np.array(pd.read_csv('RXG.csv'))
    rxns        = np.array(pd.read_csv('rxns.csv'))
    features    = np.array(pd.read_csv('data_features.csv'))
    genes       = pd.read_csv('genes.csv').values.tolist()
    
    #rxn_per_gene         = np.array((pd.read_excel (r'less_mets/rxns_per_gene.xlsx')))
    #pdb.set_trace()
    
    scaler = SS()
    scaler.fit(np.transpose(features))
    features = np.transpose(scaler.transform(np.transpose(features)))
    
    
    
    N = np.array(pd.read_excel ('N.xlsx'))    
    rxn_list    = []
    gene_list   = []
    emb_new     = np.zeros((embedding.shape[0],1))
    feat_new    = np.zeros((features.shape[0],1))
    #for each gene extract reactions
    for k,col in enumerate(np.transpose(RxG)):
        idx     = np.where(col == 1)[0]
        rxn_emb = embedding[:,idx]
        rxn     = rxns[idx]
        gene    = np.tile(genes[k],rxn.shape[0])
        gene_id_tiled = np.tile(k, rxn.shape[0])
        feat    = features[:,gene_id_tiled]
        for kk in range(idx.shape[0]):
            rxn_list.append(str(rxn[kk][0]))
            gene_list.append(str(gene[0]))
        
        emb_new     = np.concatenate((emb_new,rxn_emb), axis = 1)
        feat_new    = np.concatenate((feat_new, feat), axis = 1)
 
    emb_new = emb_new[:,1:]
    feat_new = feat_new[:,1:]
    
    ### set up graph  #cols and rows correspond to rxns
    N = np.abs(np.array(N))
    Nnew = (np.matmul(np.transpose(N),N) > 0)*1
    G = nx.from_numpy_matrix(Nnew)
    
    rxn_list, gene_list, emb_new, feat_new, clusterx = rearrange(G,rxn_list, gene_list, emb_new, feat_new,rxns,cutoff = 1)
    
    return rxn_list, gene_list, emb_new, feat_new, rxns, embedding, clusterx

#====================================================================================================================================== 
def load_and_rearrange_data():
    embedding   = np.array(pd.read_csv('data_500d\Embedding.csv'))
    RxG         = np.array(pd.read_csv('data\RXG.csv'))
    df          = (pd.read_excel (r'data\N.xlsx','rxns'))
    rxns        = np.array(df.values.tolist())
    features    = np.array(pd.read_csv('data\data_features.csv'))
    genes       = pd.read_csv('data\genes.csv').values.tolist()  
    
    scaler = SS()
    scaler.fit(np.transpose(features))
    features = np.transpose(scaler.transform(np.transpose(features)))
    
    #decompose gene features into reaction features
    RxG_new = np.empty([RxG.shape[0],1])
    lin_dep_row_ids = [0]
    for j,col in enumerate(np.transpose(RxG)):
        if j == 0:
           col = np.reshape(col,(-1,1))
           RxG_new = np.concatenate((RxG_new, col), axis = 1)
           RxG_new = np.delete(RxG_new,0,1)
           rankx = np.linalg.matrix_rank(RxG_new)
        else:
           col = np.reshape(col,(-1,1))
           ranktest = np.linalg.matrix_rank(np.concatenate((RxG_new, col), axis = 1))
           if ranktest > rankx:
               RxG_new = np.concatenate((RxG_new, col), axis = 1)
               rankx = np.linalg.matrix_rank(RxG_new)
               lin_dep_row_ids.append(j)
    
    features_new = features[:,lin_dep_row_ids]    
    mask = RxG_new == 0
    rows = np.flatnonzero((~mask).sum(axis = 1)) #rxn ids
    RxG_new1 = RxG_new[rows,:] #new RxG
    RxG_inv = np.linalg.pinv(RxG_new1)
    rxn_feats = np.dot(features_new,RxG_inv) # new reaction features
    rxns_new = rxns[rows] #new reactions
    genes = np.array(genes)
    genes_new = genes[lin_dep_row_ids]  #new genes
    embedding_new = embedding[:,rows] #new embedding
    
    rxn_list    = []
    gene_list   = []    
    emb_new     = np.zeros((embedding_new.shape[0],1))
    feat_new    = np.zeros((rxn_feats.shape[0],1))

    for k,col in enumerate(np.transpose(RxG_new1)):
        idx     = np.where(col == 1)[0]
        rxn_emb = embedding_new[:,idx]
        rxn     = rxns_new[idx]
        gene    = np.tile(genes_new[k],rxn.shape[0])
        gene_id_tiled = np.tile(k, rxn.shape[0])
        feat    = rxn_feats[:,gene_id_tiled]
        for kk in range(idx.shape[0]):
            rxn_list.append(str(rxn[kk][0]))
            gene_list.append(str(gene[0]))
        
        emb_new     = np.concatenate((emb_new,rxn_emb), axis = 1)
        feat_new    = np.concatenate((feat_new, feat), axis = 1)

    emb_new = emb_new[:,1:]
    feat_new = feat_new[:,1:]
    
    return rxn_list, gene_list, emb_new, feat_new, rxns, embedding

#==============================================================================================
def rearrange_alternative(G,rxn_list, gene_list, emb_new, feat_new,rxns,RxG, genes_orig):

    #here we perform matrix inversion to calculate the reaction features from the gene features of some genes
    RxG_new = np.empty([RxG.shape[0],1])
    lin_dep_row_ids = [0]
    for j,col in enumerate(np.transpose(RxG)):
        if j == 0:
           col = np.reshape(col,(-1,1))
           RxG_new = np.concatenate((RxG_new, col), axis = 1)
           RxG_new = np.delete(RxG_new,0,1)
           rankx = np.linalg.matrix_rank(RxG_new)
        else:
           col = np.reshape(col,(-1,1))
           ranktest = np.linalg.matrix_rank(np.concatenate((RxG_new, col), axis = 1))
           if ranktest > rankx:
               RxG_new = np.concatenate((RxG_new, col), axis = 1)
               rankx = np.linalg.matrix_rank(RxG_new)
               lin_dep_row_ids.append(j)
    
    feat_new_trunc = feat_new[:,lin_dep_row_ids]    
    mask = RxG_new == 0
    rows = np.flatnonzero((~mask).sum(axis = 1)) #rxn ids
    RxG_new1 = RxG_new[rows,:]
    RxG_inv = np.linalg.pinv(RxG_new1)
    rxn_feats = np.dot(feat_new_trunc,RxG_inv) # new reaction features
    
    #get corresponding reactions
    rxn_new = rxns[rows]
    #get corresponding genes via the new RxG
    return gene_list, gene_list

#=========================================================================================================
def rearrange_part2(gene_dist, gene_list, cutoff, feat_new, emb_new, unique_gene_list, rxn_list):
    
    #for each gene one reaction is chosen (the first one listed)
    select_id = gene_dist < cutoff
    gene_list_new = unique_gene_list[select_id]
    gene_list_ids = np.array(np.intersect1d(gene_list,gene_list_new, return_indices = bool)[1])
    rxn_list = np.array(rxn_list)
    rxn_list1 = rxn_list[gene_list_ids]
    rxn_list1.tolist()
    
    gene_list = np.array(gene_list)
    gene_list1 = gene_list[gene_list_ids].tolist()
    feat_new1 = feat_new[:,gene_list_ids]
    emb_new1 = emb_new[:,gene_list_ids]  
    
    return gene_list, gene_list1, emb_new1, feat_new1, rxn_list1


def rearrange(G,rxn_list, gene_list, emb_new, feat_new,rxns,cutoff):
    gene_dist = []
    gene_len  = []
    unique_gene_list = np.unique(gene_list)
    
    for k,gene in enumerate(unique_gene_list):
        indexes = [i for i,x in enumerate(gene_list) if x == gene]
        indexes1 = []
        for ids in indexes:
            rx = rxn_list[ids]
            indexes1.append(np.where(rxns==rx))
        distx = []
        if len(indexes1) > 1:
           for kk in itertools.permutations(indexes1, 2):
               
               distance = (nx.shortest_path_length(G,kk[0][0][0],kk[1][0][0]))  
               distx.append((distance+1)/2)
               
           gene_dist.append(np.mean(distx[1:int(len(distx)/2+1)]))
           gene_len.append(len(distx))
        else:
             gene_dist.append(0)
             gene_len.append(1)
    
    gene_dist = np.array(gene_dist)
    gene_list, gene_list1, emb_new1, feat_new1, rxn_list1 = rearrange_part2(gene_dist, gene_list, cutoff, feat_new, emb_new, unique_gene_list, rxn_list)  
    
    clusterx = [] #get node degree

    for kkk in range(len(rxns)):
        clusterx.append(nx.clustering(G,(kkk)))
    
    return rxn_list1, gene_list1, emb_new1, feat_new1, clusterx



def pred_evaluation (pred, rxn_target, embeddings, rxns, dist):
    
    
    tic = time.clock()
    
    pred = pred.detach().numpy()
    rxns = rxns.tolist()
    rxns = [item for sublist in rxns for item in sublist]
    rxn_target = list(rxn_target)
    

    rxn_pred=[]
    for p in pred:
        g=([cos_dist(p,e) for e in np.transpose(embeddings)])
        rxn_pred.append(rxns[np.argmin(g)])
        
    toc = time.clock()
    print('time for distance: ' + str(toc-tic))
        
    tic = time.clock()
    rxn_pred_id = [rxns.index(r) for r in rxn_pred]
    rxn_target_id = [rxns.index(r) for r in rxn_target]
    toc = time.clock()
    print('time for lookup: ' + str(toc-tic))
    
    
    dists = [dist[x[0], x[1]] for x in zip (rxn_pred_id, rxn_target_id)]
    mea_dist = np.mean(dists)
    std_dist = np.std(dists)
    med_dist = np.median(dists)
    
    return mea_dist, med_dist, std_dist






def pred_evaluation2 (pred, rxn_target, embeddings, rxns, dist, val ):
    
    pred = pred.detach().to(torch.double)
    embeddings = embeddings.transpose(0,1).to(torch.double)
    
    
    #euclidean distance
    #x_norm = (pred**2).sum(1).view(-1,1)
    #y_norm = (embeddings**2).sum(1).view(1,-1)
    #res = x_norm + y_norm -2.0*torch.mm(pred, embeddings.transpose(0,1))
    #rxn_pred_id  = torch.argmax(res, dim = 1)
    
    #euclidean distance row wise
    #rxn_pred_id = torch.tensor([])
    #for i,row in enumerate(pred):
        #r1 = row.expand_as(embeddings)
        #sq_dist= torch.sum((r1-embeddings)**2,1)
    
        
    #cosine sim
    a_norm = (pred / pred.norm(dim = 1)[:, None])
    b_norm = embeddings / embeddings.norm(dim = 1)[:, None]
    res    = torch.mm(a_norm, b_norm.transpose(0,1)) #cosine similarity
    rxn_pred_id  = torch.argmax(res, dim = 1) #old
    mean_sim = torch.mean(torch.max(res,dim = 1)[0]).cpu()

    #if val == 0:
     #  topk = torch.topk(res,res.shape[0], dim = 1, largest = True)
     #  mean_sim = torch.mean(topk[0][:,0]) #1 = take the second biggest elements
     #  rxn_pred_id = topk[1][:,0] #1 = take the IDs of the second biggest elements
   # else:
     #  topk = torch.topk(res,res.shape[0], dim = 1, largest = True)
      # mean_sim = torch.mean(topk[0][:,1]) 
      # rxn_pred_id = topk[1][:,1]
    rxns = rxns.tolist()
    rxns = [item for sublist in rxns for item in sublist]
    rxn_target_id = [rxns.index(r) for r in rxn_target]
    
    #control
    #pred = pred.cpu()
    #emb = embeddings.cpu()
    #pred1 = pred[0,:].to(torch.double)
    #emb1 = emb[0,:]
    #dis = 1-cos_dist(np.array(pred1),np.array(emb1))
    
    rxn = np.array(rxns)
    rxn_out =rxn[rxn_target_id]
    rxn_pred = rxn[rxn_pred_id.cpu()]
       
    dists = dist[rxn_pred_id.cpu(),rxn_target_id]   
    mea_dist = np.mean(dists)
    
    return mea_dist,dists,mean_sim, rxn_out, rxn_pred, rxn_pred_id, rxn_target_id




def clustering(features, genes,rxn_list, rxn_target, embeddings, rxns, dist,gene_rxn_assosiation, distfun):
    gene_dists = squareform(pdist(np.transpose(features),distfun))
    gene_targets = []
    targets_ids = []
    gene_start = []
    start_ids = []
    rxn_targets = []
    rxn_start = []
    all_target_reactions = []  
    rxns = rxns.tolist()
    rxns = [item for sublist in rxns for item in sublist]    
    
    for jj in range(gene_dists.shape[0]):
        row = gene_dists[jj,:]
        out = np.where(row == np.min(row[np.nonzero(row)]))[0]
        out = tuple(out) #gets most similar genes
        gene_startx = genes[jj]
        for genesx in out:
            gene_targets.append(genes[genesx])
            targets_ids.append(genesx)
            gene_start.append(gene_startx)
            start_ids.append(jj)
            
            rxn_targets.append(rxn_list[genesx])
            rxn_start.append(rxn_list[jj])
    
    start_rxns = np.array(rxn_start)
    targets_rxns = np.array(rxn_targets)
    dist_final = []
    for kk in range(len(gene_rxn_assosiation.genes)):  # here we use the test set
        gene = gene_rxn_assosiation.genes[kk]  
        idsx = np.where(np.array(gene_start) == gene)[0]
        start_reactions = np.unique(start_rxns[idsx])
        target_reactions = np.unique(targets_rxns[idsx])
        all_target_reactions.append(target_reactions)
        dist_list = []
        for rx in start_reactions:
            for rx2 in target_reactions:
                start_rxn_id = np.where(np.array(rxns)==rx)[0]
                target_rxn_id = np.where(np.array(rxns)==rx2)[0]
                dist_list.append(dist[start_rxn_id, target_rxn_id][0])
        
        dist_final.append(min(dist_list))        
    
    mea_dist = np.mean(dist_final)
    return dist_final, mea_dist, gene_rxn_assosiation.reactions, list(all_target_reactions)

def clustering_old(features, genes,rxn_list, rxn_target, embeddings, rxns, dist,gene_rxn_assosiation,distfun):
    gene_dists = squareform(pdist(np.transpose(features),distfun))
      
    genes2 = []
    gene2_id = []
    for jj in range(gene_dists.shape[0]):
        row = gene_dists[jj,:]
        out = np.where(row == np.min(row[np.nonzero(row)]))[0][0]
        genes2.append(genes[out])
        gene2_id.append(out)
    
    
    
    test_genes = gene_rxn_assosiation.genes
    intersecting_test_ids = np.where(np.in1d(genes,test_genes))[0]
    
    
    start_gene = []
    target_gene = []
    start_rxn = []
    target_rxn = []
    start_rxn_id = []
    for val in intersecting_test_ids:
        start_gene.append(genes[val])
        target_gene.append(genes2[val])
        start_rxn.append(rxn_list[val])
        start_rxn_id.append(val)
        
    target_rxn = []
    target_rxn_id = []
    for gene in target_gene:  
        idx = np.where(np.array(genes) == gene)[0][0]
        target_rxn.append(rxn_list[idx])
        target_rxn_id.append(idx)
    

    rxns = rxns.tolist()
    rxns = [item for sublist in rxns for item in sublist]    
    target_rxn_id = [rxns.index(r) for r in target_rxn]
    start_rxn_id = [rxns.index(r) for r in start_rxn]
    
    #distance for the test set
    dists = dist[start_rxn_id, target_rxn_id]
    mea_dist = np.mean(dists)
       
    return dists, mea_dist





def NNet(epochs,dataloader,optimizer,model, TrainAccLoss, TestAccLoss, test_dataset,embeddings_unique, rxn_list_unique, dist):
    for epoch in range(epochs):
        running_loss = 0 #accumulates loss of each batch
        running_metri = []
        model.train()
        for ii,(x, target, gen_id, reaction_id) in enumerate(dataloader):
            optimizer.zero_grad()
            pred = model(x)
            #loss = F.l1_loss(pred.float(),target.float())
            #loss = F.mse_loss(pred.float(),target.float())
            loss = F.smooth_l1_loss(pred.float(),target.float())
            loss.backward()
            optimizer.step()
            running_loss +=loss.item()
            train_metri, all_dist_train,ACC_train, rxn_train_target, rxn_train_pred, rxn_pred_id_train, rxn_tar_id_train = pred_evaluation2 (pred, reaction_id, embeddings_unique, rxn_list_unique, dist, val = 0)
            running_metri.append(train_metri)
    
        TrainAccLoss['loss'].append(running_loss)  #final loss for each epoch
        TrainAccLoss['metric'].append(np.mean(running_metri))  #final loss for each epoch
        TrainAccLoss['epoch'].append(epoch)  #final loss for each epoch
        TrainAccLoss['ACC'].append(ACC_train.item()) 
        
        test_target_emb = test_dataset.y
        test_target_rxn = test_dataset.R_id

        model.eval()
        test_pred       = model(test_dataset.x)
        test_loss       = F.smooth_l1_loss(test_pred.float(), test_target_emb.float())
        #test_loss       = F.mse_loss(test_pred.float(), test_target_emb.float())
        #test_loss        = F.l1_loss(pred.float(),target.float())
        
        test_metri, all_dist_test,ACC_test, rxn_test_target,rxn_test_pred, rxn_pred_id_test, rxn_tar_id_test      = pred_evaluation2 (test_pred, test_target_rxn, embeddings_unique, rxn_list_unique, dist, val = 1)
        TestAccLoss['loss'].append(test_loss.item())  #final loss for each epoch
        TestAccLoss['metric'].append(test_metri)  #final loss for each epoch
        TestAccLoss['epoch'].append(epoch)
        TestAccLoss['ACC'].append(ACC_test.item())
    return test_pred,TrainAccLoss, TestAccLoss,train_metri, all_dist_train,ACC_train, rxn_train_target, rxn_train_pred, rxn_pred_id_train, rxn_tar_id_train, test_metri, all_dist_test,ACC_test, rxn_test_target,rxn_test_pred, rxn_pred_id_test, rxn_tar_id_test


