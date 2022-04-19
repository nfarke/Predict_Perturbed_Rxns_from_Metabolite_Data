import numpy as np
import torch
import pandas as pd
#requires the following files -
#RxR_dist.xlsx
#Embedding.xlsx
#RXG
#rxns
#data_features
#genes



from numpy import random
from MAIN_utils_NF import load_and_rearrange_data_old, MetData, Net, train_test_split_CV,clustering, make_some_plots, NNet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,squareform
from sklearn.manifold import TSNE


seed_num = 6
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
np.random.seed(seed_num)
random.seed(seed_num)
torch.backends.cudnn.deterministic= True

# =============================================================================
# hyperparameters
# =============================================================================
lr                  = 0.0005
weight_decay        = 5e-4
epochs              = 100
dropout             = 0.2
split               = 10
batches              = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# data prep
# =============================================================================

rxn_list, gene_list, embeddings, features, rxn_list_unique, embeddings_unique, clusterx   = load_and_rearrange_data_old()
dist                                                                            = np.array(pd.read_excel('RxR_dist.xlsx'))

#tsne = TSNE(n_components=2, random_state=7, perplexity=100, metric = 'precomputed')
#cosine = squareform(pdist(np.transpose(embeddings_unique),'cosine'))   
#embeddings_2d = tsne.fit_transform(cosine)  

embeddings_unique = torch.tensor(embeddings_unique).to(device)


rxn = list(set(rxn_list))
ids =     [x for x in range(0,len(rxn))]
random.shuffle(ids)

chunkids    = [0]
for kk in range(split-1):
    chunksize = np.round(len(rxn)*(1/split))*(kk+1)
    chunkids.append(chunksize)
chunkids.append(len(rxn))


TrainAccLoss = {'epoch':[], 'loss':[], 'metric': [],'ACC':[]}
TestAccLoss     = {'epoch':[], 'loss':[], 'metric': [],'ACC':[]}
cluster_hist1 = []
all_dist_test1 = []
cluster_rxns1 = []
all_test_rxns1 = []

cluster_euc1 = []
cluster_seuc1 = []
cluster_cblock1 = []
cluster_canb1 = []
cluster_rxn_preds1 = []

distance_data = {'rxn_pred':[], 'rxn_pred_id':[],'rxn_target':[],'rxn_target_id':[], 'distance':[], 'clusterx':[],'pred_emb':[],'test_emb':[]}

embedding_out = {'rxn_pred':[], 'rxn_pred_id':[]}

for chunk in range(split):
    print(chunk)
    train_x, train_y, test_x, test_y = train_test_split_CV(rxn_list, gene_list, embeddings, features, split,chunkids,chunk,rxn,ids)
    
    train_dataset  = MetData(np.array(train_x.iloc[:,2:]),  np.array(train_y.iloc[:,2:]),   list(train_x.genes),    list(train_x.reactions),device)
    test_dataset   = MetData(np.array(test_x.iloc[:,2:]),   np.array(test_y.iloc[:,2:]),    list(test_x.genes),     list(test_x.reactions),device)
    
    batch_size      = int(train_dataset.len/batches)
    dataloader      = DataLoader(train_dataset, batch_size, shuffle=True) 
    model = Net(p=dropout, input_size = train_dataset.num_features, output_size = embeddings.shape[0] ).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = lr, weight_decay = weight_decay)
    #optimizer = torch.optim.SGD(model.parameters(),lr = lr)

    test_target_rxn = test_dataset.R_id
    #test different distance functions
    cluster_euc, cluster_mea_dist, cluster_rxns, cluster_rxn_preds = clustering(features, gene_list,rxn_list,test_target_rxn, embeddings_unique, rxn_list_unique, dist,test_x, distfun = 'euclidean')
    cluster_seuc, cluster_mea_dist, cluster_rxns, cluster_rxn_preds  = clustering(features, gene_list,rxn_list,test_target_rxn, embeddings_unique, rxn_list_unique, dist,test_x, distfun = 'seuclidean')
    cluster_cblock, cluster_mea_dist, cluster_rxns, cluster_rxn_preds  = clustering(features, gene_list,rxn_list,test_target_rxn, embeddings_unique, rxn_list_unique, dist,test_x, distfun = 'cityblock')
    cluster_canb, cluster_mea_dist, cluster_rxns, cluster_rxn_preds  = clustering(features, gene_list,rxn_list,test_target_rxn, embeddings_unique, rxn_list_unique, dist,test_x, distfun = 'canberra')
    cluster_hist, cluster_mea_dist, cluster_rxns, cluster_rxn_preds  = clustering(features, gene_list,rxn_list,test_target_rxn, embeddings_unique, rxn_list_unique, dist,test_x, distfun = 'cosine')
    
    #train model
    pred_emb,TrainAccLoss, TestAccLoss,train_metri, all_dist_train,ACC_train, rxn_train_target, rxn_train_pred, rxn_pred_id_train, rxn_tar_id_train, test_metri, all_dist_test,ACC_test, rxn_test_target,rxn_test_pred, rxn_pred_id_test, rxn_tar_id_test = NNet(epochs,dataloader,optimizer,model, TrainAccLoss, TestAccLoss, test_dataset,embeddings_unique, rxn_list_unique, dist)
    
    if chunk == 0:
       distance_data['pred_emb'] = pred_emb.cpu().detach().numpy()
       distance_data['test_emb'] = test_dataset.y.cpu().detach().numpy()

    else: 
       distance_data['pred_emb'] = np.concatenate((distance_data['pred_emb'],pred_emb.cpu().detach().numpy()),axis = 0)
       distance_data['test_emb'] = np.concatenate((distance_data['test_emb'],test_dataset.y.cpu().detach().numpy()),axis = 0)
    
    #store results
    distance_data['rxn_pred'].extend(list(rxn_test_pred))
    distance_data['rxn_pred_id'].extend(np.array(rxn_pred_id_test.cpu()))
    distance_data['rxn_target'].extend(list(rxn_test_target))
    distance_data['rxn_target_id'].extend(rxn_tar_id_test)
    distance_data['distance'].extend(all_dist_test)
    
    #clustering and NN results (to ensure consistent order)
    cluster_hist1.extend(cluster_hist)
    cluster_euc1.extend(cluster_euc)
    cluster_seuc1.extend(cluster_seuc)
    cluster_cblock1.extend(cluster_cblock)
    cluster_canb1.extend(cluster_canb)
    cluster_rxns1.extend(cluster_rxns)
    cluster_rxn_preds1.extend(cluster_rxn_preds)
    all_dist_test1.extend(all_dist_test)
    all_test_rxns1.extend(rxn_test_target)
    

#plot results
make_some_plots(TrainAccLoss, TestAccLoss, cluster_hist1, all_dist_test1,epochs, split, distance_data, cluster_euc1, cluster_seuc1, cluster_cblock1, cluster_canb1)

x = []
x1 = []
distdiff = []
for k,val in enumerate(all_dist_test1):
    x.append((val <= cluster_hist1[k])*1) # in which cases is the NN better than or as good as clustering
    x1.append((val < cluster_hist1[k])*1) # in which cases is the NN better
    distdiff.append(val - cluster_hist1[k])

plt.figure(10)
plt.hist(distdiff, bins = 25)    
plt.xlabel('distance difference NN and Clustering')
plt.ylabel('counts')

print('the NN is better/equal in '+ str(sum(x)/len(x)*100)+' %')    
print('the NN is better in '+ str(sum(x1)/len(x1)*100)+' %')    
print('average distance improvement: '+ str(np.mean(distdiff)))   

print('average distance using the NN:' + str(np.mean(all_dist_test1))) 
print('average distance using the Clustering:' + str(np.mean(cluster_hist1)))


combined_embedding = np.concatenate((np.transpose(embeddings_unique.cpu()),distance_data['pred_emb']), axis = 0)

tsne = TSNE(n_components=2, random_state=1, perplexity=0.5, metric = 'precomputed')
cosine = squareform(pdist(combined_embedding,'cosine'))   
embeddings_2d = tsne.fit_transform(cosine)  

plt.figure(3213)
plt.scatter(embeddings_2d[:embeddings_unique.shape[1],0],embeddings_2d[:embeddings_unique.shape[1],1],c='b', alpha = 0.3)
plt.scatter(embeddings_2d[embeddings_unique.shape[1]:,0],embeddings_2d[embeddings_unique.shape[1]:,1],c='orange', alpha = 0.3)

coordinates = []
tsne = TSNE(n_components=2, random_state=7, perplexity=100, metric = 'precomputed')






















