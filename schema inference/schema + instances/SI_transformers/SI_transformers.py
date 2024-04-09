from sklearn import preprocessing
#Set Seed
import random
import os 
import pandas as pd 
import numpy as np
import torch
import pandas as pd
import numpy as np
from time import time
import torch
from sklearn.model_selection import train_test_split
from statistics import mean, median, variance
from src.DeepClusteringAlgorithm.SDCN.data.pretrain import AE, LoadDataset, pretrain_ae
from src.DeepClusteringAlgorithm.SDCN.sdcn import load_data, train_sdcn
from src.DeepClusteringAlgorithm.SDCN.calcu_graph import construct_graph 
    
import torch

random.seed(555)
np.random.seed(555)
torch.manual_seed(555)

dim = 693  #693 for TabNet, 208 for TabTransformer, 768 for SBERT

vectors=pd.read_csv('src/DeepClusteringAlgorithm/SDCN/data/X.txt', header =None, sep = ' ')
vectors = vectors.to_numpy()

y=pd.read_csv('src/DeepClusteringAlgorithm/SDCN/data/labels.txt', sep=' ', header = None)
y= pd.DataFrame(y)
y = y.to_numpy()
y=y.flatten()
y


cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
pretrain_path = 'src/DeepClusteringAlgorithm/SDCN/' + 'data/{}.pkl'.format('X')
n_input = dim
dataset = load_data(vectors, y)
name = 'X'


# Choose deep clustering algorithm among : 'SDCN', 'IFDF', 'DCSS'

clustering_algorithm_name = 'SDCN' 

if clustering_algorithm_name == 'SDCN':

    
    # Preprocessing
    
    
    # Compute encoder model
    model = AE(
        n_enc_1=1000, #500 256
        n_dec_1=1000,
        n_input=dim, # 256 embedding_dimension
        n_z=100,).cuda()
    
    print(vectors)
    dataset = LoadDataset(vectors)
    # Hyperparameters
    n_clusters = 26 #number of class
    nb_epochs = 100
    k = 0 #  version(number) of graph generated for this dataset
    lr = 1e-3 # learning rate 1e-3
    n_z = 100

    # Compute graph
    construct_graph(vectors, y, 'ncos')

    # Compute pretrain on the dataset
    pretrain_ae(model, dataset, y, nb_epochs)
    
    # Parameters for model:
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    pretrain_path = 'src/DeepClusteringAlgorithm/SDCN/' + 'data/{}.pkl'.format('X')
    n_input = dim
    dataset = load_data(vectors, y)
    name = 'X'
    start = time()
    # Compute train on the dataset
    y_pred = train_sdcn(dataset, device, name, vectors, pretrain_path, n_clusters, n_input, nb_epochs, k, lr, n_z)
    end = time()
    print("Training took ",{end-start}," sec to run.")
    print(y_pred)

