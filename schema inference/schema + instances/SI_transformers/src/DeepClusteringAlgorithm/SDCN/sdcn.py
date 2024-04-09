from __future__ import print_function, division
import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import silhouette_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from src.DeepClusteringAlgorithm.SDCN.utils import load_data, load_graph
from src.DeepClusteringAlgorithm.SDCN.GNN import GNNLayer
from src.DeepClusteringAlgorithm.SDCN.evaluation import eva
from collections import Counter


# torch.cuda.set_device(1)

nb_dimension = 693 # 

class AE(nn.Module):

    def __init__(self, n_enc_1,  n_dec_1,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.z_layer = Linear(n_enc_1, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.x_bar_layer = Linear(n_dec_1, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        z = self.z_layer(enc_h1)

        dec_h1 = F.relu(self.dec_1(z))
        x_bar = self.x_bar_layer(dec_h1)

        return x_bar, enc_h1, z
# set seed
random.seed(555)
np.random.seed(555)
torch.manual_seed(555)

class SDCN(nn.Module):

    def __init__(self, n_enc_1,  n_dec_1,
                n_input, n_z, n_clusters,pretrain_path, v=1):
        super(SDCN, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_dec_1=n_dec_1,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(pretrain_path, map_location='cpu'))

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_4 = GNNLayer(n_enc_1, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra3, z = self.ae(x)
        
        sigma = 0.5

        # GCN Module
        h = self.gnn_1(x, adj)
        h = self.gnn_4((1-sigma)*h + sigma*tra3, adj)
        h = self.gnn_5((1-sigma)*h + sigma*z, adj, active=False)
        predict = F.softmax(h, dim=1)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

#def train_sdcn(dataset, device, pretrain_path, n_clusters, n_input, nb_epochs, ): #only : dataset
def train_sdcn(dataset, device, name, vectors, pretrain_path, n_clusters, n_input, nb_epochs, k, lr, n_z):
    model = SDCN(1000, 1000,
                n_input=n_input,
                n_z=n_z,
                n_clusters=n_clusters,
                pretrain_path=pretrain_path,
                v=1.0).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=lr)

    # KNN Graph
    adj = load_graph(vectors, name, k)
    adj = adj.cuda()

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y
    with torch.no_grad():
        _,_, z = model.ae(data)

    kmeans = KMeans(n_clusters=n_clusters, n_init=2, random_state= 2)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred,0, 'pae')

    y_pred = []

    for epoch in range(nb_epochs):
        if epoch % 1 == 0:
        # update_interval
            _, tmp_q, pred, _ = model(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
        
            res1 = tmp_q.cpu().numpy().argmax(1)       #Q
            sil= silhouette_score(tmp_q.cpu().numpy(), tmp_q.cpu().numpy().argmax(1))
            res2 = pred.data.cpu().numpy().argmax(1)   #Z
            res3 = p.data.cpu().numpy().argmax(1)      #P
            eva(y, res1,sil, str(epoch) + 'Q')
            eva(y, res2,sil, str(epoch) + 'Z')
            eva(y, res3,sil, str(epoch) + 'P')
            

        x_bar, q, pred, _ = model(data, adj)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)

        loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == nb_epochs - 1:
            y_pred = res1


    return y_pred


def main (embedding_dimension):
    print("use cuda: {}".format(args.cuda))
    cuda = torch.cuda.is_available()
    dataset = load_data('X') 
    device = torch.device("cuda" if cuda else "cpu")
    pretrain_path = 'src/DeepClusteringAlgorithm/SDCN/' + 'data/{}.pkl'.format('X')
    n_clusters = 26 #number of class
    n_input = embedding_dimension


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='X') # 'reut' - 'X'
    parser.add_argument('--k', type=int, default=0) # version(number) of graph generated for this dataset
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_clusters', default=26, type=int)
    parser.add_argument('--n_z', default=104, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    args.pretrain_path = 'src/DeepClusteringAlgorithm/SDCN/data/{}.pkl'.format(args.name)
    dataset = load_data('X') 

    if args.name == 'X':
        args.n_clusters = 26 #number of class
        args.n_input = nb_dimension


    print(args)

    train_sdcn(dataset)
