from __future__ import print_function, division
import argparse
import random
from time import time
from sklearn.cluster import Birch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from utils import load_data
from evaluation import eva
from tqdm import tqdm

# set seed
random.seed(555)
np.random.seed(555)
torch.manual_seed(555)


nb_dimension = 768
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
        return x_bar, z

class TableDC(nn.Module):

    def __init__(self, n_enc_1, n_dec_1, 
                n_input, n_z, n_clusters, v=1):
        super(TableDC, self).__init__()
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_dec_1=n_dec_1,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x):
        x_bar, z = self.ae(x)        
        device = z.device
        assert device == self.cluster_layer.device, "z and self.cluster_layer must be on the same device"

        # Ensure that z and self.cluster_layer have the correct shapes
        assert len(z.shape) == 2 and len(self.cluster_layer.shape) == 2, "z and self.cluster_layer must have shape (batch_size, n_features)"
        assert z.shape[1] == self.cluster_layer.shape[1], "z and self.cluster_layer must have the same number of features"
        
        # Create the covariance matrix
        cov_scale = 0.01  # Scale factor
        cov_matrix = torch.eye(z.size(1), device=device) * cov_scale
        
        # Compute the Mahalanobis distance using the Cholesky decomposition
        chol_cov = torch.linalg.cholesky(cov_matrix)
        z_centered = z.unsqueeze(1) - self.cluster_layer.unsqueeze(0)
        z_chol = torch.linalg.solve(chol_cov.unsqueeze(0), z_centered.transpose(2, 1)).transpose(2, 1)
        mahalanobis_distance = torch.norm(z_chol, dim=2)
        
        # Incorporate the Cauchy distribution as the kernel    
        gamma = 0.5  # Gamma is a hyperparameter for the Cauchy distribution
        q = 1.0 / (1.0 + mahalanobis_distance / (gamma ** 2))
        
        # Normalize and apply softmax
        predict = F.softmax(q, dim=1)
        
        return x_bar, q, predict, z

def target_distribution(q):
    weight = q ** 2 / torch.sum(q, dim=0)
    return (weight.t() / torch.sum(weight, dim=1)).t()



def mahalanobis_distance(z, cluster_layer, cov_matrix_inv):
    diff = z.unsqueeze(1) - cluster_layer
    temp = torch.matmul(diff.unsqueeze(-2), cov_matrix_inv)
    temp = temp.squeeze(-2)
    elliptical_distance = torch.sum(temp * diff, dim=-1)
    return elliptical_distance


def train_TableDC(dataset): 
    print(len(dataset.x[0]))
    model = TableDC(1000, 1000, 
                n_input=args.n_input,
                n_z=args.n_z,
                n_clusters=args.n_clusters,
                v=1.0).to(device)
    print(model)
    print(len(dataset.x[0]))

    optimizer = Adam(model.parameters(), lr=args.lr)

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y
    with torch.no_grad():
        _, z = model.ae(data)
    
    # Birch Initialization
    birch = Birch(n_clusters=args.n_clusters)
    y_pred = birch.fit_predict(z.data.cpu().numpy())
    
    # Compute cluster centers using labels and data points
    cluster_centers = []
    for i in range(args.n_clusters):
        cluster_centers.append(z[y_pred == i].mean(dim=0).cpu().numpy())
    
    model.cluster_layer.data = torch.tensor(cluster_centers).to(device)
    
    nb_epochs = 200
    
    for epoch in tqdm(range(nb_epochs), desc="Epochs"):
        if epoch % 1 == 0:
            # update_interval
            _, tmp_q, pred, _ = model(data)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)     
            res2 = pred.data.cpu().numpy().argmax(1)   
            if epoch == 199:  
                eva(y, res2, str(epoch))
    
        x_bar, q, pred, _ = model(data)
        
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)

        loss = 0.9 * ce_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='X') 
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_clusters', default=26, type=int)
    parser.add_argument('--n_z', default=100, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    args.pretrain_path = 'data/{}.pkl'.format(args.name)
    dataset = load_data(args.name) 
    print(len(dataset.x[0]))

    if args.name == 'X':
        args.n_clusters = 26 #number of GT clusters
        args.n_input = nb_dimension


    print(args)
    start = time()
    train_TableDC(dataset)
    end = time()
    print("Training took ",{end-start}," sec to run.")