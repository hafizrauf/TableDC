import numpy as np
import h5py
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from src.DeepClusteringAlgorithm.SDCN.data.evaluation import eva

#torch.cuda.set_device(3)
# set seed
random.seed(555)
np.random.seed(555)
torch.manual_seed(555)


class AE(nn.Module):

    def __init__(self, n_enc_1, n_dec_1,
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


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def pretrain_ae(model, dataset, y, nb_epochs):
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    optimizer = Adam(model.parameters(), lr=1e-3)
    #optimizer = Adam(nn.Linear(1, 1).parameters(), lr=1e-3) # model.parameters
    nb_epochs = 30
    for epoch in range(nb_epochs):
        # adjust_learning_rate(optimizer, epoch)
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.cpu()

            x_bar, _ = model(x)
            loss = F.mse_loss(x_bar, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x = torch.Tensor(dataset.x).cpu().float()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss)) 
            if epoch == 29:
                kmeans = KMeans(n_clusters=37, n_init=2, random_state= 2).fit(z.data.cpu().numpy())
                eva(y, kmeans.labels_, epoch)

        torch.save(model.state_dict(), 'src/DeepClusteringAlgorithm/SDCN/data/X' + '.pkl')
