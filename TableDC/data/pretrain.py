import random
import torch
import numpy as np
from time import time
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn.functional import mse_loss

# Set random seed for reproducibility
random.seed(555)
np.random.seed(555)
torch.manual_seed(555)

CORPORA_NAME = 'X'  # Dataset identifier
NB_DIMENSION = 768  # Number of dimensions (replace the value as per encoding scheme)

class AutoEncoder(nn.Module):
    """
    AutoEncoder for sentence embedding.
    """
    def __init__(self, n_input, n_enc_1, n_z, n_dec_1):
        super(AutoEncoder, self).__init__()
        self.enc_1 = nn.Linear(n_input, n_enc_1)
        self.z_layer = nn.Linear(n_enc_1, n_z)
        self.dec_1 = nn.Linear(n_z, n_dec_1)
        self.x_bar_layer = nn.Linear(n_dec_1, n_input)

    def forward(self, x):
        enc_h1 = torch.relu(self.enc_1(x))
        z = self.z_layer(enc_h1)
        dec_h1 = torch.relu(self.dec_1(z))
        x_bar = self.x_bar_layer(dec_h1)
        return x_bar, z

class DatasetLoader(Dataset):
    """
    Dataset loader.
    """
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float), torch.tensor(idx)

class Trainer:
    """
    Trainer for AutoEncoder.
    """
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.loss_history = []

    def train(self, epochs=30, batch_size=256):
        train_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            for x, _ in train_loader:
                x = x.cpu()
                x_bar, _ = self.model(x)
                loss = mse_loss(x_bar, x)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.log_loss(epoch)
        torch.save(self.model.state_dict(), CORPORA_NAME + '.pkl')

    def log_loss(self, epoch):
        with torch.no_grad():
            x = torch.tensor(self.dataset.x, dtype=torch.float).cpu()
            x_bar, _ = self.model(x)
            loss = mse_loss(x_bar, x)
            self.loss_history.append(loss.item())
            print(f'Epoch {epoch}, Loss: {loss.item()}')

# Main execution
def main():
    model = AutoEncoder(n_input=NB_DIMENSION, n_enc_1=1000, n_z=100, n_dec_1=1000).cpu()
    x = np.loadtxt(CORPORA_NAME + '.txt', dtype=float)
    dataset = DatasetLoader(x)
    
    trainer = Trainer(model, dataset)
    start = time()
    trainer.train()
    end = time()

    print("Pretraining took ", {end - start}, " sec to run.")

if __name__ == "__main__":
    main()
