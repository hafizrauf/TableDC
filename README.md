# TableDC

# Prerequisites
Python 3.x
PyTorch
NumPy
scikit-learn

# Dataset
The dataset comprises embeddings stored in data/X.txt and ground truth labels in data/label.txt.

# Setup

# Pretraining the AutoEncoder
Navigate to the data/ and run the pretraining script (pretrain_ae.py):
python pretrain_ae.py --pretrain_path data/X.txt

# Applying TableDC
Ensure that the pretrained model (X.pkl) and dataset (X.txt and label.txt) are in the data/ directory.
Run the script (TableDC.py):
python TableDC.py --pretrain_path data/X.pkl --name X
