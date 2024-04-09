import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

def encode_with_batches(texts, model, batch_size=32, device=None):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch, device=device)
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = SentenceTransformer('paraphrase-mpnet-base-v2').to(device)

data = pd.read_csv('dataset.csv', sep=',')

# Convert all elements to strings
data = data.applymap(lambda x: str(x))

# header + value columns
header = data.iloc[:, 0]
values = data.iloc[:, 1]
batch_size = 32

# embeddings for header and values 
header_embeddings = encode_with_batches(header.tolist(), model, batch_size=batch_size, device=device)
value_embeddings = encode_with_batches(values.tolist(), model, batch_size=batch_size, device=device)

# Compute the mean embedding
mean_embeddings = np.mean([header_embeddings, value_embeddings], axis=0)

print(f"Mean embeddings matrix shape: {mean_embeddings.shape}")

np.savetxt('X.txt', mean_embeddings, delimiter=' ', fmt='%s')