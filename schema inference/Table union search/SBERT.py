import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import torch


labels_df = pd.read_csv('label.csv')

# Initialize the SBERT model
model = SentenceTransformer('all-minilm-l6-v2')

# Initialize a list to store the embeddings and labels
embeddings = []
labels = []

# Process each file
for file, label in tqdm(zip(labels_df['File'], labels_df['Label']), desc="Processing files"):

    if os.path.exists(os.path.join('benchmark_parquets/', file)):
     
        df = pd.read_parquet(os.path.join('benchmark_parquets/', file))

        # Get the headers
        headers = df.columns.tolist()

        # Compute the SBERT
        header_embeddings = model.encode(headers, convert_to_tensor=True)

        # Compute the average embedding
        avg_embedding = torch.mean(header_embeddings, dim=0)

        
        embeddings.append(avg_embedding)
        labels.append(label)


embeddings = np.stack(embeddings)


np.savetxt('X.txt', embeddings)


np.savetxt('labels.txt', labels)
