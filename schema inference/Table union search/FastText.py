import pandas as pd
import numpy as np
from gensim.models.fasttext import FastText
from tqdm import tqdm
import os


labels_df = pd.read_csv('label.csv')

# Load the FastText
model_path = 'cc.en.300.bin'  
model = FastText.load_fasttext_format(model_path)

# Initialize a list 
embeddings = []
labels = []


for file, label in tqdm(zip(labels_df['File'], labels_df['Label']), desc="Processing files"):

    if os.path.exists(os.path.join('benchmark_parquets/', file)):
        # Load the data
        df = pd.read_parquet(os.path.join('benchmark_parquets/', file))

        # Get the headers
        headers = df.columns.tolist()

        # Compute the FastText 
        header_embeddings = [model.wv[header] for header in headers]

        avg_embedding = np.mean(header_embeddings, axis=0)

        # Add the average embedding
        embeddings.append(avg_embedding)
        labels.append(label)

embeddings = np.stack(embeddings)


np.savetxt('X.txt', embeddings)

# Write the labels to a text file
np.savetxt('labels.txt', labels)
