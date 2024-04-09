import requests
import os
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import numpy as np
from gensim.models import KeyedVectors


url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec'
model_path = 'wiki.en.vec'

if not os.path.exists(model_path):
    print("Downloading FastText .vec model. This might take a while...")
    response = requests.get(url, stream=True)
    with open(model_path, 'wb') as file:
        for chunk in tqdm(response.iter_content(chunk_size=1024*1024), desc="Downloading", unit="MB"):
            if chunk:  # Filter out keep-alive chunks
                file.write(chunk)

# Load the .vec
model = KeyedVectors.load_word2vec_format(model_path, binary=False)


df = pd.read_csv('Data/monitor.csv')


embeddings = []
labels = []

# Prepare the label encoder
le = LabelEncoder()

# Fit the label encoder 
df['TARGET_ATTRIBUTE_ID'] = le.fit_transform(df['TARGET_ATTRIBUTE_ID'])

for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):

    attribute_name = row['ATTRIBUTE_NAME'].split()  # Split into words

    # Compute FastText embeddings for each word in attribute name
    attribute_name_embs = [model[word] for word in attribute_name if word in model]

    # Average the embeddings
    if attribute_name_embs:
        avg_embedding = np.mean(attribute_name_embs, axis=0)
    else:
        avg_embedding = np.zeros(model.vector_size)  

    # Append the averaged embedding to the list
    embeddings.append(avg_embedding)

   
    labels.append(row['TARGET_ATTRIBUTE_ID'])


embeddings = np.array(embeddings)
labels = np.array(labels)

# Save embeddings and labels
np.savetxt('X.txt', embeddings)
np.savetxt('labels.txt', labels, fmt='%d')
