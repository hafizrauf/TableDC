import os
import pandas as pd
import numpy as np
import torch
import random
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from pytorch_widedeep import Trainer, Tab2Vec
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.models import (
    SAINT, TabFastFormer, TabNet, TabPerceiver,
    TabTransformer, FTTransformer, WideDeep
)

# Setting seeds for reproducibility
def set_seeds(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

set_seeds(555)


labels_df = pd.read_csv('label.csv')


vectors = []
processed_labels = []
dim_sizes = []
error = 0


file_errors = []


for file, label in tqdm(zip(labels_df['File'], labels_df['Label']), desc="Processing files"):

    if os.path.exists(os.path.join('benchmark_parquets/', file)):
        
        df = pd.read_parquet(os.path.join('benchmark_parquets/', file))

        # Identify category and continuous data
        cont_cols = df._get_numeric_data().columns
        cat_cols = list(set(df.columns) - set(cont_cols))

        # Preprocess the data
        tab_preprocessor = TabPreprocessor(cat_embed_cols=cat_cols, continuous_cols=cont_cols, scale=False)
        try:
            X_tab = tab_preprocessor.fit_transform(df)
        except Exception as e:
            file_errors.append((file, str(e)))
            error += 1
            continue

        # Define the model
        model_mapper = {
            'SAINT': SAINT,
            'TabFastFormer': TabFastFormer,
            'TabNet': TabNet,
            'TabPerceiver': TabPerceiver,
            'TabTransformer': TabTransformer,
            'FTTransformer': FTTransformer
        }
        tabmodel = model_mapper['TabNet'](
            column_idx=tab_preprocessor.column_idx,
            cat_embed_input=tab_preprocessor.cat_embed_input,
            continuous_cols=tab_preprocessor.continuous_cols,
            input_dim=8
        )
        model = WideDeep(deeptabular=tabmodel, pred_dim=768)

        # Embedding transformation
        t2v = Tab2Vec(model, tab_preprocessor)
        try:
            X_vec = t2v.transform(df)
            vectors.append(np.mean(X_vec, axis=0))
            processed_labels.append(label)
            dim_sizes.append(X_vec.shape[1])
        except Exception as e:
            file_errors.append((file, str(e)))
            error += 1
            continue

dim_size = max(dim_sizes)


df_vectors = pd.DataFrame(vectors)


is_varied_dimensions = len(set(dim_sizes)) > 1

# If vectors have different dimensions or missing values, interpolate
if is_varied_dimensions or df_vectors.isnull().any().any():
    df_vectors = df_vectors.iloc[:, :dim_size]
    df_vectors = df_vectors.interpolate(method='linear', limit_direction='both')


np.savetxt("X.txt", df_vectors.values)
np.savetxt("labels.txt", LabelEncoder().fit_transform(processed_labels))

print(f"Processed {len(vectors)} vectors with {error} errors.")

# Print errors
print("\nFiles with Errors:")
for file, error_msg in file_errors:
    print(f"{file} -> {error_msg}")
