import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from tqdm import tqdm
import community as community_louvain
import networkx as nx
import pandas as pd
from collections import Counter

# Load the att_groundtruth 
df = pd.read_parquet('groundtruth_parquets/att_groundtruth.parquet')

# Calculate the total number of columns for each table
total_columns_query = df.groupby('query_table').nunique()['query_col_name']
total_columns_candidate = df.groupby('candidate_table').nunique()['candidate_col_name']

# Count the number of unionable columns 
unionable_columns = df.groupby(['query_table', 'candidate_table']).nunique()['query_col_name']

# Calculate the percentage of unionable columns
df_unionable = pd.DataFrame(unionable_columns).reset_index()
df_unionable['total_query'] = df_unionable['query_table'].map(total_columns_query)
df_unionable['total_candidate'] = df_unionable['candidate_table'].map(total_columns_candidate)
df_unionable['percentage_query'] = df_unionable['query_col_name'] / df_unionable['total_query']
df_unionable['percentage_candidate'] = df_unionable['query_col_name'] / df_unionable['total_candidate']


df_filtered = df_unionable[(df_unionable['percentage_query'] >= 0.4) & (df_unionable['percentage_candidate'] >= 0.4)]

# Create a network 
G = nx.from_pandas_edgelist(df_filtered, 'query_table', 'candidate_table')

# Compute the best partition using the Louvain method
partition = community_louvain.best_partition(G)


labels = {node: i for node, i in partition.items()}

# Count the number of tables in each community
unionable_counts = Counter(labels.values())


labels_df = pd.DataFrame.from_dict(labels, orient='index', columns=['Label']).reset_index()
labels_df.columns = ['File', 'Label']


labels_df = labels_df[labels_df['Label'].map(unionable_counts) > 1]


labels_df.to_csv('label.csv', index=False)

# Load the recall_groundtruth data 
recall_df = pd.read_parquet('groundtruth_parquets/recall_groundtruth.parquet')


merged_df = recall_df.merge(labels_df, left_on='query_table', right_on='File')

# Map community label to unionable_counts
merged_df['predicted'] = merged_df['Label'].map(unionable_counts)

# Compare the unionable counts from the groundtruth 
comparison = merged_df[['unionable_count', 'predicted']]

comparison.to_csv('comparison.csv', index=False)



