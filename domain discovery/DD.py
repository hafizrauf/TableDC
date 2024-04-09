#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 1. Preprocessing adjust to EmbDi
import os 
import pandas as pd 
import csv
import json
import numpy as np 


# Select the dataset you want to test 
CAMERA  = 'Data/camera.csv'

DIRECTORY_INPUT = CAMERA 

# DIRECTORY_OUTPUT: contains the directory of the tables after preprocessing process
DIRECTORY_OUTPUT = 'embdi_master/pipeline/datasets/'

# Drop previous tables preproceed before perform new preprocessing
for f in os.listdir(DIRECTORY_OUTPUT):
    os.remove(os.path.join(DIRECTORY_OUTPUT, f))


# Open dataset as df
df = pd.read_csv(DIRECTORY_INPUT, error_bad_lines = False, sep=',')

for i in range(len(df)):
    # Build csv columns files from each df
    lines = str(df['VALUE'][i]).split('|')
    column_name = DIRECTORY_INPUT.split('/')[-1].split('.csv')[0] + '.' + str(i)
    column_name11 = str(df['ATTRIBUTE_NAME'][i])
    # Write csv file
    with open(DIRECTORY_OUTPUT + column_name, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([column_name])  # write the header
        writer.writerow([column_name11])  # write the header
        for line in lines:
            writer.writerow([line])  # write multiple rows




# >>>>>>>>>>>>>>>>>>>>>>>>>>> 2. Generates edgelist txt file for each table

from embdi_master.edgelist import main
# Drop all previous files saved in this folder: embdi_master/pipeline/edgelist
DIRECTORY_EDGE = 'embdi_master/pipeline/edgelist/'
for f in os.listdir(DIRECTORY_EDGE):
    os.remove(os.path.join(DIRECTORY_EDGE, f))

# Generate edgelist

for filename in os.listdir(DIRECTORY_OUTPUT):
    input_file = DIRECTORY_OUTPUT + filename 
    output_file = DIRECTORY_EDGE + filename.split('.csv')[0] + '.txt'
    main(input_file, output_file)



# >>>>>>>>>>>>>>>>>>>>>>>>>> 3. Generate random walks files fron edgelist files : too heavy 

import os
DIRECTORY_EDGE = 'embdi_master/pipeline/edgelist/'

from embdi_master.generate_random_walks import main_walks
# Drop all previous files saved in this folder: embdi_master/pipeline/walks
DIRECTORY_WALKS = 'embdi_master/pipeline/walks/'
for f in os.listdir(DIRECTORY_WALKS):
    os.remove(os.path.join(DIRECTORY_WALKS, f))

# Generate random walks
bad_filename = []
for filename in os.listdir(DIRECTORY_EDGE):
    input_file = DIRECTORY_EDGE + filename 
    output_file = filename.split('.txt')[0]
    
    #main_walks(input_file, output_file)
    
    try:
        main_walks(input_file, output_file)
    except:
        bad_filename.append(filename)
    else:
        pass

print('Number of bad file generated: ' + str(len(bad_filename)))
# Drop bad file name generated
for filename in bad_filename:
    os.remove(os.path.join(DIRECTORY_EDGE, filename))        


# >>>>>>>>>>>>>>>>>>>>>> 4. Generate info files from raw datasets(tables)

DIRECTORY_INFO = 'embdi_master/pipeline/info/'
DIRECTROY_RAW = 'embdi_master/pipeline/datasets/'
import pandas as pd

# Drop all previous files saved in this folder: embdi_master/pipeline/info/
for f in os.listdir(DIRECTORY_INFO):
    os.remove(os.path.join(DIRECTORY_INFO, f))

#Generate info
for filename in os.listdir(DIRECTROY_RAW):
    df = pd.read_csv(DIRECTROY_RAW + filename)
    with open(DIRECTORY_INFO + filename.split('.csv')[0] + '.txt', 'w', encoding='utf-8') as f:
        f.write(str(DIRECTROY_RAW + filename) + ',' + str(len(df)))


# >>>>>>>>>>>>>>>>>>>> 5. Generate config files from walks and edgelist files

DIRECTORY_WALKS = 'embdi_master/pipeline/walks/'
DIRECTORY_CONFIG_FILE = 'embdi_master/pipeline/config_files/'
DIRECTORY_EMB = 'pipeline/embeddings/'

# Drop all previous files saved in this folder: embdi_master/pipeline/config_files
for f in os.listdir(DIRECTORY_CONFIG_FILE):
    os.remove(os.path.join(DIRECTORY_CONFIG_FILE, f))

# str_common: contain common parameters(configs) that must have all tables
str_common = """# Walks configuration:
n_sentences:default
smoothing_method:smooth,0.5,200
flatten:all

# Embeddings configuration:
learning_method:skipgram
window_size:3
n_dimensions:300

# Test configuration:
ntop:10

# Miscellaneous:
indexing:basic
epsilon:0.1
"""

for filename in os.listdir(DIRECTORY_EDGE):    
    task = 'train'
    input_file = DIRECTORY_EDGE + filename # edgelist_file
    walks_file = DIRECTORY_WALKS + filename.split('.txt')[0] + '.walks'
    info_file = DIRECTORY_INFO + filename
    emb_file = DIRECTORY_EMB + filename.split('.txt')[0] + '.emb'
    output_file = filename.split('.txt')[0]
    
    
    # Init string
    str_all = '# Input configuration:\n'
    
    # Add task type
    str_all += 'task:' + task + '\n'
    
    # Add experiment type
    str_all += 'experiment_type:SM\n'
    
    # Add match file 
    str_all += 'match_file:\n'
    
    # Add edgelist file
    str_all += 'input_file:' + input_file + '\n'
    
    # Add walks file
    str_all += 'walks_file:' + walks_file + '\n'
    
    # Add enbedding file
    if task == 'match':
        str_all += 'embeddings_file:' + emb_file + '\n'
    
    # Add output file name
    str_all += 'output_file:' + output_file + '\n'
    
    # Add dataset info 
    str_all += 'dataset_info:' + info_file + '\n'
    
    # Add common parameters to each file
    str_all += str_common
    
    # Save config file
    with open(DIRECTORY_CONFIG_FILE + filename.split('.txt')[0], 'w', encoding='utf-8') as f:
        f.write(str_all)


# >>>>>>>>>>>>>>>>>>>> 6. Generate rows vectors (tables embedding)

# Drop all previous files saved in this folder: 'embdi_master/pipeline/embeddings/'
for f in os.listdir(DIRECTORY_EMB):
    os.remove(os.path.join(DIRECTORY_EMB, f))
    


# Column Embedding
#python main.py -d 'embdi_master/pipeline/config_files/' 



#------------Columns vector generation---------------------------------
domains = []
columns = []
texts = []

mean_vector= []

DIRECTORY_DOMAINS = 'domains/'


DIRECTORY_EMB = 'pipeline/embeddings/'

df = pd.read_csv(DIRECTORY_INPUT, error_bad_lines = False, sep=',')

for file in os.listdir(DIRECTORY_EMB):    
    
    raw_id = int(file.split('.')[1])
    domains.append(df['TARGET_ATTRIBUTE_ID'][raw_id])
    texts.append(str(df['VALUE'][raw_id]))
    columns.append(str(df['ATTRIBUTE_NAME'][raw_id]))
    
    # Add vector
    file_name_vector = DIRECTORY_EMB + file
    with open(file_name_vector, 'r', encoding='utf-8') as fv:
        lines = fv.readlines()
        lines=np.delete(lines, 0, 0)
        vectors = []
        for line in lines: 
            if 'idx' in line.split(' ')[0] or 'tt__' in line.split(' ')[0] or 'tn__' in line.split(' ')[0] or 'cid__' in line.split(' ')[0] :
                vectors.append(np.array([float(x) for x in line.split(' ')[1:]]))
        #print(len(vectors))
        vectors = np.mean(vectors,axis=0)
        mean_vector.append(vectors)
        
print('Number of columns: ' + str(len(mean_vector)))
print('Number of labels(domains): ' + str(len(set(domains))))

  
# Encode labels in numerical format
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(domains)
    
# Save vectors 
vectors= pd.DataFrame(mean_vector)
y = pd.DataFrame(y)
vectors.to_csv('X.txt', header =None, sep = ' ',index=False)
y.to_csv('labels.txt', header =None, sep = ' ',index=False)





        