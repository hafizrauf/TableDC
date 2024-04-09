#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 1. Input dataset

import string # use to remove punctuation from corpora
import os
import pandas as pd
import csv

# >>>>>>>>>>>>>>>>>>>> 1.1 Choose the dataset to input from raw
# 'music_brainz_20k', 'north_carolina_voters_5m'
dataset_name = 'music_brainz_20k'

if dataset_name == 'music_brainz_20k':
    DIRECTORY_INPUT = 'Data/Music_Brainz_20K/'
    
    
DIRECTORY_OUTPUT = 'embdi_master/pipeline/datasets/' 

# >>>>>>>>>>>>>>>>>>>>> 1.2 Preprocessing : remove ',' into text from columns

if dataset_name == 'music_brainz_20k':
    header_lab = ['CID'] # labels
    header_lines = ['title', 'length', 'artist', 'album', 'year', 'language'] # datas


d_save = DIRECTORY_OUTPUT


# Drop previous files saved
for f in os.listdir(DIRECTORY_OUTPUT):
    os.remove(os.path.join(DIRECTORY_OUTPUT, f))

# Proceeding to save preprocess files    
for filename in os.listdir(DIRECTORY_INPUT):
    l_save = filename.split('.csv')[0] + '_labels.csv'

    df = pd.read_csv(DIRECTORY_INPUT + filename, sep=',')
    
    datas_lab = []
    datas_lines = []

    for i in range(len(df)):
        # Proceed labels
        d = []
        d.append(df[header_lab[0]][i]) ; datas_lab.append(d)
    
        # Proceed lines
        d = []
        for col in header_lines:
            d.append(str(df[col][i]).replace('\n', '').replace(',', ' '))
        datas_lines.append(d)

    
    # Save labels as csv file
    with open(l_save, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header_lab)  # write the header
        writer.writerows(datas_lab)  # write multiple rows

    # Save lines as csv file
    with open(d_save + filename, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header_lines)  # write the header
        writer.writerows(datas_lines)  # write multiple rows
    
    
    
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
    
    main_walks(input_file, output_file)
    
    #try:
    #    main_walks(input_file, output_file)
    #except:
    #    bad_filename.append(filename)
    #else:
    #    pass

print('Number of bad file generated: ' + str(len(bad_filename)))
# Drop bad file name generated
for filename in bad_filename:
    os.remove(os.path.join(DIRECTORY_EDGE, filename))        




# >>>>>>>>>>>>>>>>>>>>>> 4. Generate info files from raw datasets(tables)

DIRECTORY_INFO = 'embdi_master/pipeline/info/'
DIRECTROY_RAW = DIRECTORY_OUTPUT
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
DIRECTORY_EMB = 'embdi_master/pipeline/embeddings/'

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
    str_all += 'experiment_type:ER\n'
    
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

# Drop all previous files saved in this folder: 'pipeline/embeddings/'
for f in os.listdir(DIRECTORY_EMB):
    os.remove(os.path.join(DIRECTORY_EMB, f))
    
# Rows Embedding
#python main.py -d 'embdi_master/pipeline/config_files/' 

        
        
        
        
#-------------- Getting row vectors for Music Brainz-----------------


DIRECTORY_DATASET = DIRECTORY_OUTPUT
DIRECTORY_EMB = 'pipeline/embeddings/'

vectors = []
Text = []
labels = []

for filename in os.listdir(DIRECTORY_EMB): 
    if '.emb' in str(filename):
        with open(DIRECTORY_EMB + filename, 'r', encoding='utf-8') as fp: 
            lines = fp.readlines()
            for line in lines:
                if line[:5] == 'idx__' :
                    # Get index of line int the raw dataset
                    index_line = int(line.split(' ')[0].split('__')[1])
                    
                    # Add label
                    df_lab = pd.read_csv(filename.split('.emb')[0] + '_labels.csv', sep=',')
                    
                    if dataset_name == 'music_brainz_20k':
                        col_lab = 'CID'
                    if dataset_name == 'north_carolina_voters_5m':
                        col_lab = 'recid'
                    
                    labels.append(int(df_lab[col_lab][index_line]))
                
                    #print(len(set(labels)))
                                    
                    # Add vector
                    vectors.append(np.array([float(x) for x in line.split(' ')[1:]]))
                
                    # Add text
                    with open(DIRECTORY_DATASET + filename.split('.emb')[0] + '.csv', 'r', encoding='utf-8') as ftt:
                        lines_txt = ftt.readlines()
                        # +1: because we remove header line
                        Text.append(lines_txt[index_line + 1]) 

print('Number of records/entities: ' + str(len(vectors)))
print('Number of labels: ' + str(len(set(labels))))


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>> 8. Deep clustering 

number_instance = len(vectors) #(vectors)
vectors = vectors[:number_instance]
Text = Text[:number_instance]
labels = labels[:number_instance]
# Transform list to array
vectors = np.array(vectors)
vectors1=pd.DataFrame(vectors) 
labels=pd.DataFrame(labels) 
vectors1.to_csv('X.txt', index=False, header = False,sep=' ')
labels.to_csv('labels.txt', index=False, header = False,sep=' ')


#-------------- Getting row vectors for geographicalSettelments-----------------


# vectors = []
# Text = []
# labels = []

# for filename in os.listdir(DIRECTORY_EMB): 
#     if '.emb' in str(filename):
#         with open(DIRECTORY_EMB + filename, 'r', encoding='utf-8') as fp: 
#             lines = fp.readlines()
#             for line in lines:
#                 if line[:5] == 'idx__' :
#                     # Get index of line int the raw dataset
#                     index_line = int(line.split(' ')[0].split('__')[1])
                    
#                     # Add label
#                     df_lab = pd.read_csv(filename.split('.emb')[0] + '_labels.csv', sep=',')
                    
#                     if dataset_name == 'geographicalSettelments':
#                         col_lab = 'label'
                    
#                     labels.append(int(df_lab[col_lab][index_line]))
                
#                     #print(len(set(labels)))
                                    
#                     # Add vector
#                     vectors.append(np.array([float(x) for x in line.split(' ')[1:]]))
                
#                     # Add text
#                     with open(DIRECTORY_DATASET + filename.split('.emb')[0] + '.csv', 'r', encoding='utf-8') as ftt:
#                         lines_txt = ftt.readlines()
#                         # +1: because we remove header line
#                         Text.append(lines_txt[index_line + 1]) 

# print('Number of records/entities: ' + str(len(vectors)))
# print('Number of labels: ' + str(len(set(labels))))

# number_instance = len(vectors) #(vectors)
# vectors = vectors[:number_instance]
# Text = Text[:number_instance]
# labels = labels[:number_instance]

# # Transform list to array
# vectors = np.array(vectors)
# y = labels

# vectors1=pd.DataFrame(vectors) 
# y1=pd.DataFrame(y) 
# vectors1.to_csv('X.txt', index=False, header = False,sep=' ')
# y1.to_csv('labels.txt', index=False, header = False,sep=' ')



        
