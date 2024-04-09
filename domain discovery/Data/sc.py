

DIR_1 = 'Tables_Education/'
DIR_2 = 'Tables_Finance/'


import os
from sh import gunzip
import pandas as pd 


'''
for f in os.listdir(DIR_1):
	gunzip( DIR_1 + f)

for f in os.listdir(DIR_2):
	gunzip( DIR_2 + f)

'''


for f in os.listdir(DIR_2):
	csv_table = pd.read_table(DIR_2 + f, sep='\t')
	csv_table.to_csv(DIR_2 + f.split('.tsv')[0] + '.csv', index=False)


