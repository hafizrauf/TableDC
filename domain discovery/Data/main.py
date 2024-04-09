
import os 
import pandas as pd 
import csv

# Input folder 

DIRECTORY_INPUT = 'education/'
DIRECTORY_OUTPUT = 'education_clean/'

for f in os.listdir(DIRECTORY_INPUT):

	column_terms = {} # Map column with terms

	file_name  = f.split('.silver')[0]

	# Open as df
	df = pd.read_csv(DIRECTORY_INPUT + f, error_bad_lines = False, header=None, sep='\t')

	for i in range(len(df)):
		if int(df[0][i]) not in column_terms.keys():
			column_terms[int(df[0][i])] = [str(df[2][i]).replace(',', ' ')]
		else:
			column_terms[int(df[0][i])].append(str(df[2][i]).replace(',', ' '))


	# Save in files as csv
	for column, terms in column_terms.items():
		column_name = [file_name + '.' + str(column)]
		lines = list(terms)
		#print(lines)

		# Write csv file
		with open(DIRECTORY_OUTPUT + str(column_name[0]), 'w', encoding='UTF8', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(column_name)  # write the header
			for line in lines:
				writer.writerow([line])  # write multiple rows
