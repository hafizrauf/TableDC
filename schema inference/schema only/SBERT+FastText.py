
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import random
import os
import fasttext.util
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')
# set seed
random.seed(555)
np.random.seed(555)

embedding_method = "bert" # 'bert', 'fasttext'


dataset_path = 'TextPre.csv'
df = pd.read_csv(dataset_path, sep=',')

if embedding_method == 'bert':

	model_ = SentenceTransformer('all-mpnet-base-v2') #all-mpnet-base-v2
	x = model_.encode(np.array(df['Text']))  # Contains all sentence embedding
	y =  np.array(df['Labels'])  # Contains labels of each sentence embbeded

if embedding_method == 'fasttext':
	sentences_native = df['Text'] # Retrieve raw sentence according to the dataframe
	sentences = [] # Contains all embedding for each word in every sentence
	x = []
	y =  np.array(df['Labels']) 

	stopwords_all = stopwords.words('english') # Recurring words that appears in common documents
	lemmatizer = WordNetLemmatizer() # For lematization

	# Perform faxtest model
	lang = 'en'
	fasttext.util.download_model(lang, if_exists='ignore')  # English
	model = 'cc.' + lang + '.300.bin'
	ft = fasttext.load_model(model)

	# Perform embedding for each word in every sentence
	for sentence in sentences_native:
		print(sentence)
		word_tokenized = word_tokenize(sentence)
		words = [lemmatizer.lemmatize(word) for word in word_tokenized if (word not in stopwords_all and len(word) > 1)]
		sentences.append(words)
		
	# Perform each sentence embedding by using mean of all word embedding in this sentence 
	for sentence in sentences:
		sentence_embedding=[]
		for word in sentence:
			sentence_embedding.append(ft.get_word_vector(word))

		x.append(np.mean(sentence_embedding, axis=0))


# Save sentences embedding and labels into two file





X= pd.DataFrame(x)
X.to_csv('X.txt', header =None, sep = ' ',index= None)


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(y)
Y= pd.DataFrame(Y)
Y.to_csv('labels.txt', header =None, sep = ' ',index= None)



