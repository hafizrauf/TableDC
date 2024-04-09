import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer

# Download required resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load data from a CSV file
file_path = 'music_brainz_20k.csv'   
df = pd.read_csv(file_path)


model = SentenceTransformer('all-mpnet-base-v2')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Tokenize, remove stopwords and perform lemmatization
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [lemmatizer.lemmatize(w) for w in tokens if w.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def encode_rows(df, columns):
    preprocessed_rows = []
    for _, row in df[columns].iterrows():
        preprocessed_row = [preprocess_text(str(x)) for x in row]
        preprocessed_rows.append(preprocessed_row)
    
    # Flatten the list and encode all items at once
    flat_list = [item for sublist in preprocessed_rows for item in sublist]
    encoded_items = model.encode(flat_list)

    # Calculate sum embeddings for each row
    embeddings = []
    step = len(columns)
    for i in range(0, len(encoded_items), step):
        sum_embedding = encoded_items[i:i+step].sum(axis=0)
        embeddings.append(sum_embedding)
    
    return embeddings

# Select column
columns = ['title', 'length', 'artist', 'album', 'year', 'language']

# Encode rows
embeddings = encode_rows(df, columns)


embeddings_df = pd.DataFrame(embeddings)

print(embeddings_df)
embeddings_df.to_csv('X.txt', header=None, sep=' ', index=False)
print("Shape of the final embeddings:", embeddings_df.iloc[0].shape)
