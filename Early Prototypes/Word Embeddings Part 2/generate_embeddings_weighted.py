import pandas as pd
import numpy as np
from gensim.models import FastText, KeyedVectors
from nltk.tokenize import word_tokenize
from joblib import load
from tokenizer import nepali_nltk_tokenizer
'''
this custom nepali tokenizer which was used to initialize tfidf vectorizer needs to be imported
for loading support
'''

# Load saved tf-idf model
tfidf_vectorizer = load('tfidf_vectorizer.joblib')

# Returns weighted vectors
def text_to_weighted_embedding(text, model, tfidf_vectorizer):
    '''Works for Word2Vec, Fasttext'''
    words = word_tokenize(text)
    valid_word_vectors = []
    tfidf_weights = []
    
    # Get the TF-IDF scores for each word
    tfidf_scores = tfidf_vectorizer.transform([text]).toarray().flatten()
    
    # Get feature names from the TF-IDF vectorizer
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    for word in words:
        if word in feature_names:
            index = feature_names.tolist().index(word)  # Get index of the word
            valid_word_vectors.append(model[word] * tfidf_scores[index])
            tfidf_weights.append(tfidf_scores[index])
        else:
            print(f"Word '{word}' not in TF-IDF vocabulary.")

    if valid_word_vectors:
        return np.sum(valid_word_vectors, axis=0) / np.sum(tfidf_weights)  # Weighted averaging
    else:
        return np.zeros(model.vector_size)  # Return zero vector if no word matches

def load_word2vec_model(path):
    """
    Load a Word2Vec model from the given path.
    """
    return KeyedVectors.load_word2vec_format(path, binary=False)


# Word2Vec embeddings for dataset
def generate_weighted_word2vec_embeddings(df, text_column, word2vec_model):
    if 'word2vec_embeddings' not in df.columns:
        df['word2vec_embeddings'] = df[text_column].apply(
            text_to_weighted_embedding, args=(word2vec_model, tfidf_vectorizer))
    else:
        print("word2vec_embeddings column already exists. Appending new embeddings.")
        df['word2vec_embeddings'] = df[text_column].apply(
            text_to_weighted_embedding, args=(word2vec_model, tfidf_vectorizer))

    # Convert the embeddings into separate columns
    embeddings_df = pd.concat([df["word2vec_embeddings"].apply(pd.Series)], axis=1)
    df = pd.concat([df.drop('word2vec_embeddings', axis=1), embeddings_df], axis=1)

    # Move the 'Target' column to the last position
    df = df[[col for col in df if col != 'Target'] + ['Target']]

    return df

def load_fasttext_model(path):
    """
    Load a FastText model from the given path.
    """
    return FastText.load(path)

def generate_weighted_fasttext_embeddings(df, text_column, fasttext_model):
    if 'fasttext_embeddings' not in df.columns:
        df['fasttext_embeddings'] = df[text_column].apply(
            text_to_weighted_embedding, args=(fasttext_model, tfidf_vectorizer))
    else:
        print("fasttext_embeddings column already exists. Appending new embeddings.")
        df['fasttext_embeddings'] = df[text_column].apply(
            text_to_weighted_embedding, args=(fasttext_model, tfidf_vectorizer))

    # Convert the embeddings into separate columns
    embeddings_df = pd.concat([df["fastext_embeddings"].apply(pd.Series)], axis=1)
    df = pd.concat([df.drop('fasttext_embeddings', axis=1), embeddings_df], axis=1)

    # Move the 'Target' column to the last position
    df = df[[col for col in df if col != 'Target'] + ['Target']]

    return df

def load_glove_embeddings_dict(path):
    embeddings_dict = {}
    with open(path, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict

# Function to generate_weighted glove embeddings
def generate_weighted_glove_embeddings(df, text_column, embeddings_dict):
    def text_to_embedding(text):
        words = text.split()
        valid_word_vectors = [embeddings_dict[word] for word in words if word in embeddings_dict]
        if valid_word_vectors:
            return np.mean(valid_word_vectors, axis=0)
        else:
            return np.zeros(len(next(iter(embeddings_dict.values()))))  # Return zero vector if no word matches
    
    # Add the glove_embeddings column to the DataFrame
    df['glove_embeddings'] = df[text_column].apply(text_to_embedding)

     # Convert the embeddings into separate columns
    embeddings_df = pd.concat([df["glove_embeddings"].apply(pd.Series)], axis=1)
    df = pd.concat([df.drop('glove_embeddings', axis=1), embeddings_df], axis=1)  

    # Move the 'Target' column to the last position
    df = df[[col for col in df if col != 'Target'] + ['Target']]
    
    return df


