import pandas as pd
import numpy as np
from gensim.models import Word2Vec, FastText, KeyedVectors


def load_word2vec_model(path):
    """
    Load a Word2Vec model from the given path.
    """
    return KeyedVectors.load_word2vec_format(path, binary=False)


def generate_word2vec_embeddings(df, text_column, word2vec_model):
    def text_to_embedding(text):
        words = text.split()
        valid_word_vectors = [word2vec_model[word] for word in words if word in word2vec_model]
        if valid_word_vectors:
            return np.mean(valid_word_vectors, axis=0)
        else:
            return np.zeros(word2vec_model.vector_size)  # Return zero vector if no word matches
    
    if 'word2vec_embeddings' not in df.columns:
        df['word2vec_embeddings'] = df[text_column].apply(text_to_embedding)
    else:
        print("word2vec_embeddings column already exists. Appending new embeddings.")
        df['word2vec_embeddings'] = df[text_column].apply(text_to_embedding)

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

def generate_fasttext_embeddings(df, text_column, fasttext_model):
    def text_to_embedding(text):
        words = text.split()
        valid_word_vectors = [fasttext_model.wv[word] for word in words if word in fasttext_model.wv]
        if valid_word_vectors:
            return np.mean(valid_word_vectors, axis=0)
        else:
            return np.zeros(fasttext_model.vector_size)  # Return zero vector if no word matches
    
    if 'fasttext_embeddings' not in df.columns:
        df['fasttext_embeddings'] = df[text_column].apply(text_to_embedding)
    else:
        print("fasttext_embeddings column already exists. Appending new embeddings.")
        df['fasttext_embeddings'] = df[text_column].apply(text_to_embedding)

    # Convert the embeddings into separate columns
    embeddings_df = pd.concat([df["fasttext_embeddings"].apply(pd.Series)], axis=1)
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

# Function to generate glove embeddings
def generate_glove_embeddings(df, text_column, embeddings_dict):
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


