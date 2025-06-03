import torch
import numpy as np
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, sentences, binary, model, labels=None, max_len=100, pad_token="<pad>"):
        self.sentences = sentences # Each sentence should already be tokenized, i.e sentence is a list of tokens
        self.labels = labels  # Can be None for inference
        self.model = model
        self.binary = binary # To determine if the model passed is loaded through KeyedVectors or Model.load()
        self.max_len = max_len
        self.pad_token = pad_token
        self.embed_dim = model.vector_size

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        # Pad or truncate
        padded = sentence[:self.max_len] + [self.pad_token] * (self.max_len - len(sentence))
        padded = padded[:self.max_len]

        # Mask
        mask = [1 if word != self.pad_token else 0 for word in padded]

        # Word vectors
        if self.binary:
            # Model sent as argument, i.e Fasttext
            vectors = [
                np.zeros(self.embed_dim) if word == self.pad_token else
                self.model.wv[word] if word in self.model.wv else
                np.zeros(self.embed_dim)
                for word in padded
            ]
        else:
            # KeyedVectors sent as argument for W2V also the W2V model should have embedding for "<pad>" token
            vectors = [
                self.model[word] if word in self.model else self.model[self.pad_token]
                for word in padded
            ]
            
                

        inputs = (
            torch.tensor(vectors, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool)
        )

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return inputs + (label,)
        else:
            return inputs