import torch
from torch.utils.data import Dataset
import spacy

class Word2VecDataset(Dataset):

    def __init__(self, path_to_data='corpus.txt', window_size=2):
        with open(path_to_data, 'r', encoding='utf-8') as f:
            data = f.read().lower()
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(data)
        self.context_target = [([doc[i - (j + 1)] for j in range(window_size)] + \
                                [doc[i + (j + 1)] for j in range(window_size)],
                                doc[i])
                               for i in range(window_size, len(doc) - window_size)]

        self.vocab = doc
        self.vocab_size = len(self.vocab)
        self.window_size = window_size

    def __getitem__(self, idx):
        idx+=2
        target = torch.tensor([idx])
        context = torch.tensor([idx-1, idx-2, idx+1, idx+2])
        return context, target

    def __len__(self):
        return len(self.context_target)

