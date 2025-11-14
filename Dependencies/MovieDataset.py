import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from Dependencies.AdditionalFunctions import get_indices
import ast



global PAD_VALUE
global EPOCH_NUMBER
PAD_VALUE = -1
EPOCH_NUMBER = 1


global file_path
directory = r"Datasets"
x_file_path = directory+"\movies_overview.csv"
label_file_path = directory+"\movies_genres.csv"

movie_genres = pd.read_csv(x_file_path)

class MovieGenresDataset:
    def __init__(self):
        self.movie_ds = pd.read_csv(x_file_path)
        self.movie_lables = pd.read_csv(label_file_path)

    def __getitem__(self, idx):
        return self.movie_ds.iloc[idx]
    def __len__(self):
        return len(self.movie_ds)
    def getDs(self):
        return self.movie_ds
    def getLabel(self):
        return self.movie_lables
    def get_classes(self): # convert genre id's to classes by index for the model
        movie_genre_labels = self.getLabel()['id'].tolist()
        movie_id_loc = self.movie_ds['genre_ids'].map(ast.literal_eval).tolist()
        return [get_indices(movie_genre_labels, one_id_loc) for one_id_loc in movie_id_loc]




class MovieOverviewDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_overviews, id_loc_set):
        self.tokenized_ovw = tokenized_overviews
        self.id_loc_set = id_loc_set

    def __getitem__(self, idx):
        return self.tokenized_ovw[idx], torch.tensor(self.id_loc_set[idx])

    def __len__(self):
        return len(self.id_loc_set)

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Properly pads sequences and returns sequence lengths.
    This is CRITICAL for RNNs to ignore padding tokens.
    
    Args:
        batch: List of tuples (embeddings, targets)
        
    Returns:
        padded_sequences: Padded embeddings (batch, max_seq_len, embed_dim)
        padded_targets: Padded targets (batch, max_target_len)
        seq_lengths: Actual sequence lengths before padding (batch,)
    """
    # Separate embeddings and targets
    embeddings_list = [item[0] for item in batch]  # List of (seq_len, embed_dim) tensors
    targets_list = [item[1] for item in batch]     # List of (num_genres,) tensors
    
    # Get sequence lengths BEFORE padding (critical!)
    seq_lengths = torch.LongTensor([emb.shape[0] for emb in embeddings_list])
    
    # Pad embeddings to the longest sequence in the batch
    # batch_first=True gives shape (batch, seq_len, embed_dim)
    padded_sequences = pad_sequence(embeddings_list, batch_first=True, padding_value=0.0)
    
    # Pad targets (genre labels)
    padded_targets = pad_sequence(targets_list, batch_first=True, padding_value=-1)
    
    return padded_sequences, padded_targets, seq_lengths
