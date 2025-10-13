import pandas as pd
from Dependencies.AdditionalFunctions import get_indices
import ast

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




