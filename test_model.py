import torch
from torch.utils.data import DataLoader
from torch import nn

from matplotlib import pyplot as plt
import pandas as pd
import ast
import numpy as np
from pandas import DataFrame

from Dependencies.RNN_model_class import RNN
from Dependencies.MovieDataset import MovieOverviewDataset, collate_fn
from Dependencies.AdditionalFunctions import *

pad_value = -1

def create_smoothed_list(target):
    # Filter out padding values from the target tensor
    valid_targets = target[target != pad_value]
    one_hot_target = topK_one_hot(valid_targets.cpu().numpy(), 19)
    smoothed_target = smooth_multi_hot(torch.tensor(one_hot_target), len(valid_targets))
    return smoothed_target


# Loading RNN and device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
test_rnn = RNN().to(device)

# Initializing the test result csv if it does not exist:
try:
    inference = pd.read_csv('model_test_run.csv')
except FileNotFoundError:
    print("Couldn't find test run file.")
    # Loading the dataset
    test_loader = torch.load('test_ds_loader.pt', weights_only=False)

    # ---TestRun---
    test_loss = []
    loss_func = nn.BCEWithLogitsLoss()

    for x, target in test_loader:
        pred = test_rnn.forward(x.to(device)).flatten().cpu()
        logit_target = create_smoothed_list(target=target)
        loss_value = loss_func(pred, logit_target)
        # Store as float, not tensor
        test_loss.append(loss_value.item())

    # Preserve test inference values
    df = pd.DataFrame({'inf_output': test_loss})
    df.to_csv('model_test_run.csv', index=False, header=True)


# Plot comparison of model test performance VS model training performance 

try:
    lrn_grph = pd.read_csv("model_track.csv")
except FileNotFoundError:
    lrn_grph = pd.read_csv("track.csv")

# Process learning loss - convert from string to numeric if needed
lrn_loss_grph = lrn_grph["loss_arr"]

# Check if values are strings that need parsing
if isinstance(lrn_loss_grph.iloc[0], str):
    # If stored as string representation of list/tensor
    lrn_loss_grph = lrn_loss_grph.apply(ast.literal_eval)
    # Flatten if nested
    lrn_loss_grph = np.array([item if isinstance(item, (int, float)) else item[0] 
                               for item in lrn_loss_grph])
else:
    lrn_loss_grph = lrn_loss_grph.values

# Convert to float array
lrn_loss_grph = np.array(lrn_loss_grph, dtype=float)

# Process test loss
tst_grph_df = pd.read_csv("model_test_run.csv")
tst_grph = tst_grph_df['inf_output'].values
 
# Convert to float array
tst_grph = np.array(tst_grph, dtype=float)

# Debug prints
print("Learning loss shape:", lrn_loss_grph.shape)
print("Learning loss range:", lrn_loss_grph.min(), "to", lrn_loss_grph.max())
print("Test loss shape:", tst_grph.shape)
print("Test loss range:", tst_grph.min(), "to", tst_grph.max())
print("\nFirst 5 learning losses:", lrn_loss_grph[:5])
print("First 5 test losses:", tst_grph[:5])

# Plot the graphs
plt.figure(figsize=(10, 6))
plt.plot(range(len(lrn_loss_grph)), lrn_loss_grph, label='learning loss', color='red', alpha=0.7)
plt.plot(range(len(tst_grph)), tst_grph, label='test loss', color='blue', alpha=0.7)

plt.xlabel('Batch/Epoch Index')
plt.ylabel('Loss Value')
plt.title('Learning and Test Losses')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()