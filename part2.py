# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa

import torch
import torch.nn as nn
import torch.optim as optim

import premades.parser


torch.manual_seed(42)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# %% STEP 9
(X_train, X_test,
 y_train, y_test,
 spk_train, spk_test) = premades.parser.parser('data/part2/recordings')

