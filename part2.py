# %%
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import librosa
# import pomegranate
# from pomegranate.distributions import MultivariateGaussianDistribution
# from pomegranate.gmm import GeneralMixtureModel
# from pomegranate.hmm import HiddenMarkovModel
from pomegranate import *
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

import premades.parser


torch.manual_seed(42)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# %% STEP 9
(X_train_all, X_test,
 y_train_all, y_test,
 spk_train_all, spk_test) = premades.parser.parser('data/part2/recordings', n_mfcc=13)

(X_train, X_val,
 y_train, y_val,
 spk_train, spk_val) = train_test_split(X_train_all, y_train_all, spk_train_all,
                                        test_size=0.8, stratify=y_train_all)

 
# %% STEP 10
def group_by_label(X, y):
    grouped = defaultdict(list)
    for a, b in zip(X, y):
        grouped[b].append(a)
    return grouped


grouped_train = group_by_label(X_train, y_train)
grouped_test = group_by_label(X_test, y_test)
classes = sorted(grouped_train.keys())

n_components = 3 # 1..5
n_states = 2  # 1..4

c = classes[0]  # TODO: variable in a for loop
g_train = grouped_train[c]

#%%
def create_and_fit_gmhmm(group):
    
    group_cat = np.concatenate(group, axis=0, dtype=np.float64)
    distributions = []
    # Initialize the Gaussian Mixtures
    for _ in range(n_states):
        d = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
                                            n_components,
                                            group_cat)
        distributions.append(d)


    # Create a Left-Right uniform transition matrix
    transition_matrix = np.triu(np.ones((n_states, n_states)))
    transition_matrix /= np.arange(n_states, 0, -1)[..., np.newaxis]

    # Start at state 0 and end at state n_states-1
    starts = np.zeros(n_states)
    starts[0] = 1.
    ends = np.zeros(n_states)
    ends[-1] = 1.

    # Create the GMHMM
    state_names = [f's{i}' for i in range(n_states)]
    model = HiddenMarkovModel.from_matrix(transition_matrix,
                                      distributions, starts, ends,
                                      state_names=state_names)
    
    # Fit and return the GMHMM
    model.fit(group, max_iterations=5)
    return model

    
# models = {c: create_and_fit_gmhmm(group) for c, group in grouped_train.items()}

group = grouped_train[classes[0]]
model = create_and_fit_gmhmm(group)
# %%