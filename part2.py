# %%
from collections import defaultdict
from sys import implementation
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import librosa
from hmmlearn import hmm

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

 
# %% STEP 10, STEP 11
def group_by_label(X, y):
    grouped = defaultdict(list)
    for a, b in zip(X, y):
        grouped[b].append(a)
    return grouped


def reshape(G):
    G_cat = np.concatenate(G, axis=0)
    G_lengths = np.array(list(map(len, G)))
    return G_cat, G_lengths


grouped_train = group_by_label(X_train, y_train)
grouped_val = group_by_label(X_val, y_val)
grouped_test = group_by_label(X_test, y_test)
classes = sorted(grouped_train.keys())

##################################################################################
# # The following doesn't work. Pomegranate calculates covariance matrices which
# # are not positive definite for some reason.
# # I will use hmmlearn instead.

# from pomegranate.distributions import MultivariateGaussianDistribution
# from pomegranate.gmm import GeneralMixtureModel
# from pomegranate.hmm import HiddenMarkovModel

# def create_and_fit_gmmhmm(group, n_components=4, n_mix=5):

#     group_cat = np.concatenate(group, axis=0, dtype=np.float64)
#     distributions = []
#     # Initialize the Gaussian Mixtures
#     for _ in range(n_components):
#         d = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
#                                             n_mix,
#                                             group_cat)
#         distributions.append(d)


#     # Create a Left-Right uniform transition matrix
#     transition_matrix = np.diag(np.ones(n_components))
#     transition_matrix += np.diag(np.ones(n_components-1), 1)
#     transition_matrix /= 2.
#     transition_matrix[-1, -1] = 1.
#     transition_matrix

#     # Start at state 0 and end at state n_components-1
#     starts = np.zeros(n_components)
#     starts[0] = 1.
#     ends = np.zeros(n_components)
#     ends[-1] = 1.

#     # Create the GMMHMM
#     state_names = [f's{i}' for i in range(n_components)]
#     model = HiddenMarkovModel.from_matrix(transition_matrix,
#                                       distributions, starts, ends,
#                                       state_names=state_names)
    
#     # Fit and return the GMMHMM
#     model.fit(group, max_iterations=5)
#     return model

    
# models = {c: create_and_fit_gmmhmm(group) for c, group in grouped_train.items()}
###################################################################################


def create_gmmhmm(n_components=4, n_mix=5,
                  covariance_type='full', algorithm='viterbi',
                  tol=1e-2, n_iter=10, verbose=True, implementation='scale'):

    # Create a Left-Right uniform transition matrix
    transmat = np.diag(np.ones(n_components))
    transmat += np.diag(np.ones(n_components-1), 1)
    transmat /= np.sum(transmat, axis=1)[..., np.newaxis]

    # Start at state 0
    startprob = np.zeros(n_components)
    startprob[0] = 1.
    # No need for an end_prob because n_components-1 is a unique
    # absorbing state by the values of the initial transition matrix.
    # The 0s of the transition matrix, stay at 0,
    # and the non 0s stay above 0 (up to some numerical error).
    # Thus, n_components-1 remains a unique absorbing state after training.
    # Also, there is no option for an endprob in hmmlearn.
    # See also: https://github.com/hmmlearn/hmmlearn/blob/
    #     03dd25107b940542b72513ca45ef57da22aac298/hmmlearn/tests/test_hmm.py#L214

    # Create the model.
    # ‘s’ for startprob, ‘t’ for transmat, ‘m’ for means,
    # ‘c’ for covars, and ‘w’ for GMM mixing weights
    model = hmm.GMMHMM(n_components=n_components, n_mix=n_mix,
                       covariance_type=covariance_type,
                       n_iter=n_iter,
                       tol=tol,  # loglikelihood increase
                       init_params='mcw',
                       params='tmcw',
                       algorithm=algorithm,  # Decoder algorithm
                       verbose=verbose,
                       implementation=implementation)
    model.startprob_ = startprob
    model.transmat_ = transmat

    return model

# %%
# Create and fit the models
path = 'hmmgmm-models.joblib'
models = {}
for c in classes:
    print(f'TRAINING CLASS {c}')
    G_train, lens_train = reshape(grouped_train[c])
    # Using diagonal covariances, instead of full covariances
    # so that the models donn't become too complicated
    model = create_gmmhmm(n_iter=1000, covariance_type='diag')
    model.fit(G_train, lens_train)
    models[c] = model
    print('\n'*2)
# joblib.dump(models, path)
# models = joblib.load(path)


# %% STEP 12
def predict(models, X_val):
    loglikelihoods = np.empty((len(X_val), len(classes)))
    for i, x in enumerate(X_val):
        for j, c in enumerate(classes):
            loglikelihoods[i, j] = models[c].score(x)
    pred_indices = np.argmax(loglikelihoods, axis=1)
    preds = classes[pred_indices]
    return preds    
    
    
    
    
# %%
