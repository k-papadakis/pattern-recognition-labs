# %%
import os
from collections import defaultdict
import pprint
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import ConfusionMatrixDisplay
from hmmlearn import hmm

import torch
import torch.nn as nn
import torch.optim as optim

import premades.parser

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
(X_train, X_test,
 y_train, y_test,
 spk_train, spk_test) = premades.parser.parser('data/part2/recordings', n_mfcc=13)

# %% STEP 9
# We'll use these indices in GridSearchCV
indices_train, indices_val  = train_test_split(np.arange(len(y_train)),
                                               test_size=0.2,
                                               stratify=y_train,
                                               random_state=RANDOM_STATE)
 
# %% STEP 10, STEP 11, STEP 12, STEP 13

##################################################################################
# The following doesn't work. Pomegranate calculates degenerate covariance matrices,
# then attempts to apply Cholesky decomposition and breaks.
# I will use hmmlearn instead.
# In hmmlearn degenerate covariance matrices still occur,
# but the implementation is able to handle them in most cases.

# from pomegranate.distributions import MultivariateGaussianDistribution
# from pomegranate.gmm import GeneralMixtureModel
# from pomegranate.hmm import HiddenMarkovModel


# def group_by_label(X, y):
#     grouped = defaultdict(list)
#     for a, b in zip(X, y):
#         grouped[b].append(a)
#     return grouped


# (X_train, X_val,
#  y_train, y_val,
#  spk_train, spk_val) = train_test_split(X_train, y_train, spk_train,
#                                                 test_size=0.8, stratify=y_train) 

# grouped_train = group_by_label(X_train, y_train)
# grouped_val = group_by_label(X_val, y_val)
# grouped_test = group_by_label(X_test, y_test)
# classes = sorted(grouped_train.keys())


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


def _create_gmmhmm(n_components=4, n_mix=5,
                   covariance_type='full', algorithm='viterbi',
                   tol=1e-2, n_iter=10, verbose=False, **kwargs
                   ):

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
                       **kwargs)
    model.startprob_ = startprob
    model.transmat_ = transmat

    return model


class EnsembleGMMHMM(BaseEstimator, ClassifierMixin):
    
    def __init__(self, n_components=4, n_mix=5, *,
                 covariance_type='diag', algorithm='viterbi',
                 tol=1e-2, n_iter=200, verbose=False,
                 ):
        self.n_components = n_components
        self.n_mix = n_mix
        self.covariance_type = covariance_type
        self.algorithm = algorithm
        self.tol = tol
        self.n_iter = n_iter
        self.verbose = verbose
        
    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        # Group by label
        grouped_dict = defaultdict(list)
        for a, b in zip(X, y):
            grouped_dict[b].append(a)
        grouped = [grouped_dict[c] for c in self.classes_]
        
        self.models_ = []
        for i, c in enumerate(self.classes_):
            if self.verbose:
                print(f'------- TRAINING CLASS {c} -------')    
            G = np.concatenate(grouped[i])  # hmmlearn requires the data in this form
            lengths = np.array(list(map(len, grouped[i])))
            model = _create_gmmhmm(n_components=self.n_components, n_mix=self.n_mix,
                                   covariance_type=self.covariance_type,
                                   algorithm=self.algorithm,
                                   tol=self.tol,
                                   n_iter=self.n_iter,
                                   verbose=self.verbose)
            model.fit(G, lengths)
            self.models_.append(model)
            
    def predict(self, X):
        check_is_fitted(self)
        n_samples = len(X)
        n_classes = len(self.classes_)
        loglikelihoods = np.empty((n_samples, n_classes))
        for i in range(n_samples):
            for j in range(n_classes):
                loglikelihoods[i, j] = self.models_[j].score(X[i])
        indices = np.argmax(loglikelihoods, axis=1)
        preds = self.classes_[indices]
        return preds
    

def grid_search(cv, path='gmmhmm-cv.joblib'):
    
    if os.path.exists(path):
        clf = joblib.load(path)
        return clf
        
    params = {'n_components': np.arange(1, 5),
              'n_mix': np.arange(1, 6),
              'covariance_type': ['spherical', 'diag', 'full', 'tied']}
    clf = GridSearchCV(EnsembleGMMHMM(), params,
                       cv=cv,
                       scoring='accuracy',
                       n_jobs=-1,
                       verbose=3)    
    clf.fit(X_train, y_train)
    joblib.dump(clf, path)
    return clf


def plot_val_test_confusion_matrices(estimator,
                                     X_train, y_train, indices_val,
                                     X_test, y_test
                                     ):
    fig, axs = plt.subplots(ncols=2, figsize=(12, 8))

    ConfusionMatrixDisplay.from_estimator(
        estimator,
        [X_train[idx] for idx in indices_val],
        [y_train[idx] for idx in indices_val],
        ax=axs[0],
        colorbar=False
    )
    axs[0].set_title('Confusion Matrix on the Validation Set')

    ConfusionMatrixDisplay.from_estimator(
        estimator,
        X_test,
        y_test,
        ax=axs[1],
        colorbar=False
    )
    axs[1].set_title('Confusion Matrix on the Test Set')
   

def step_10_11_12_13():
    # We use a validation set plus a test set, because when we choose
    # the estimator with the best fit, we introduce an overestimating bias
    # on the score, which might be significant.
    # To see this, consider the trivial case where we train and validate
    # the same estimator multiple times and then we pick the best one of them.
    # It's clear that the expected output of if this process would not be
    # the expected value of the accuracy, but higher than it.
    # On the other hand, the processes of evaluating of the score on the test set,
    # has an expected output equal to the true accuracy.
    
    cv = [(indices_train, indices_val)]
    clf = grid_search(cv)
    test_score = clf.best_estimator_.score(X_test, y_test)
    
    print('-------GRID-SEARCH RESULTS-------')
    pprint.pprint(clf.cv_results_)
    print()

    print(f'BEST PARAMS: {clf.best_params_}')
    print(f'VALIDATION ACCURACY: {clf.best_score_:6f}')
    print(f'TEST ACCURACY: {test_score:6f}')
    
    plot_val_test_confusion_matrices(clf.best_estimator_,
                                     X_train, y_train, indices_val,
                                     X_test, y_test)
    plt.show()


step_10_11_12_13()
# %% STEP 14
