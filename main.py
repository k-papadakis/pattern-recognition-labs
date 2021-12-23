# %%
import pathlib
import re
from librosa.feature.spectral import mfcc, poly_features, tonnetz, zero_crossing_rate
from numpy.lib.npyio import fromregex
from sklearn import neighbors

from lab1gnb import CustomNBClassifier

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

SR = 22050
# %% STEP 2
def data_parser(dir_path, sr=SR):

    path = pathlib.Path(dir_path)
    pattern = re.compile(r'(\w+?)(\d+)')
    word_to_digit = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
                     'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}

    speakers = []
    digits = []
    waves = []

    for p in path.iterdir():
        m = pattern.search(p.stem)
        digit = word_to_digit[m.group(1)]
        speaker = int(m.group(2))
        wave, _ = librosa.load(p, sr=sr)

        digits.append(digit)
        speakers.append(speaker)
        waves.append(wave)

    return waves, speakers, digits


waves, speakers, digits = data_parser('./data/digits')


# %% STEP 3
def compute_mfcc(wave):
    return librosa.feature.mfcc(wave, sr=SR, n_mfcc=13, win_length=25, hop_length=10)


def compute_delta(mfcc):
    return librosa.feature.delta(mfcc)


def compute_delta2(mfcc):
    return librosa.feature.delta(mfcc, order=2)


def step_3(waves):
    mfccs = list(map(compute_mfcc, waves))
    deltas = list(map(compute_delta, mfccs))
    delta2s = list(map(compute_delta2, mfccs))
    return mfccs, deltas, delta2s


mfccs, deltas, delta2s = step_3(waves)


# %% STEP 4
def plot_hist_grid(n1, n2, suptitle=None):
    # In the pdf it say to plot each occurence of each digit,
    # but in this answer on github it says we only need to plot 4 histograms:
    # https://github.com/slp-ntua/patrec-labs/issues/109
    # I will follow the github answer and plot 4 histograms.

    n1_idx = digits.index(n1)  # First occurence only
    n2_idx = digits.index(n2)
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

    sns.histplot(mfccs[n1_idx][0], ax=ax[0, 0])
    ax[0, 0].set_title(f'Digit {n1}, Coefficient 1')

    sns.histplot(mfccs[n1_idx][1], ax=ax[0, 1])
    ax[0, 1].set_title(f'Digit {n1}, Coefficient 2')

    sns.histplot(mfccs[n2_idx][0], ax=ax[1, 0])
    ax[1, 0].set_title(f'Digit {n2}, Coefficient 1')

    sns.histplot(mfccs[n2_idx][1], ax=ax[1, 1])
    ax[1, 1].set_title(f'Digit {n2}, Coefficient 2')

    fig.suptitle(suptitle)


def compute_mfsc(wave):
    melspec = librosa.feature.melspectrogram(
        wave,
        sr=SR,
        win_length=25,
        hop_length=10,
        n_mels=13
    )
    return np.log(melspec)


def get_speaker_digit_indices(n1, n2, speaker1, speaker2):
    speaker_digits = list(zip(speakers, digits))
    i11 = speaker_digits.index((speaker1, n1))
    i12 = speaker_digits.index((speaker1, n2))
    i21 = speaker_digits.index((speaker2, n1))
    i22 = speaker_digits.index((speaker2, n2))
    return i11, i12, i21, i22


def plot_corr(v, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots()
    sns.heatmap(np.corrcoef(v), cmap='viridis', ax=ax, **kwargs)


def plot_corr_grid(dct: bool, n1=2, n2=7, speaker1=1, speaker2=2):
    i11, i12, i21, i22 = get_speaker_digit_indices(n1, n2, speaker1, speaker2)
    func = compute_mfsc if dct else compute_mfcc

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

    plot_corr(func(waves[i11]), axs[0][0])
    axs[0][0].set_title(f'Speaker {speaker1}, digit {n1}')

    plot_corr(func(waves[i12]), axs[0][1])
    axs[0][1].set_title(f'Speaker {speaker1}, digit {n2}')
    
    plot_corr(func(waves[i21]), axs[1][0])
    axs[1][0].set_title(f'Speaker {speaker2}, digit {n1}')

    plot_corr(func(waves[i22]), axs[1][1])
    axs[1][1].set_title(f'Speaker {speaker2}, digit {n2}')
    
    fig.suptitle('MFCC' if dct else 'MFSC')


def step_4():
    plot_hist_grid(2, 7, suptitle='MFCCS')
    plot_corr_grid(dct=True)
    plot_corr_grid(dct=False)
    plt.show()


# %%  STEP 5
def stack_data(*args):
    return list(map(np.vstack, zip(*args)))


def compute_means_and_stds(stacked):
    means = np.vstack([np.mean(arr, axis=1) for arr in stacked])
    stds = np.vstack([np.std(arr, axis=1) for arr in stacked])
    return means, stds


def plot_scatter(x, y, grouper, title=None, ax=None, xlabel=None, ylabel=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots()
    sns.scatterplot(x=x, y=y, hue=grouper, style=grouper, palette='tab10',legend='full', ax=ax, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def step_5():
    
    stacked = stack_data(mfccs, deltas, delta2s)
    means, stds = compute_means_and_stds(stacked)

    fig, axs = plt.subplots(ncols=2, figsize=(16, 8))

    plot_scatter(means[:, 0], means[:, 1], digits, 'Means', ax=axs[0])
    plot_scatter(stds[:, 0], stds[:, 1], digits, 'Standard Deviations', ax=axs[1])


# %% STEP 6
def reduce_dims(data, n_dims):
    reductor = Pipeline(steps=[('scaler', StandardScaler()), ('pca', PCA(n_components=n_dims))])
    reduced = reductor.fit_transform(data)
    pca = reductor.named_steps['pca']
    evr = pca.explained_variance_ratio_
    return reduced, reductor, evr


def plot_reduced_2d(reduced_m, evr_m, reduced_s, evr_s):

    fig, axs = plt.subplots(ncols=2, figsize=(16, 8))

    plot_scatter(
        reduced_m[:,0], reduced_m[:,1],
        grouper=digits, ax=axs[0],
        title=f'PCA of Means. Explained variance: {evr_m[0]:.2f}, {evr_m[1]:.2f}',
        xlabel='PCA 1',
        ylabel='PCA 2'
    )

    plot_scatter(
        reduced_s[:,0], reduced_s[:,1],
        grouper=digits, ax=axs[1],
        title=f'PCA of Standard Deviation. Explained variance: {evr_s[0]:.2f}, {evr_s[1]:.2f}',
        xlabel='PCA 1',
        ylabel='PCA 2'
    )


def plot_reduced_3d(reduced_m, evr_m, reduced_s, evr_s):
    fig = plt.figure(figsize=(16, 8))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    scatter_m = ax.scatter(reduced_m[:, 0], reduced_m[:, 1], reduced_m[:, 2], c=digits, cmap='tab10')
    legend_m = ax.legend(*scatter_m.legend_elements(), loc="best", title="Digits")
    ax.add_artist(legend_m)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    ax.set_title('PCA of Standard Deviation. Explained variance:'
                f'{evr_m[0]:.2f}, {evr_m[1]:.2f}, {evr_m[2]:.2f}')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    scatter_s = ax.scatter(reduced_s[:, 0], reduced_s[:, 1], reduced_s[:, 2], c=digits, cmap='tab10')
    legend_s = ax.legend(*scatter_s.legend_elements(), loc="best", title="Digits")
    ax.add_artist(legend_s)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    ax.set_title('PCA of Standard Deviation. Explained variance:'
                f'{evr_s[0]:.2f}, {evr_s[1]:.2f}, {evr_s[2]:.2f}')


def step_6():
    stacked = stack_data(mfccs, deltas, delta2s)
    means, stds = compute_means_and_stds(stacked)
    
    reduced_m_2d, reductor_m_2d, evr_m_2d = reduce_dims(means, 2)
    reduced_s_2d, reductor_s_2d, evr_s_2d = reduce_dims(stds, 2)
    plot_reduced_2d(reduced_m_2d, evr_m_2d, reduced_s_2d, evr_s_2d)

    reduced_m_3d, reductor_m_3d, evr_m_3d = reduce_dims(means, 3)
    reduced_s_3d, reductor_s_3d, evr_s_3d = reduce_dims(stds, 3)
    plot_reduced_3d(reduced_m_3d, evr_m_3d, reduced_s_3d, evr_s_3d)


# %% STEP 7
def score_classifier(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    clf_score = clf.score(X_test, y_test)
    return clf_score


def compute_zcr(wave):
    return librosa.feature.zero_crossing_rate(wave, frame_length=25, hop_length=10)


def compute_poly(wave):
    return librosa.feature.poly_features(wave, sr=SR, hop_length=20, win_length=25, order=3)


def compare_augmented(clfs):
    stacked = stack_data(mfccs, deltas, delta2s)
    means, stds = compute_means_and_stds(stacked)
    X = np.hstack([means, stds])
    X_train, X_test, y_train, y_test = train_test_split(X, np.array(digits), test_size=0.3)
    scores = {name: score_classifier(clf, X_train, X_test, y_train, y_test) for name, clf in clfs.items()}

    zcrs = [compute_zcr(wave) for wave in waves]
    zcr_means, zcr_stds = compute_means_and_stds(zcrs)
    polys = [compute_poly(wave) for wave in waves]
    poly_means, poly_stds = compute_means_and_stds(polys)
    X_more = np.hstack([X, zcr_means, zcr_stds, poly_means, poly_stds])
    X_more_train, X_more_test, y_train, y_test = train_test_split(X_more, np.array(digits), test_size=0.3)
    scores_more = {name: score_classifier(clf, X_more_train, X_more_test, y_train, y_test) for name, clf in clfs.items()} 

    return scores, scores_more


def step_7():

    clfs_raw = {
        'gnb': GaussianNB(),
        'cgnb': CustomNBClassifier(),
        'svm': SVC(kernel='linear'),
        'knn': KNeighborsClassifier(n_neighbors=5, weights='distance', p=1),
        'rf': RandomForestClassifier(n_estimators=100)
    }

    clfs_scaled = {
        'gnb': make_pipeline(StandardScaler(), GaussianNB()),
        'cgnb': make_pipeline(StandardScaler(),  CustomNBClassifier()),
        'svm': make_pipeline(StandardScaler(), SVC(kernel='linear')),
        'knn': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights='distance', p=1)),
        'rf': make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100))
    }

    clfs_normalized = {
        'gnb': make_pipeline(Normalizer(), GaussianNB()),
        'cgnb': make_pipeline(Normalizer(),  CustomNBClassifier()),
        'svm': make_pipeline(Normalizer(), SVC(kernel='linear')),
        'knn': make_pipeline(Normalizer(), KNeighborsClassifier(n_neighbors=5, weights='distance', p=1)),
        'rf': make_pipeline(Normalizer(), RandomForestClassifier(n_estimators=100))
    }

    s_r, s_r_more = compare_augmented(clfs_raw)
    s_s,s_s_more = compare_augmented(clfs_scaled)
    s_n, s_n_more = compare_augmented(clfs_normalized)

    print('No preprocessing')
    print('Before augmenting:', s_r)
    print('After augmenting: ', s_r_more)
    print()
    print('Scaled')
    print('Before augmenting:', s_s)
    print('After augmenting: ', s_s_more)
    print()
    print('Normalized')
    print('Before augmenting:', s_n)
    print('After augmenting: ', s_n_more)
    

# %% STEP 8
