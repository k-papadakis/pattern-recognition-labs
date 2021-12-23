# %% STEP 2
import pathlib
import re
from librosa.feature.spectral import mfcc

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa

SR = 22050


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
def stack_data(mfccs, deltas, delta2s):
    return list(map(np.vstack, zip(mfccs, deltas, delta2s)))


def compute_means_and_stds(stacked):
    means = [np.mean(arr, axis=1) for arr in stacked]
    stds = [np.std(arr, axis=1) for arr in stacked]
    return means, stds


def plot_scatter(x, y, grouper, title=None, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots()
    sns.scatterplot(x=x, y=y, hue=grouper, style=grouper, palette='tab10',legend='full', ax=ax, **kwargs)
    ax.set_title(title)


def step_5():
    
    stacked = stack_data(mfccs, deltas, delta2s)
    means, stds = compute_means_and_stds(stacked)

    fig, axs = plt.subplots(ncols=2, figsize=(16, 8))

    m = np.vstack([mean[:2] for mean in means])
    s = np.vstack([std[:2] for std in stds])
    plot_scatter(m[:, 0], m[:, 1], digits, 'Means', ax=axs[0])
    plot_scatter(s[:, 0], s[:, 1], digits, 'Standard Deviations', ax=axs[1])


step_5() 


# %% STEP 6


