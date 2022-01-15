# %%
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split

RANDOM_SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
class_mapping = {
    'Rock': 'Rock',
    'Psych-Rock': 'Rock',
    'Indie-Rock': None,
    'Post-Rock': 'Rock',
    'Psych-Folk': 'Folk',
    'Folk': 'Folk',
    'Metal': 'Metal',
    'Punk': 'Metal',
    'Post-Punk': None,
    'Trip-Hop': 'Trip-Hop',
    'Pop': 'Pop',
    'Electronic': 'Electronic',
    'Hip-Hop': 'Hip-Hop',
    'Classical': 'Classical',
    'Blues': 'Blues',
    'Chiptune': 'Electronic',
    'Jazz': 'Jazz',
    'Soundtrack': None,
    'International': None,
    'Old-Time': None
}


def read_fused_spectrogram(spectrogram_file):
    spectrogram = np.load(spectrogram_file)
    return spectrogram.T

def read_mel_spectrogram(spectrogram_file):
    spectrogram = np.load(spectrogram_file)[:128]
    return spectrogram.T

def read_chromagram(spectrogram_file):
    spectrogram = np.load(spectrogram_file)[128:]
    return spectrogram.T

# I will implement the DataSets differently, so that they load the data on demand,
# instead of preloading everything and filling precious memory.
# Also, perform padding on batch creation and split our datasets using torch's random_split.

class SpectrogramDataset(Dataset):
    def __init__(self, path, read_spec_fn, class_mapping, train=True):
        self.class_mapping = class_mapping
        self.read_spec_fn = read_spec_fn
        t = 'train' if train else 'test'
        self.data_dir = os.path.join(path, t)
        self.labels_file = os.path.join(path, f'{t}_labels.txt')
        data_files, labels_str = self.get_file_labels()
        self.data_files = np.array(data_files)
        self.labels_str, self.labels = np.unique(labels_str, return_inverse=True)
        
    def get_file_labels(self):
        data_files = []
        labels = []
        with open(self.labels_file) as f:
            next(f)  # Skip the headers
            for line in f:
                line = line.rstrip()
                t, label = line.split('\t')
                if self.class_mapping is not None:
                    label = self.class_mapping[label]
                if label is None:
                    continue
                t, _ = t.split('.', 1)
                data_file = f'{t}.fused.full.npy'
                data_files.append(data_file)
                labels.append(label)
        return data_files, labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        x = self.read_spec_fn(os.path.join(self.data_dir, self.data_files[index]))
        y = self.labels[index]
        return torch.Tensor(x), torch.LongTensor([y]), torch.LongTensor([len(x)])


def split_dataset(dataset, train_size, seed=RANDOM_SEED):
    n = len(dataset)
    n_train = int(train_size * n)
    n_val = n - n_train
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    dataset_train, dataset_val = random_split(dataset, [n_train, n_val], generator)
    return dataset_train, dataset_val


def collate_fn(batch):
    seqs, labels, lengths = map(list, zip(*batch))
    return pad_sequence(seqs, batch_first=True), torch.LongTensor(labels), torch.LongTensor(lengths)


def plot_spectograms(spec1, spec2, title1=None, title2=None, suptitle=None, cmap='viridis'):
    fig, axs = plt.subplots(2, figsize=(9, 12))
    img = librosa.display.specshow(spec1, ax=axs[0], cmap=cmap)
    librosa.display.specshow(spec2, ax=axs[1], cmap=cmap)
    axs[0].set_title(title1)
    axs[1].set_title(title2)
    fig.colorbar(img, ax=axs)
    fig.suptitle(suptitle)


# %% Prepare all datasets and loaders
raw_path = 'data/fma_genre_spectrograms'

mel_raw_train_full = SpectrogramDataset(raw_path, read_spec_fn=read_mel_spectrogram, train=True, class_mapping=class_mapping)
mel_raw_train, mel_raw_val = split_dataset(mel_raw_train_full, train_size=0.8)
mel_raw_test = SpectrogramDataset(raw_path, read_spec_fn=read_mel_spectrogram, train=False, class_mapping=class_mapping)
mel_raw_train_loader = DataLoader(mel_raw_train, batch_size=32, shuffle=True, collate_fn=collate_fn)
mel_raw_val_loader = DataLoader(mel_raw_val, batch_size=32, collate_fn=collate_fn)
mel_raw_test_loader = DataLoader(mel_raw_test, batch_size=32, collate_fn=collate_fn)

chroma_raw_train_full = SpectrogramDataset(raw_path, read_spec_fn=read_chromagram, train=True, class_mapping=class_mapping)
chroma_raw_train, chroma_raw_val = split_dataset(chroma_raw_train_full, train_size=0.8)
chroma_raw_test = SpectrogramDataset(raw_path, read_spec_fn=read_chromagram, train=False, class_mapping=class_mapping)
chroma_raw_train_loader = DataLoader(chroma_raw_train, batch_size=32, shuffle=True, collate_fn=collate_fn)
chroma_raw_val_loader = DataLoader(chroma_raw_val, batch_size=32, collate_fn=collate_fn)
chroma_raw_test_loader = DataLoader(chroma_raw_test, batch_size=32, collate_fn=collate_fn)

beat_path = 'data/fma_genre_spectrograms_beat'

mel_beat_train_full = SpectrogramDataset(beat_path, read_spec_fn=read_mel_spectrogram, train=True, class_mapping=class_mapping)
mel_beat_train, mel_beat_val = split_dataset(mel_beat_train_full, train_size=0.8)
mel_beat_test = SpectrogramDataset(beat_path, read_spec_fn=read_mel_spectrogram, train=False, class_mapping=class_mapping)
mel_beat_train_loader = DataLoader(mel_beat_train, batch_size=32, shuffle=True, collate_fn=collate_fn)
mel_beat_val_loader = DataLoader(mel_beat_val, batch_size=32, collate_fn=collate_fn)
mel_beat_test_loader = DataLoader(mel_beat_test, batch_size=32, collate_fn=collate_fn)

chroma_beat_train_full = SpectrogramDataset(beat_path, read_spec_fn=read_chromagram, train=True, class_mapping=class_mapping)
chroma_beat_train, chroma_beat_val = split_dataset(chroma_beat_train_full, train_size=0.8)
chroma_beat_test = SpectrogramDataset(beat_path, read_spec_fn=read_chromagram, train=False, class_mapping=class_mapping)
chroma_beat_train_loader = DataLoader(chroma_beat_train, batch_size=32, shuffle=True, collate_fn=collate_fn)
chroma_beat_val_loader = DataLoader(chroma_beat_val, batch_size=32, collate_fn=collate_fn)
chroma_beat_test_loader = DataLoader(chroma_beat_test, batch_size=32, collate_fn=collate_fn)

# Getting the of labels in a python list
labels = mel_raw_train_full.labels
labels_str = mel_raw_train_full.labels_str


# %% STEP 0, 1, 2, 3
# In our example we chose Electronic music vs classical music.
# We see that the Electronic sample is more tightly structured in a disrete manner, while Classical sample is more fluid and continuous,
# and this holds both for the mel spectogram and the chromogram.
# Also, from the mel spectograms we see that the Electronic sample has harmonics over the entire frequency range,
# while the Classical sample does not. Finaly notice the regular vertical lines in the Electronic samples
# which are a result of a regular rhythm
#
# As we see, size of each raw sample is above 150,000 which is almost impossible to use for training on normal machines.
# On the other hand beat-synced samples have size of roughly 750, which is definitely something we can work with.

label1_str = 'Electronic'
label2_str = 'Classical'
label1 = labels_str.tolist().index(label1_str)
label2 = labels_str.tolist().index(label2_str)
index1 = labels.tolist().index(label1)
index2 = labels.tolist().index(label2)

for dataset, spec_type, transform in zip(
        (mel_raw_train_full, chroma_raw_train, mel_beat_train_full, chroma_beat_train_full),
        ('Mel frequencies', 'Chromagrams')*2,
        ('Raw',)*2 + ('Beat-Synced',)*2
    ):
    spec1 = dataset[index1][0].numpy()
    spec2 = dataset[index2][0].numpy()
    print(f'{spec_type} ({transform}) shape: {spec1.shape}')
    plot_spectograms(spec1.T, spec2.T, label1_str, label2_str, f'{spec_type} ({transform})')
    

# %% STEP 4
# As noted earlier I implemented the Datasets differently.
# Below I answer the questions asked in the original implementation.
# 
# QUESTION: Comment on howv the train and validation splits are created.
# ANSWER: We read the data in arrays, create an array of the indices,
#   we shuffle the indices, and then we split them.
# 
# QUESTION: It's useful to set the seed when debugging but when experimenting ALWAYS set seed=None. Why?
# ANSWER: Because we would always be training and validating on the same data,
#   which could make the model learn properties specific to that split
#   and which aren't properties of the entire set.
#
# QUESTION: Comment on why padding is needed
# ANSWER: Because PyTorch doesn't support ragged tensors.

# Create a dataset without using the class mapping, solely for computing the labels
# Note that constructing the dataset is cheap, since our implementation is lazy.
ds = SpectrogramDataset(raw_path, read_spec_fn=read_mel_spectrogram, train=True, class_mapping=None)
labels_str_original = ds.labels_str
labels_original = ds.labels

fig, axs = plt.subplots(ncols=2, figsize=(12, 8))
sns.histplot(labels_str_original[labels_original], bins=len(labels_str_original), ax=axs[0])
sns.histplot(labels_str[labels], bins=len(labels_str), ax=axs[1])
_ = plt.setp(axs[0].get_xticklabels(), rotation=45, ha='right')
_ = plt.setp(axs[1].get_xticklabels(), rotation=45, ha='right')
axs[0].set_title('Original Labels')
axs[1].set_title('Transformed Labels')


# %% STEP 5



