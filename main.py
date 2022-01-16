# %%
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import torch
from torch import nn
from torch import optim
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence, PackedSequence
from torch.utils.data import Dataset, Subset, DataLoader, random_split

RANDOM_STATE = 42
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


def split_dataset(dataset, train_size, seed=RANDOM_STATE):
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

fused_raw_train_full = SpectrogramDataset(raw_path, read_spec_fn=read_fused_spectrogram, train=True, class_mapping=class_mapping)
fused_raw_train, fused_raw_val = split_dataset(fused_raw_train_full, train_size=0.8)
fused_raw_test = SpectrogramDataset(raw_path, read_spec_fn=read_fused_spectrogram, train=False, class_mapping=class_mapping)

mel_raw_train_full = SpectrogramDataset(raw_path, read_spec_fn=read_mel_spectrogram, train=True, class_mapping=class_mapping)
mel_raw_train, mel_raw_val = split_dataset(mel_raw_train_full, train_size=0.8)
mel_raw_test = SpectrogramDataset(raw_path, read_spec_fn=read_mel_spectrogram, train=False, class_mapping=class_mapping)

chroma_raw_train_full = SpectrogramDataset(raw_path, read_spec_fn=read_chromagram, train=True, class_mapping=class_mapping)
chroma_raw_train, chroma_raw_val = split_dataset(chroma_raw_train_full, train_size=0.8)
chroma_raw_test = SpectrogramDataset(raw_path, read_spec_fn=read_chromagram, train=False, class_mapping=class_mapping)

beat_path = 'data/fma_genre_spectrograms_beat'

fused_beat_train_full = SpectrogramDataset(beat_path, read_spec_fn=read_fused_spectrogram, train=True, class_mapping=class_mapping)
fused_beat_train, fused_beat_val = split_dataset(fused_beat_train_full, train_size=0.8)
fused_beat_test = SpectrogramDataset(beat_path, read_spec_fn=read_fused_spectrogram, train=False, class_mapping=class_mapping)

mel_beat_train_full = SpectrogramDataset(beat_path, read_spec_fn=read_mel_spectrogram, train=True, class_mapping=class_mapping)
mel_beat_train, mel_beat_val = split_dataset(mel_beat_train_full, train_size=0.8)
mel_beat_test = SpectrogramDataset(beat_path, read_spec_fn=read_mel_spectrogram, train=False, class_mapping=class_mapping)

chroma_beat_train_full = SpectrogramDataset(beat_path, read_spec_fn=read_chromagram, train=True, class_mapping=class_mapping)
chroma_beat_train, chroma_beat_val = split_dataset(chroma_beat_train_full, train_size=0.8)
chroma_beat_test = SpectrogramDataset(beat_path, read_spec_fn=read_chromagram, train=False, class_mapping=class_mapping)

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

def step_0_1_2_3():
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

def step4():
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
class CustomLSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size,
                 bidirectional=False, dropout=0.
                 ):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            bidirectional=bidirectional, batch_first=True)
        self.linear = nn.Linear(hidden_size * (bidirectional + 1), output_size)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, lengths):
        
        lstm_out, *_ = self.lstm(x)
        if isinstance(lstm_out, PackedSequence):
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        # Get the final outputs of each direction and concatenate them
        end_indices = (lengths - 1)[..., None, None].to(DEVICE)
        end1 = torch.take_along_dim(lstm_out[..., :self.lstm.hidden_size],
                                    end_indices,
                                    1
                                    ).squeeze()
        end2 = torch.take_along_dim(lstm_out[..., self.lstm.hidden_size:],
                                    end_indices,
                                    1
                                    ).squeeze()
        # If self.lstm.bidirectional, end2 is an empty tensor
        lstm_out = torch.cat((end1, end2), dim=-1)
    
        dropout_out = self.dropout(lstm_out)
        linear_out = self.linear(dropout_out)
        return linear_out


def train_loop(dataloader, model, loss_fn, optimizer, device=DEVICE):
    model.train()
    train_loss = 0.
    n_batches = len(dataloader)
    
    for x, y, lengths in dataloader:
        x, y = x.to(device), y.to(device)
        x = pack_padded_sequence(x, lengths, enforce_sorted=False, batch_first=True)
        
        # Compute prediction and loss
        pred = model(x, lengths)
        loss = loss_fn(pred, y)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss /= n_batches
    return train_loss


def test_loop(dataloader, model, loss_fn, device=DEVICE):
    model.eval()
    n_batches = len(dataloader)
    test_loss = 0
    test_accuracy = 0

    with torch.inference_mode():
        for x, y, lengths in dataloader:
            x, y = x.to(device), y.to(device)
            x = pack_padded_sequence(x, lengths, enforce_sorted=False, batch_first=True)
            probs = model(x, lengths)
            test_loss += loss_fn(probs, y).item()
            preds = torch.argmax(probs, 1)
            test_accuracy += (preds == y).float().mean().item()

    test_loss /= n_batches
    test_accuracy /= n_batches
    return test_loss, test_accuracy


# %%
def train_eval(model, train_dataset, val_dataset, batch_size,epochs,
               lr=1e-3, l2=1e-2, patience=5, tolerance=1e-4,
               save_path='best-model.pth', overfit_batch=False,
               ):
    
    
    if overfit_batch:
        # Create a subset of the dataset of size 3*batch_size and use this instead
        rng = np.random.default_rng(seed=RANDOM_STATE)
        indices = rng.choice(np.arange(len(train_dataset)), size=3*batch_size, replace=False)
        train_dataset = Subset(train_dataset, indices)
        # Increase the number of epochs appropriately
        # total = epochs * len(dataset)
        #       = epochs * n_batches * batch_size
        #       = epochs * n_batches * 3 * (batch_size/3)
        # Thus, to keep roughly same total we do:
        epochs *= (batch_size // 3) + 1
        # But we will use at most 200 epochs
        epochs = min(epochs, 200)
        print('Overfit Batch mode. The dataset now comprises of only 3 Batches.'
              f'Epochs increased to {epochs}.')
        
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn,
                              pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn,
                            pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('+infinity')
    waiting = 0

    for t in range(epochs):
        # Train and validate
        print(f'----EPOCH {t}----')
        train_loss = train_loop(train_loader, model, loss_fn, optimizer)
        print(f'Train Loss: {train_loss}')
        
        # Validating is not usefull in overfit_batch mode.
        # We also won't use the scheduler in over_fit batch mode
        # because the epoch numbers become too large.
        if not overfit_batch:
            val_loss, val_accuracy = test_loop(val_loader, model, loss_fn)
            print(f'Val Loss: {val_loss}')
            print(f'Val Accuracy: {val_accuracy}')
            
            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model, save_path)
                print('Saving')
                
            # Early Stopping
            if val_losses and val_losses[-1] - val_loss < tolerance:
                if waiting == patience:
                    print('Early Stopping')
                    break
                waiting += 1
                print(f'{waiting = }')
            else:
                waiting = 0
        
            scheduler.step()
        
        train_losses.append(train_loss)
        if not overfit_batch:
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
        print()
        
    return train_losses, val_losses, val_accuracies


def train_mel_raw():
    train_dataset = mel_raw_train
    val_dataset = mel_raw_val
    input_dim = train_dataset[0][0].shape[1]
    output_dim = len(labels_str)
    model = CustomLSTM(input_dim, 64, output_dim, bidirectional=True, dropout=0.2).to(DEVICE)

    losses = train_eval(model, train_dataset, val_dataset,
                    batch_size=64, epochs=2, lr=1e-4,
                    overfit_batch=False, save_path='best-mel-raw.pth')
    with open('losses-mel-raw.pkl', 'wb') as f:
        pickle.dump(losses, f)


def train_mel_beat():
    train_dataset = mel_beat_train
    val_dataset = mel_beat_val
    input_dim = train_dataset[0][0].shape[1]
    output_dim = len(labels_str)
    model = CustomLSTM(input_dim, 64, output_dim, bidirectional=True, dropout=0.2).to(DEVICE)

    losses = train_eval(model, train_dataset, val_dataset,
                    batch_size=64, epochs=2, lr=1e-4,
                    overfit_batch=False, save_path='best-mel-beat.pth')
    with open('losses-mel-beat.pkl', 'wb') as f:
        pickle.dump(losses, f)
    

def train_chroma_beat():
    train_dataset = chroma_beat_train
    val_dataset = chroma_beat_val
    input_dim = train_dataset[0][0].shape[1]
    output_dim = len(labels_str)
    model = CustomLSTM(input_dim, 64, output_dim, bidirectional=True, dropout=0.2).to(DEVICE)

    losses = train_eval(model, train_dataset, val_dataset,
                    batch_size=64, epochs=2, lr=1e-4,
                    overfit_batch=False, save_path='best-chroma-beat.pth')
    with open('losses-chroma-beat.pkl', 'wb') as f:
        pickle.dump(losses, f)


def train_fused_beat():
    train_dataset = fused_beat_train
    val_dataset = fused_beat_val
    input_dim = train_dataset[0][0].shape[1]
    output_dim = len(labels_str)
    model = CustomLSTM(input_dim, 64, output_dim, bidirectional=True, dropout=0.2).to(DEVICE)

    losses = train_eval(model, train_dataset, val_dataset,
                    batch_size=64, epochs=2, lr=1e-4,
                    overfit_batch=False, save_path='best-fused-beat.pth')
    with open('losses-fused-beat.pkl', 'wb') as f:
        pickle.dump(losses, f)
        

# %%
train_mel_raw()
# %%
train_mel_beat()
# %%
train_chroma_beat()
# %%
train_fused_beat()
