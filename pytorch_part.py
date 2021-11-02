import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class MnistDataset(Dataset):

    def __init__(self, csv_file, delimiter=None, transform=None, target_transform=None):
        data = np.loadtxt(csv_file, delimiter=delimiter, dtype=np.single)
        self.images = data[:, 1:]
        self.labels = data[:, 0]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label


train_path = 'data/train.txt'
test_path = 'data/test.txt'
train_loader = DataLoader(MnistDataset(train_path), batch_size=64, shuffle=True)
test_loader = DataLoader(MnistDataset(test_path), batch_size=64, shuffle=True)

# %%


class CustomDNN(nn.Module):

    def __init__(self, dim_in, dim_out, d_hidden=(64, 32), activation='relu'):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden= d_hidden
        self.dim_all = (dim_in,) + d_hidden + (dim_out,)
        self.activation = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}[activation.lower()]()
        self._module_template = 'Layer_{}'

        for i in range(1, len(self.dim_all)):
            self.add_module(self._module_template.format(i), nn.Linear(self.dim_all[i-1], self.dim_all[i]))

    def forward(self, x):
        for i in range(1, len(self.dim_all)):
            layer = self.get_submodule(self._module_template.format(i))
            x = layer(x)
            x = self.activation(x)
        return x  # logits


model = CustomDNN(256, 10)
loss = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for image_batch, label_batch in train_loader:
    res = model(image_batch)
    break





