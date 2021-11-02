import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


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


class CustomDNN(nn.Module):

    def __init__(self, dim_in: int, dim_out: int, dim_hidden: tuple, activation: str):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden= dim_hidden
        self.dim_all = (dim_in,) + dim_hidden + (dim_out,)
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



class DNN(BaseEstimator, ClassifierMixin):

    def __init__(self, dim_in, dim_out, d_hidden, activation='relu', lr=1e-2, epochs=10, batch_size=64):
        # No initialization besides simply assigning the __init__ signatures to attributes of self.
        # All other initialization should go to the fit method.
        # Initializing self.model, self.criterion, self,optimizer here goes against best practices.
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.d_hidden = d_hidden
        self.activation = activation
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        # Fit is not supposed to do cross-validation
        # The train_loader, val_loader split recommend goes against best practices
        # train_loader: DataLoader = ...
        # val_loader: DataLoader = ...

        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.torch.int64)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        self.criterion_ = torch.nn.CrossEntropyLoss(reduction='mean')
        self.model_ = CustomDNN(self.dim_in, self.dim_out, self.d_hidden, self.activation)
        self.optimizer_ = torch.optim.SGD(self.model_.parameters(), lr=self.lr)
        self.losses_ = []

        for epoch in range(self.epochs):
            running_loss = 0.

            for i, (inputs, labels) in enumerate(dataloader):
                outputs = self.model_(inputs)
                loss = self.criterion_(outputs, labels)
                self.optimizer_.zero_grad()
                loss.backward()
                self.optimizer_.step()

                running_loss += loss.item() / len(dataloader)
            self.losses_.append(running_loss)

        self.losses_ = np.array(self.losses_)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor)
            probs = torch.nn.Softmax(dim=1)(logits)
        return probs.numpy()

    def predict(self, X):
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


def read_data(file, delimiter=None):
    data = np.loadtxt(file, delimiter=delimiter)
    X = data[:, 1:]
    y = data[:, 0]
    return X, y


X_train, y_train = read_data('data/train.txt')
X_test, y_test = read_data('data/test.txt')


# train_path = 'data/train.txt'
# test_path = 'data/test.txt'
# train_loader = DataLoader(MnistDataset(train_path), batch_size=64, shuffle=True)
# test_loader = DataLoader(MnistDataset(test_path), batch_size=64, shuffle=True)


model = DNN(256, 10, (64,))
model.fit(X_train, y_train)
score = model.score(X_test, y_test)





