from sklearn import datasets
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
from fastprogress.fastprogress import master_bar, progress_bar
from time import sleep

cancer = datasets.load_breast_cancer()
cancer_df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
cancer_df['type'] = cancer['target']

class CancerDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        cols = [col for col in dataframe.columns if col != 'target']
        self.parameters = torch.from_numpy(dataframe[cols].values).type(torch.float32)
        self.targets = torch.from_numpy(dataframe['type'].values).type(torch.float32)
        self.targets = self.targets.reshape((len(self.targets), 1))

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, idx):
        parameters = self.parameters[idx]
        target = self.targets[idx]
        return parameters, target

ds = CancerDataset(cancer_df)
train_ds, valid_ds = torch.utils.data.random_split(ds, [int(0.8*len(ds)), len(ds) - int(0.8*len(ds))], generator=torch.Generator().manual_seed(42))

BS_train = 64
BS_val = 128 # Solo hay 114 datos de validación, por lo que no se puede dividir en batches
train_dl = DataLoader(train_ds, batch_size=BS_train, shuffle=True)
val_dl = DataLoader(valid_ds, batch_size=BS_val, shuffle=False)

class CancerNeuralNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_layers=[100, 50, 20]):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, hidden_layers[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layers[0], hidden_layers[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layers[1], hidden_layers[2]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layers[2], num_outputs),
        )
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        logits = self.network(x)
        probs = self.activation(logits)
        return logits, probs

num_inputs = 31
num_outputs = 1
model = CancerNeuralNetwork(num_inputs, num_outputs)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
model.to(device)

LR = 1e-3

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

def plot_loss_update(epoch, epochs, mb, train_loss, valid_loss):
    """ dynamically print the loss plot during the training/validation loop.
        expects epoch to start from 1.
    """
    x = range(1, epoch+1)
    y = np.concatenate((train_loss, valid_loss))
    graphs = [[x,train_loss], [x,valid_loss]]
    x_margin = 0.2
    y_margin = 0.05
    x_bounds = [1-x_margin, epochs+x_margin]
    y_bounds = [np.min(y)-y_margin, np.max(y)+y_margin]

    mb.update_graph(graphs, x_bounds, y_bounds)

epochs = 14
mb = master_bar(range(1, epochs+1))
train_loss, valid_loss = [], []
for epoch in mb:
    # train loop
    model.train()
    for (X, y) in progress_bar(train_dl, parent=mb):
        # X and y to device
        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        logits, probs = model(X)
        loss = loss_fn(logits, y)
        mb.child.comment = f'train loss: {loss.item():>7f}'

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sleep(0.2)
    
    # validation loop
    num_batches = len(val_dl)
    val_loss, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for X, y in progress_bar(val_dl, parent=mb):
            # X and y to device
            X, y = X.to(device), y.to(device)
            
            logits, probs = model(X)
            val_loss += loss_fn(logits, y).item()
            correct += (probs.round() == y).type(torch.float).sum().item()
            mb.child.comment = f'val loss: {val_loss:>7f}, correct: {int(correct):03d}'

            sleep(0.2)

    train_loss.append(loss.item())
    valid_loss.append(val_loss)
    mb.main_bar.comment = f'epoch: {epoch}/{epochs}, train loss: {train_loss[-1]:>7f}, valid loss: {valid_loss[-1]:>7f}'
    mb.write(f'epoch: {epoch}, train loss: {train_loss[-1]:>7f}, valid loss: {valid_loss[-1]:>7f}, correct: {int(correct):>3d}')

    plot_loss_update(epoch, epochs, mb, train_loss, valid_loss)