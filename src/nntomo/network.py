import os
import time
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from nntomo.dataset_slices import DatasetSlices
from nntomo import DATA_FOLDER


class NNFBP(nn.Module):
    """General neural network structure from Pelt and al. (2013), for 2D images. For each pixel, the input is a 1D vector z (see article for its
    definition), and the output is the reconstructed pixel. The input vector might have undergone exponential binning beforehand. Can be used for
    3D objects too, either by considering each slice independently for the input, or by applying a 3D shift and a 2D exponential binning. 

    Args:
        train_dataset (DatasetSlices): The dataset used for training.
        Nh (int): The number of hidden nodes.
        id (str): An identifiant for the network, used for the automatic generation of files names.
    """

    def __init__(self, train_dataset: DatasetSlices, Nh: int, id: str) -> None:
        super().__init__()

        self.Nh = Nh
        self.id = id

        self.raw_input_size = train_dataset.raw_input_size
        self.inputs_per_bin = train_dataset.inputs_per_bin
        self.bin_indexes = train_dataset.bin_indexes
        self.binning = train_dataset.binning
        self.Nth = train_dataset.Nth
        self.Nd = train_dataset.Nd

        self.W = nn.Linear(train_dataset.input_size, Nh)

        self.linear_sigm_stack = nn.Sequential(
            nn.BatchNorm1d(Nh),
            nn.ReLU(),
            #nn.Sigmoid(),
            nn.Linear(Nh, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_sigm_stack(self.W(x))
    
    def end_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for the end part of the network, used in the NN reconstruction phase, where the input computations and the first forward
        step (with weights W) are combined for efficiency by doing Nh filtered backprojections with custom weights W, and then applying this function.

        Args:
            x (torch.Tensor): A tensor of dimension 3+, coresponding to a stack (or several stacks) of FBP with custom weights.

        Returns:
            torch.Tensor: The output(s) of the network.
        """
        original_shape = x.shape
        x = torch.flatten(x, end_dim=-2)
        return self.linear_sigm_stack(x).view(original_shape[:-1])
    
    def get_FBP_weights(self) -> np.array:
        """Returns the weights for the custom FBP in the reconstruction process. If exponential binning was performed, unbins the weights."""
        weights = self.W.weight.numpy()
        if self.binning:
            unbinned_weights = np.zeros((self.Nh, self.raw_input_size), dtype=np.float32)
            for i in range(self.raw_input_size):
                unbinned_weights[:,i] = weights[:,self.bin_indexes[i]]/self.inputs_per_bin[self.bin_indexes[i]] # Divide to compensate the averaging.
        
        return unbinned_weights


def train_loop(dataloader, model, loss_fn, optimizer):
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    for X, y in dataloader:
        # Compute prediction and loss
        pred = model(X).view(X.shape[0])
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X).view(X.shape[0])
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Avg {loss_fn}: {test_loss:>8f} \n")
    return test_loss


def model_training(train_dataset: DatasetSlices, valid_dataset: DatasetSlices, Nh: int, batch_size: int = 32, learning_rate: float = 1e-4,
                   Nstop: int = 25, max_epoch: int = None, custom_id: str = None) -> NNFBP:
    """Training procedure, with data saves every 30s. The training is done on CPU. Notably, The data saved contains an historic of the
    networks and losses during the training.

    Args:
        train_dataset (DatasetSlices): Dataset used for training.
        valid_dataset (DatasetSlices): Dataset used for validation.
        Nh (int): Number of hidden nodes in the neural network.
        batch_size (int, optional): How many sample per batch to load. Defaults to 32.
        learning_rate (float, optional): Learning rate for Adam optimizer. Defaults to 1e-4.
        Nstop (int, optional): Number of epoch with no performance improvement before stopping the training. Defaults to 25.
        max_epoch (int, optional): Optionnal, a maximum number of epochs before forcing the training to stop. Defaults to None.
        custom_id (str, optional): Identifiant for the network, used for the automatic generation of files names. If None, a default identifiant
            is given.

    Returns:
        NNFBP: The best neural network.
    """

    if custom_id is None:
        network_id = f"{train_dataset.id}_{Nh}h"
    else:
        network_id = custom_id
    file_path = DATA_FOLDER / f"nn_models/{network_id}.tar"

    model = NNFBP(train_dataset, Nh, network_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Faster than SGD

    if not os.path.isfile(file_path):
        checkpoint = {
            'is_done': False,
            'epoch': 1,
            'n': 0,
            'min_loss': None,
            'model_states_history': [deepcopy(model.state_dict())],
            'optim_state': optimizer.state_dict(), #TODO to test!!!!
            'best_model': None}
        print("Start of training.")
    else:
        checkpoint = torch.load(file_path)
        if checkpoint['is_done']:
            return checkpoint['best_model']
        print("Resume of training.")

    ttemp = time.perf_counter()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    model.load_state_dict(deepcopy(checkpoint['model_states_history'][-1]))
    optimizer.load_state_dict(checkpoint['optim_state'])
    loss_fn = nn.MSELoss()

    while checkpoint['n'] < Nstop:
        print(f"Epoch {checkpoint['epoch']} (n={checkpoint['n']})\n-------------------------------")

        train_loop(train_dataloader, model, loss_fn, optimizer)
        loss = test_loop(valid_dataloader, model, loss_fn)
        checkpoint['model_states_history'].append(deepcopy(model.state_dict()))

        if checkpoint['min_loss'] is None or loss < checkpoint['min_loss'] :
            checkpoint['min_loss']  = loss
            checkpoint['best_model'] = deepcopy(model)
            checkpoint['n'] = 0
        else:
            checkpoint['n'] += 1
        checkpoint['epoch'] += 1

        if max_epoch is not None and checkpoint['epoch'] > max_epoch:
            break

        if time.perf_counter() - ttemp > 30 : # save data every 30s
            checkpoint['optim_state'] = optimizer.state_dict()
            torch.save(checkpoint, file_path)
            print("Data saved.")
            ttemp = time.perf_counter()

    checkpoint['is_done'] = True
    torch.save(checkpoint, file_path)

    print("End of training.")

    return checkpoint['best_model']


def edit_model(network_id, dict):
    file_path = DATA_FOLDER / f"nn_models/{network_id}.tar"
    if not os.path.isfile(file_path):
        raise ValueError(f"This model ({network_id}) doesn't exists.")
    else:
        checkpoint = torch.load(file_path)
        for key, value in dict.items():
            checkpoint[key] = value
        torch.save(checkpoint, file_path)

def get_model_infos(network_id):
    file_path = DATA_FOLDER / f"nn_models/{network_id}.tar"
    if not os.path.isfile(file_path):
        raise ValueError(f"This model ({network_id}) doesn't exists.")
    else:
        return torch.load(file_path)