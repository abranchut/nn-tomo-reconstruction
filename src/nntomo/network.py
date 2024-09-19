import os
import time
from copy import deepcopy
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from nntomo.nnfbp import NNFBP, DatasetNNFBP
from nntomo.msdnet import MSDNET, DatasetMSDNET
from nntomo.utilities import optimizer_to, get_MSE_loss, progressbar, empty_cached_gpu_memory
from nntomo import DATA_FOLDER


def train_loop(dataloader, model, loss_fn, optimizer, device):
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test_loop(dataloader, model, loss_fn, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    return test_loss

@empty_cached_gpu_memory
def model_training(model: nn.Module, train_dataset: Dataset, valid_dataset: Dataset, device: str, batch_size: int, learning_rate: float, Nstop: int,
                   max_epoch: int, network_id: str) -> nn.Module:
    """Training procedure, with data saves every 30s. Notably, The data saved contains an historic of the networks and losses during the training.

    Args:
        model (nn.Module): The network to train.
        train_dataset (Dataset): Dataset used for training.
        valid_dataset (Dataset): Dataset used for validation.
        device (str): The device ('cpu' or 'cuda:0') where the training is done.
        batch_size (int): How many sample per batch to load.
        learning_rate (float): Learning rate for Adam optimizer.
        Nstop (int): Number of epoch with no performance improvement before stopping the training.
        max_epoch (int): A maximum number of epochs before forcing the training to stop, or None (no maximum).
        network_id (str): Identifiant for the network, used for the automatic generation of files names.
    Returns:
        nn.Module: The best neural network.
    """

    
    file_path = DATA_FOLDER / f"nn_models/{network_id}.tar"

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Faster than SGD

    if not os.path.isfile(file_path):
        checkpoint = {
            'is_done': False,
            'epoch': 1,
            'n': 0,
            'min_loss': 1.,
            'model_states_history': [deepcopy(model.state_dict())],
            'optim_state': optimizer.state_dict(),
            'best_model': None}
    else:
        checkpoint = torch.load(file_path)
        if checkpoint['is_done']:
            return checkpoint['best_model']

    ttemp = time.perf_counter()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    model.load_state_dict(deepcopy(checkpoint['model_states_history'][-1]))
    optimizer.load_state_dict(checkpoint['optim_state'])
    loss_fn = nn.MSELoss(reduction='sum')

    model.to(device)
    optimizer_to(optimizer, device)

    loss = 1.
    while checkpoint['n'] < Nstop:
        print(f"Epoch {checkpoint['epoch']} (n={checkpoint['n']}) | Best avg {loss_fn}: {checkpoint['min_loss']:>8f} | "
              f"Last avg {loss_fn}: {loss:>8f}  ", end='\r')

        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        loss = test_loop(valid_dataloader, model, loss_fn, device)
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
            ttemp = time.perf_counter()

    model.to('cpu')

    print(f"Epoch {checkpoint['epoch']} (n={checkpoint['n']}) | Best avg {loss_fn}: {checkpoint['min_loss']:>8f} | "
          "End of training                                 ")

    checkpoint['is_done'] = True
    torch.save(checkpoint, file_path)

    return checkpoint['best_model']

def nnfbp_training(train_dataset: DatasetNNFBP, valid_dataset: DatasetNNFBP, Nh: int = 8, batch_size: int = 32, learning_rate: float = 1e-4,
                   Nstop: int = 25, max_epoch: Optional[int] = None, custom_id: Optional[str] = None, activation: str = 'relu') -> NNFBP:
    """Training procedure, with data saves every 30s. The training is done on CPU. Notably, The data saved contains an historic of the
    networks and losses during the training.

    Args:
        train_dataset (DatasetNNFBP): Dataset used for training.
        valid_dataset (DatasetNNFBP): Dataset used for validation.
        Nh (int): Number of hidden nodes in the neural network. Defaults to 8.
        batch_size (int, optional): How many sample per batch to load. Defaults to 32.
        learning_rate (float, optional): Learning rate for Adam optimizer. Defaults to 1e-4.
        Nstop (int, optional): Number of epoch with no performance improvement before stopping the training. Defaults to 25.
        max_epoch (Optional[int], optional): Optionnal, a maximum number of epochs before forcing the training to stop. Defaults to None.
        custom_id (Optional[str], optional): Identifiant for the network, used for the automatic generation of files names. If None, a default identifiant
            is given.
        activation (str, optional): The activation function of the hidden layer, either 'relu' or 'sigmoid'. Default to relu'.

    Returns:
        NNFBP: The best neural network.
    """
    if custom_id is None:
        network_id = f"{train_dataset.id}{Nh}h"
    else:
        network_id = custom_id
    
    model = NNFBP(train_dataset, Nh, network_id, activation)

    return model_training(model, train_dataset, valid_dataset, 'cpu', batch_size, learning_rate, Nstop, max_epoch, network_id)



def msdnet_training(train_dataset: DatasetMSDNET, valid_dataset: DatasetMSDNET, depth: int = 100, batch_size: int = 1, learning_rate: float = 1e-4,
                   Nstop: int = 25, max_epoch: Optional[int] = None, custom_id: Optional[str] = None) -> MSDNET:
    """Training procedure, with data saves every 30s. The training is done on CPU. Notably, The data saved contains an historic of the
    networks and losses during the training.

    Args:
        train_dataset (DatasetMSDNET): Dataset used for training.
        valid_dataset (DatasetMSDNET): Dataset used for validation.
        depth (int): Depth of the network. Defaults to 100.
        batch_size (int, optional): How many sample per batch to load. Defaults to 32.
        learning_rate (float, optional): Learning rate for Adam optimizer. Defaults to 1e-4.
        Nstop (int, optional): Number of epoch with no performance improvement before stopping the training. Defaults to 25.
        max_epoch (Optional[int], optional): Optionnal, a maximum number of epochs before forcing the training to stop. Defaults to None.
        custom_id (Optional[str], optional): Identifiant for the network, used for the automatic generation of files names. If None, a default identifiant
            is given.
    Returns:
        MSDNET: The best neural network.
    """
    if custom_id is None:
        network_id = f"{train_dataset.id}"
    else:
        network_id = custom_id
    
    model = MSDNET(train_dataset, depth)

    return model_training(model, train_dataset, valid_dataset, 'cuda:0', batch_size, learning_rate, Nstop, max_epoch, network_id)


def get_MSE_evolution(network_id: str, proj_stack: 'ProjectionStack', ref_volume: 'Volume', threshold: Optional[int] = None) -> list[float]: # type: ignore  # noqa: F821
    """Computes the MSE between a reference volume (ref_volume) and a network reconstruction from a projection stack (network_id, proj_stack), at
    each training epoch.

    Args:
        network_id (str): The id of the network training.
        proj_stack (ProjectionStack): The stack of projection to compute the reconstruction for each network.
        ref_volume (Volume): The reference volume.
        threshold (Optional[int], optional): A threshold for segmentation of the NN reconstuctions. Defaults to None (no thresholding).

    Returns:
        list[float]: The list of computed MSE. 
    """
    checkpoint = get_model_infos(network_id)
    model = checkpoint['best_model']
    mse_list = []
    for state in progressbar(checkpoint['model_states_history'], "MSE computations:"):
        model.load_state_dict(state)
        if type(model) is NNFBP:
            nn_reconstruction = proj_stack.get_NNFBP_reconstruction(model, show_progressbar = False).get_segmented_volume(threshold)
        elif type(model) is MSDNET:
            nn_reconstruction = proj_stack.get_MSDNET_reconstruction(model, show_progressbar = False).get_segmented_volume(threshold)
        mse_list.append(get_MSE_loss(ref_volume, nn_reconstruction))
    print('')
    return mse_list


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