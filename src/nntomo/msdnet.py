import pickle
from typing import Union, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from nntomo.projection_stack import ProjectionStack
from nntomo.volume import Volume
from nntomo import DATA_FOLDER

class MSDNET(nn.Module):
    """Implementation of the Mixed-Scale Dense Convolutional Neural Network used in [Improving Tomographic Reconstruction from Limited Data Using
    Mixed-Scale Dense Convolutional Neural Networks, Pelt and al., 2018] for tomographic reconstruction. The width of the network as defined in
    [A mixed-scale dense convolutional neural network for image analysis, Pelt and al., 2018] is always 1 in this implementation. See this article
    for the definition of depth and maximum dilation.

    Args:
        train_dataset (DatasetMSDNET): The dataset used for training.
        depth (int, optional): The depth of the network. Defaults to 100.
        max_dilation (int, optional): The maximum dilation. Defaults to 10.
    """

    def __init__(self, train_dataset: 'DatasetMSDNET', id: str, depth: int = 100, max_dilation: int = 10) -> None:
        super().__init__()
        
        self.Nth = train_dataset.Nth
        self.Nd = train_dataset.Nd
        self.angles_range = train_dataset.angles_range
        self.id = id

        self.depth = depth

        self.conv_list = nn.ModuleList()
        self.bach_norm_list = nn.ModuleList()

        for i in range(depth):
            self.conv_list.append(nn.Conv2d(
                in_channels = i+1,    
                out_channels = 1,
                kernel_size = 3,
                dilation = 1+(i%max_dilation),
                padding='same'))
            self.bach_norm_list.append(nn.BatchNorm2d(1))

        self.last_conv = nn.Conv2d(in_channels = depth + 1,    
                out_channels = 1,
                kernel_size = 1,
                padding='same')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        relu = nn.ReLU()
        sigm = nn.Sigmoid()
        previous_features = [x.view(x.size(0), 1, x.size(1), x.size(2))]

        for conv, batch_norm in zip(self.conv_list, self.bach_norm_list):
            x = torch.cat(previous_features, dim=1)
            x = relu(batch_norm(conv(x)))
            previous_features.append(x)
        
        x = torch.cat(previous_features, dim=1)
        return sigm(self.last_conv(x)).view(x.size(0), x.size(2), x.size(3))



class DatasetMSDNET(Dataset):
    """The pytorch Dataset class for MSDNET in tomographic reconstuction. A FBP reconstruction is computed with the given stack of projections; slices
    of this reconstruction are given as inputs for the network, while stacks of given reference volumes are given as outputs.

    Args:
            proj_stacks (Union[ProjectionStack, list[ProjectionStack]]): The stack(s) of projections used to calculate he inputs.
            volumes (Union[Volume, list[Volume]]): The volume(s) representing the real voxel values (expected outputs) associated with proj_stacks.
            custom_id (Optional[str], optional): Identifiant for the dataset, used for the automatic generation of files names. If None, a default identifiant
                is given.
    """

    def __init__(self, proj_stacks: Union[ProjectionStack, list[ProjectionStack]], volumes: Union[Volume, list[Volume]], custom_id: Optional[str] = None) -> None:
        ### Checking that proj_stacks and volumes are compatible ###
        if type(proj_stacks) is not list:
            proj_stacks = [proj_stacks]
        if type(volumes) is not list:
            volumes = [volumes]
        if len(proj_stacks) != len(volumes):
            raise ValueError("proj_stacks and volumes must be of the same size.")
        proj_shapes = [(proj_stack.shape[1], proj_stack.shape[2]) for proj_stack in proj_stacks]
        if len(set(proj_shapes)) > 1:
            raise ValueError("projection stacks should have same shapes along axis 1 and 2.")
        vol_shapes = [(volume.shape[1], volume.shape[2]) for volume in volumes]
        if len(set(vol_shapes)) > 1:
            raise ValueError("volumes should have same shapes along axis 1 and 2.")
        if len(set([stack.angles_range for stack in proj_stacks])) > 1:
            raise ValueError("projection stacks should have same angles_range.")
        
        ### Id and file_path in case we want to save the dataset ###
        if custom_id is None:
            self.id = f"msdn[{proj_stacks[0].id}][{volumes[0].id}]"
        else:
            self.id = custom_id
        self.file_path = DATA_FOLDER / f"datasets_files/{self.id}.pickle"

        self.Nth = proj_stacks[0].Nth
        self.Nd = proj_stacks[0].Nd
        self.angles_range = proj_stacks[0].angles_range

        fbp_volume = np.concatenate([stack.get_FBP_reconstruction().volume for stack in proj_stacks], axis=0).astype(np.float32)
        volume = np.concatenate([vol.volume for vol in volumes], axis=0).astype(np.float32)

        fbp_volume = (fbp_volume-fbp_volume.min()) / (fbp_volume.max()-fbp_volume.min())####
        volume = (volume-volume.min()) / (volume.max()-volume.min())####

        self.inputs = torch.from_numpy(fbp_volume)
        self.outputs = torch.from_numpy(volume)

    def save(self) -> None:
        """Saving of the dataset in the folder dataset_files."""
        print("Saving dataset...\r")
        with open(self.file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"File saved at {self.file_path}.\n ID: {self.id}")
    
    @classmethod
    def retrieve(clas, id: str) -> 'DatasetMSDNET':
        """Retrieves the dataset in the folder dataset_files, given an provided id.

        Args:
            id (str): The id of the dataset to retreive.

        Returns:
            DatasetSlices: The retrieved dataset.
        """
        file_path = DATA_FOLDER / f"datasets_files/{id}.pickle"
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset

    
    def __len__(self) -> None:
        return len(self.inputs)

    def __getitem__(self, index: int) -> None:
        return self.inputs[index], self.outputs[index]