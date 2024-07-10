from typing import Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from nntomo.projection_stack import ProjectionStack
from nntomo.volume import Volume
from nntomo import DATA_FOLDER

class MSDNET(nn.Module):
    """width=1"""
    
    def __init__(self, depth: int = 100, max_dilation: int = 10) -> None:
        super().__init__()

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

        previous_features = [x]
        for conv, batch_norm in zip(self.conv_list, self.bach_norm_list):
            x = torch.stack(previous_features)
            x = nn.ReLU(batch_norm(conv(x)))
            previous_features.append(x)
        
        x = torch.stack(previous_features)
        return nn.Sigmoid(self.last_conv(x))



class DatasetMSDNET(Dataset):
    """_description_

    Args:
            proj_stacks (Union[ProjectionStack, list[ProjectionStack]]): The stack(s) of projections used to calculate he inputs.
            volumes (Union[Volume, list[Volume]]): The volume(s) representing the real voxel values (expected outputs) associated with proj_stacks.
            custom_id (str, optional): Identifiant for the dataset, used for the automatic generation of files names. If None, a default identifiant
                is given.
    """

    def __init__(self, proj_stacks: Union[ProjectionStack, list[ProjectionStack]], volumes: Union[Volume, list[Volume]], custom_id: str = None) -> None:
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
            self.id = f"msdn_{proj_stacks[0].id}_{volumes[0].id}"
        else:
            self.id = custom_id
        self.file_path = DATA_FOLDER / f"datasets_files/{self.id}.pickle"


        fbp_volume = np.concatenate([stack.get_FBP_reconstruction().volume for stack in proj_stacks], axis=0).astype(np.float32)
        volume = np.concatenate([vol.volume for vol in volumes], axis=0).astype(np.float32)

        fbp_volume = (fbp_volume-fbp_volume.min()) / (fbp_volume.max()-fbp_volume.min())
        volume = (volume-volume.min()) / (volume.max()-volume.min())
                
        self.inputs = torch.from_numpy(fbp_volume)
        self.outputs = torch.from_numpy(volume)
    
    def to(self, device: str) -> None:
        """Transfers inputs and ouputs to the specified device."""
        self.inputs = self.inputs.to(device)
        self.outputs = self.outputs.to(device)

    
    def __len__(self) -> None:
        return len(self.inputs)

    def __getitem__(self, index: int) -> None:
        return self.inputs[index], self.outputs[index]