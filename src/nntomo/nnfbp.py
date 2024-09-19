import math
import pickle
from typing import Union, Optional

import cupy as cp
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from nntomo.custom_interp import custom_interp_dataset_init
from nntomo.projection_stack import ProjectionStack
from nntomo.volume import Volume
from nntomo.utilities import empty_cached_gpu_memory
from nntomo import DATA_FOLDER, GPU_MEM_LIMIT


class NNFBP(nn.Module):
    """General neural network structure from Pelt and al. (2013), for 2D images. For each pixel, the input is a 1D vector z (see article for its
    definition), and the output is the reconstructed pixel. The input vector might have undergone exponential binning beforehand. Can be used for
    3D objects too, either by considering each slice independently for the input, or by applying a 3D shift and a 2D exponential binning. 

    Args:
        train_dataset (DatasetNNFBP): The dataset used for training.
        Nh (int): The number of hidden nodes.
        id (str): An identifiant for the network, used for the automatic generation of files names.
        activation (str, optional): The activation function of the hidden layer, either 'relu' or 'sigmoid'. Default to relu'.
    """

    def __init__(self, train_dataset: 'DatasetNNFBP', Nh: int, id: str, activation: str = 'relu') -> None:
        super().__init__()

        if activation not in ['relu', 'sigmoid']:
            raise ValueError(f"activation must have value 'relu' or 'sigmoid' but has value {activation}")

        self.Nh = Nh
        self.id = id

        self.raw_input_size = train_dataset.raw_input_size
        self.binning = train_dataset.binning
        if self.binning:
            self.inputs_per_bin = train_dataset.inputs_per_bin
            self.bin_indexes = train_dataset.bin_indexes
        self.Nth = train_dataset.Nth
        self.Nd = train_dataset.Nd
        self.a = train_dataset.a
        self.b = train_dataset.b

        self.W = nn.Linear(train_dataset.input_size, Nh)


        if activation == 'relu':
            act_func = nn.ReLU()
        elif activation == 'sigmoid':
            act_func = nn.Sigmoid()
        self.linear_sigm_stack = nn.Sequential(
            nn.BatchNorm1d(Nh),
            act_func,
            nn.Linear(Nh, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_sigm_stack(self.W(x))
        return x.view(x.size(0))
    
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
        else:
            return weights





class DatasetNNFBP(Dataset):
    """A Dataset object commonly used by torch neural networks: inputs and expected outputs of the NN are processed here. To access the input
    and expected output n°i, use dataset[i] where dataset is a DatasetSlices object. For 3D reconstruction. Each layer is considered independently
    to apply 2D NN-FBP. See [Pelt 2013] for precisions about the NN inputs computations. n_inputs pixels are chosen randomly among the provided volume
    and the associated NN inputs are computed with the provided projections.

    Args:
            proj_stacks (Union[ProjectionStack, list[ProjectionStack]]): The stack(s) of projections used to calculate he inputs.
            volumes (Union[Volume, list[Volume]]): The volume(s) representing the real voxel values (expected outputs) associated with proj_stacks.
            binning (bool, optional): Whether or not exponential binning is applied to the inputs. See [Pelt 2013] for precisions. Defaults to True.
            n_input (int, optional): The number of random voxels in the volume(s) for which the NN inputs are calculated, and as such, the number of
                inputs and outputs in the dataset. Defaults to 100_000.
            custom_id (Optional[str], optional): Identifiant for the dataset, used for the automatic generation of files names. If None, a default identifiant
                is given.
    """

    @empty_cached_gpu_memory
    def __init__(self, proj_stacks: Union[ProjectionStack, list[ProjectionStack]], volumes: Union[Volume, list[Volume]], binning: bool = True,
                 n_input: int = 100_000, custom_id: Optional[str] = None) -> None:

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
            self.id = f"nnfbp[{proj_stacks[0].id}][{volumes[0].id}]{binning*'bin'}"
        else:
            self.id = custom_id
        self.file_path = DATA_FOLDER / f"datasets_files/{self.id}.pickle"



        ### All projections and volumes are stacked along the z-axis for computation convenience. Voxels values are set between 0 and 1 ###
        proj_stack = np.concatenate([stack.interp_stack for stack in proj_stacks], axis=0).astype(np.float32)
        volume = np.concatenate([vol.volume for vol in volumes], axis=0).astype(np.float32)
        
        proj_stack = np.transpose(proj_stack, (1,0,2)) # (Nz, Nth, Nd) -> (Nth, Nz, Nd)
        proj_stack = cp.asarray(proj_stack, dtype = cp.float32)
        
        self.a = 1/(volume.max()-volume.min())
        self.b = - volume.min()*self.a
        volume = self.a*volume + self.b


        ### Attributes of the dataset ###
        self.binning = binning
        Nz_training, self.Ny, self.Nx = volume.shape # astra convention for volumes: (Nz, Ny, Nx)
        self.Nth, _, self.Nd = proj_stack.shape
        self.raw_input_size = 2*self.Nd-1 # size of inputs before binning
        self.tilt_angles = np.linspace(-np.pi/2, np.pi/2, self.Nth, endpoint=False, dtype = np.float32)

        # horizontal positions of the detectors
        T = cp.linspace(-(self.Nd-1)/2, (self.Nd-1)/2, self.Nd, dtype = cp.float32) 

        # horizontal positions of the detectors for the inputs
        big_T = cp.linspace(-(self.raw_input_size-1)/2, (self.raw_input_size-1)/2, self.raw_input_size, dtype = cp.float32) 

        if binning:
            self.input_size = 1 + math.ceil(math.log2(self.Nd)) # size of inputs after binning
            self.bin_indexes = np.ceil(np.log2(1 + np.abs(big_T.get()))).astype(np.int32) # example: [3,3,3,3,2,2,1,0,1,2,2,3,3,3,3]
        else:
            self.input_size = self.raw_input_size



        ### The inputs and inputs of the dataset before binning ###
        raw_inputs = []
        outputs = np.zeros(n_input, dtype = np.float32)



        ### The randomly chosen voxels in the volume ###
        rng = np.random.default_rng()
        random_voxels = cp.asarray(rng.integers([Nz_training, self.Ny, self.Nx], size = (n_input, 3)), dtype = cp.int32)



        ### Reshapes for numpy/cupy computations ###
        tilt_angles_ax2 = cp.asarray(self.tilt_angles.reshape(1, self.Nth, 1))
        big_T_ax3 = big_T.reshape(1, 1, self.raw_input_size)



        ### Batch voxel computation. This is the number of voxels for which the NN inputs are calculated in parallel using the GPU. The computation ###
        ### is done so that the GPU memory use doesn't pass a certain limit. ###
        batch_voxel = int(GPU_MEM_LIMIT*2**30/4/self.Nth/self.raw_input_size/4)


        ### For each voxel, the raw input is computed (equation (18) in [Pelt 2013]). For efficiency, the computation is done with batch_voxels voxels ###
        ### at a time. The equation require an 1D interpolation of the projection data for each tilt angle. Thus, in the code, this 1D interpolaion is ###
        ### computed for all tilt_angles and all voxels of the batch at the same time using the GPU and a custom kernel. ###
        voxel_index = 0
        for batch_index in range(math.ceil(n_input/batch_voxel)):

            voxels = random_voxels[batch_index*batch_voxel : min(n_input, (1+batch_index)*batch_voxel)]
            n = len(voxels)

            X = voxels[:, 2].reshape(n, 1, 1) - (self.Nx-1)/2
            Y = voxels[:, 1].reshape(n, 1, 1) - (self.Ny-1)/2
            UU = (X*cp.cos(tilt_angles_ax2) + Y*cp.sin(tilt_angles_ax2) - big_T_ax3).astype(cp.float32)
            
            pre_input = custom_interp_dataset_init(UU, voxels[:, 0], T, proj_stack, Nz_training)

            raw_inputs.append(cp.sum(pre_input, axis = 1))

            for i,j,k in voxels.get():
                outputs[voxel_index] = volume[i,j,k]
                voxel_index += 1
        
        
        ### Binning transformation of the inputs ###
        raw_inputs = cp.concatenate(raw_inputs)
        if binning:
            self.inputs_per_bin = np.zeros(self.input_size, dtype=np.int32)
            inputs = cp.zeros((n_input, self.input_size), dtype = cp.float32)
            for i in range(self.raw_input_size):
                inputs[:,self.bin_indexes[i]] += raw_inputs[:,i]
                self.inputs_per_bin[self.bin_indexes[i]] += 1
            for j in range(self.input_size):
                inputs[:,j] /= self.inputs_per_bin[j] # not in the article, average of all inputs of one bin. 
        else:
            inputs = raw_inputs
                
        self.inputs = torch.as_tensor(inputs)
        self.outputs = torch.from_numpy(outputs)
    
    def to(self, device: str) -> None:
        """Transfers inputs and ouputs to the specified device."""
        self.inputs = self.inputs.to(device)
        self.outputs = self.outputs.to(device)

    def save(self) -> None:
        """Saving of the dataset in the folder dataset_files."""
        print("Saving dataset...\r")
        with open(self.file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"File saved at {self.file_path}.\n ID: {self.id}")
    
    @classmethod
    def retrieve(clas, id: str) -> 'DatasetNNFBP':
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