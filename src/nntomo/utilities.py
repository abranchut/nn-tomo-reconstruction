import gc
import time
from typing import Generator, Iterable

import cupy as cp
import torch
import astra
from numba import cuda
from numba.cuda.cudadrv import enums

def empty_cached_gpu_memory(func) -> None:
    """Suppress the unused cache memory that cupy, pytorch, astra are keeping for themselves."""
    def inner(*args, **kwargs):
        result = func(*args, **kwargs)
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()
        astra.algorithm.clear()
        astra.data2d.clear()
        astra.data3d.clear()
        astra.functions.clear()
        astra.matrix.clear()
        astra.projector.clear()
        return result
    return inner


def optimizer_to(optim: torch.optim.Optimizer, device: str) -> None:
    """Code from https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/2. Transfer an optimizer to a device.

    Args:
        optim (torch.optim.Optimizer): The optimizer to transfer.
        device (str): The device where the optimizer should be transfered.
    """

    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def progressbar(it: Iterable, prefix: str = "", size:int = 60) -> Generator:
    """Code from https://stackoverflow.com/questions/3160699/python-progress-bar. Modify an iterable so that a progressbar is displayed, which shows
    how much it has been explored yet.
    Args:
        it (Iterable): The iterable on which a progressbar will be displayed.
        prefix (str, optional): An optional prefix to the progressbar. Defaults to "".
        size (int, optional): The size of the font. Defaults to 60.

    Yields:
        Generator: A generator of the iterable. A progressbar is also displayed.
    """
    count = len(it)
    start = time.time() # time estimate start
    def show(j):
        x = int(size*j/count)
        # time estimate calculation and string
        remaining = ((time.time() - start) / j) * (count - j)        
        mins, sec = divmod(remaining, 60) # limited to minutes
        time_str = f"{int(mins):02}:{sec:03.1f}"
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count} Est wait {time_str}", end='\r', flush=True)
    show(0.1) # avoid div/0 
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True)

@empty_cached_gpu_memory
def get_MSE_loss(original_volume: 'Volume', reconstructed_volume: 'Volume', only_inscribed_disc: bool = False) -> float: # type: ignore  # noqa: F821
    """Computes the Mean Square Error between two volumes.

    Args:
        original_volume (Volume): The reference volume.
        reconstructed_volume (Volume): The volume for comparison.
        only_inscribed_disc (bool, optional): Whether or not to compute the MSE only for voxels within the centered disc of radius Nd/2. The disc is
        considered in the plane of the second and third axis of the volume (axis 1 and 2) to follow the ASTRA convention. Default to False.

    Returns:
        float: The MSE.
    """
    if original_volume.shape != reconstructed_volume.shape:
        raise ValueError(f"Volumes should have the same shapes but have shapes {original_volume.shape} and {reconstructed_volume.shape}.")

    cp_original_volume = cp.asarray(original_volume.volume)
    cp_reconstructed_volume = cp.asarray(reconstructed_volume.volume)

    Nz, Ny, Nx = original_volume.shape
    if not only_inscribed_disc:
        if Nx != Ny:
            raise ValueError(f"Volumes should have same shapes along axis 1 and 2 but have shapes {Nx} and {Ny}.")
        X = cp.arange(Nx, dtype=cp.float16) - Nx/2
        Y = cp.arange(Ny, dtype=cp.float16) - Ny/2
        XX, YY = cp.meshgrid(X, Y)
        XX = XX.reshape((1, Ny, Nx))
        YY = YY.reshape((1, Ny, Nx))
        cp_original_volume = cp.where(XX**2 + YY**2 < (Nx/2)**2, cp_original_volume, 0)
        cp_reconstructed_volume = cp.where(XX**2 + YY**2 < (Nx/2)**2, cp_reconstructed_volume, 0)

        n_voxels = cp.sum(cp.where(XX**2 + YY**2 < (Nx/2)**2, 1, 0))*Nz
    else:
        n_voxels = Nx*Ny*Nz

    return float((cp.sum(cp.square(cp_reconstructed_volume - cp_original_volume))/n_voxels))


def get_cuda_caracteristics() -> None:
    """Print caracteristics of the GPU."""
    device = cuda.get_current_device()
    attribs= [name.replace("CU_DEVICE_ATTRIBUTE_", "") for name in dir(enums) if name.startswith("CU_DEVICE_ATTRIBUTE_")]
    for attr in attribs:
        print(attr, '=', getattr(device, attr))