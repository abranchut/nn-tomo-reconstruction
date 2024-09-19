from typing import Optional
import random

import numpy as np
import cupy as cp
import mrcfile

from nntomo.utilities import progressbar, empty_cached_gpu_memory
from nntomo import DATA_FOLDER


class Volume:
    """A class representing a 3d volume, extension of a 3d numpy array. Stores an id and the file_path of the associated .mrc file.
    
    Args:
        volume (np.ndarray): The volume as a numpy array.
        id (str): Identifiant for the volume, used for the automatic generation of files names.
    """

    def __init__(self, volume: np.ndarray, id: str) -> None:

        self.id = id
        self.file_path = DATA_FOLDER / f"volume_files/{id}.mrc"

        self.volume = volume.astype(np.float32)
        self.shape = volume.shape


    @classmethod
    def from_mrc_file(cls, volume_file: str, custom_id: Optional[str] = None) -> 'Volume':
        """Creation of a volume object, provided an .mrc file.

        Args:
            volume_file (str): The .mrc file path which represents a volume.
            custom_id (Optional[str], optional): Custom identifiant for the volume, used for the automatic generation of files names. If None, a default
                identifiant is given.

        Returns:
            Volume: The volume object.
        """
        with mrcfile.open(volume_file, permissive=True) as mrc:
            volume = mrc.data

        if custom_id is None :
            id = volume_file.split('.')[0].split('/')[-1]
        else:
            id = custom_id
        
        return cls(volume, id)
    

    @classmethod
    @empty_cached_gpu_memory
    def random_spheres(cls, nb_spheres: int, shape: int = 512, radius_range: tuple[int, int] = (5, 50), padding: int = 25) -> 'Volume':
        """Generates a volume object with random spheres in a cube with random intensities. The voxel values of the output volume are between
        0 and 1.

        Args:
            nb_spheres (int): The number of spheres.
            shape (int, optional): The width of the cube. Defaults to 512 voxels.
            radius_range (tuple[int, int], optional): The range of values of the randomly choosen radius. Default to (5, 100) (voxels).
            padding (int, optional): The distance limit of the randomly choosen centers from the sides of the box. Default to 50 voxels.

        Returns:
            Volume: The volume object.
        """

        X = cp.arange(shape, dtype=cp.float16)
        Y = cp.arange(shape, dtype=cp.float16)
        Z = cp.arange(shape, dtype=cp.float16)
        XX, YY, ZZ = cp.meshgrid(X, Y, Z)
        volume = cp.zeros((shape,shape,shape), dtype=cp.float16)

        for _ in progressbar(range(nb_spheres), "Generation of the spheres: "):
            radius = cp.random.randint(*radius_range).astype(cp.float16)
            x0, y0, z0 = cp.random.randint(padding, shape-padding, size=3).astype(cp.float16)
            intensity = random.random()
            volume = volume + cp.where(cp.sqrt((XX-x0)**2 + (YY-y0)**2 + (ZZ-z0)**2, dtype = cp.float16) <= radius, intensity, 0)

        result = cls(volume.get(), f"randspheres{shape}")
        result.normalize()
        return result
    
    @classmethod
    @empty_cached_gpu_memory
    def stack_7ellipses(cls, thickness: int, shape: int = 512, semi_axis_range: tuple[int, int] = (10, 100), padding: int = 50) -> 'Volume':
        """Generates a volume object with random ellipses arranged in a stack. The voxel values of the output volume are between 0 and 1.

        Args:
            thickness (int): The thickness of the stack.
            shape (int, optional): The length and width of the stack. Defaults to 512.
            semi_axis_range (tuple[int, int], optional): The range of values of the randomly choosen semi_axis lengths. Defaults to (10, 100).
            padding (int, optional): The distance limit of the randomly choosen centers from the sides of the squares. Default to 50 voxels.

        Returns:
            Volume: The volume object.
        """
        
        X = cp.arange(shape).reshape(shape,1,1)
        Y = cp.arange(shape).reshape(1,shape,1)

        volume = []

        for _ in progressbar(range(thickness), "Generation of the ellipses: "):
            angles = cp.random.random(7).reshape(1,1,7)*cp.pi
            x0s = cp.random.randint(padding, shape-padding, size=7).reshape(1,1,7)
            y0s = cp.random.randint(padding, shape-padding, size=7).reshape(1,1,7)
            a_vals = cp.random.randint(*semi_axis_range, size=7).reshape(1,1,7)
            b_vals = cp.random.randint(*semi_axis_range, size=7).reshape(1,1,7)
            intensities = cp.random.random(7).reshape(1,1,7)
            UUs = X*cp.cos(angles) + Y*cp.sin(angles)
            VVs = - X*cp.sin(angles) + Y*cp.cos(angles)
            u0s = x0s*cp.cos(angles) + y0s*cp.sin(angles)
            v0s = - x0s*cp.sin(angles) + y0s*cp.cos(angles)
            ellipses_stack = cp.where((UUs-u0s)**2/a_vals**2 + (VVs-v0s)**2/b_vals**2 <= 1, intensities, 0)
            ellipses = cp.sum(ellipses_stack, axis=2)
            volume.append(ellipses/ellipses.max())
        
        volume_np = cp.stack(volume, axis=0).get()
        
        id = f"rand7ellipses{shape}"
        return cls(volume_np, id)

    def normalize(self) -> None:
        """Put voxel values of the volume between 0 and 1."""
        self.volume = (self.volume - self.volume.min())/(self.volume.max()- self.volume.min()).astype(np.float32)

    @empty_cached_gpu_memory
    def get_segmented_volume(self, iso_value: float) -> 'Volume':
        """Computes a new volume in which the intensity values are set either to 0 or 1, depending of iso_value (the volume is normalized beforehand).

        Args:
            iso_value (float): The segmentation value, a float between 0 and 1. All voxel values below iso_value are set to 0, all values above
                are set to 1. If iso_value is None, returns self.
        Returns:
            Volume: The segmented volume.
        """
        if iso_value is None:
            return self

        new_vol = cp.asarray(self.volume)
        new_vol = (new_vol - new_vol.min())/(new_vol.max()- new_vol.min())
        new_vol = cp.where(new_vol <= iso_value, 0, 1)
        new_id = f"{self.id}-segm{iso_value:.02f}"
        return Volume(new_vol.get(), new_id)


    def save(self) -> None:
        """Saving of the volume in the folder volume_files."""

        print("Saving volume...\r")
        with mrcfile.new(self.file_path, overwrite=True) as mrc:
            mrc.set_data(self.volume)
        print(f"File saved at {self.file_path}.\n ID: {self.id}")
    
    @classmethod
    def retrieve(cls, id: str) -> 'Volume':
        """Retrieves the volume in the folder volume_files, given an provided id.

        Args:
            id (str): The id of the volume to retreive.

        Returns:
            Volume: The retrieved volume.
        """
        file_path = DATA_FOLDER / f"volume_files/{id}.mrc"
        with mrcfile.open(file_path) as mrc:
            volume = mrc.data
        return cls(volume, id)

