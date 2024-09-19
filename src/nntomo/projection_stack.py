import os
import warnings
from typing import Union, Optional

import abtem.core
import abtem.core.energy
import dask.array
import numpy as np
import cupy as cp
import abtem
import ase
import astra
import dask
import mrcfile
import PIL
import PIL.Image
import torch
import py4DSTEM

from nntomo.custom_interp import custom_interp_reconstruction
from nntomo.utilities import progressbar, empty_cached_gpu_memory
from nntomo.volume import Volume
from nntomo import DATA_FOLDER


class ProjectionStack:
    """An object representing a stack of projections from tomography imaging, extension of a 3d numpy array to take into consideration the different
    conventions for axis orientation, ASTRA or IMOD. The ASTRA convention is (Nz, Nth, Nd) and the IMOD convention is (Nth, Nd, Nz), where Nth is the
    number of projection angles, Nz is the number of detectors in the direction of the rotation axis, and Nd is the number of detectors in the direction
    perpendicular to the rotation axis. The corresponding set of tilt angles is stored. Currently, the code only works for orientation ranging from -90°
    to 90° ('full' range) and from -70° to 70° ('tem' range): for the neural network input computations, an extended stack of projection with interpolation
    when the angle range is limited has to be computed. Also stores an id and the file_path of the associated .mrc file.

    Args:
        stack (np.ndarray): The stack of projection as a numpy array.
        angles_range (str): The range of projection angles in stack, either 'full' (-90° to 90° projections) or 'tem' (-70° to 70° projections).
            Defaults to 'tem'.
        id (str): Identifiant for the stack, used for the automatic generation of files names.
        axes_convention (str, optional): Convention for the orientation of axes for stack, either 'astra' or 'imod'. Defaults to 'astra'.
    """

    def __init__(self, stack: np.ndarray, angles_range: str, id: str, axes_convention: str = 'astra') -> None:
        stack = stack.astype(np.float32)

        self.id = id
        self.file_path = DATA_FOLDER / f"projection_files/{id}.mrc"
        
        if axes_convention == 'imod':
            self.stack = np.transpose(stack, (2,0,1))
        elif axes_convention == 'astra':
            self.stack = stack
        else:
            raise ValueError(f"axes_convention must be in ('imod', 'astra') but has value '{axes_convention}'")
        
        self.shape = self.stack.shape
        self.Nz, self.Nth, self.Nd = self.shape
        self.angles_range = angles_range
        self.tilt_angles = self._get_tilt_serie(self.Nth, angles_range, 'rad')

        if angles_range == 'full':
            self.interp_Nth = self.Nth
            self.interp_tilt_angles = self.tilt_angles
            self.interp_stack = np.copy(self.stack)

        elif angles_range == 'tem':
            if (self.Nth-1)*9 % 7 == 0:
                #This insure that a proper interpolation can be done from a 140° to a 180° tilt range with regular spacing
                self.interp_Nth = (self.Nth-1)*9 // 7
                self.interp_tilt_angles = np.linspace(-np.pi/2, np.pi/2, self.interp_Nth, endpoint=False, dtype=np.float32)

                og_stack = np.transpose(self.stack, (1,0,2))
                interp_stack = np.zeros((self.interp_Nth, self.Nz, self.Nd), np.float32)
                n_angles_right = (self.interp_Nth - self.Nth)//2
                n_angles_left = n_angles_right + 1

                if n_angles_right == 0:
                        interp_stack[n_angles_left:] = og_stack
                else:
                    interp_stack[n_angles_left:-n_angles_right] = og_stack
                interpolation = np.linspace(np.flipud(og_stack[-1]), og_stack[0], self.interp_Nth - self.Nth + 2, endpoint=True)####

                if n_angles_right > 0:####
                    interp_stack[-n_angles_right:] = np.flip(interpolation[1:n_angles_left], axis=1)####
                interp_stack[:n_angles_left] = interpolation[n_angles_left:-1]####

                self.interp_stack = np.transpose(interp_stack, (1,0,2))

            else:
                raise ValueError(f"tem reconstruction for {self.Nth} projections not yet implemented.")

    @classmethod
    @empty_cached_gpu_memory
    def from_volume(cls, volume: Volume, Nth: int, angles_range: str, custom_id: Optional[str] = None) -> 'ProjectionStack':
        """Creation of a projection stack, provided a volume object. The computation of the projections is done with ASTRA.

        Args:
            volume (Volume): The volume to compute the projections from.
            Nth (int): The number of projections to compute.
            angles_range (str): The range of projection angles in proj_files stacks, either 'full' (-90° to 90° projections) or 'tem'
                (-70° to 70° projections).
            custom_id (Optional[str], optional): Custom identifiant for the stack, used for the automatic generation of files names. If None, a default
                identifiant is given.

        Returns:
            ProjectionStack: The projection stack object.
        """

        Nz, Ny, Nx = volume.shape # ASTRA convention for volume shape. The projections are taken around the Z axis
        Nd = max(Nx, Ny)

        tilt_angles = cls._get_tilt_serie(Nth, angles_range, 'rad')
        proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, Nz, Nd, tilt_angles)
        vol_geom = astra.create_vol_geom(Ny, Nx, Nz)
        _, stack = astra.create_sino3d_gpu(volume.volume, proj_geom, vol_geom)

        if custom_id is None:
            id = f"{volume.id}-{angles_range}{Nth}th"
        else:
            id = custom_id
        proj_stack = cls(stack, angles_range, id, axes_convention='astra')
        return proj_stack

    @classmethod
    @empty_cached_gpu_memory
    def from_cif_file(cls, cif_file: str, Nth: int, angles_range: str, cell_repetition: tuple[int, int, int] = (1,1,1), mode: str = 'haadf', padding: bool = False,
                      nb_frozen_phonons: int = None, dose_per_area_noise: Optional[float] = None, gaussian_filter: Optional[float] = None, energy: float = 300e3,
                      prism_interpolator: tuple[int, int] = (16,16), custom_id: Optional[str] = None, allow_file_retrieval: bool = True)-> 'ProjectionStack':
        """Computes a projection stack object from an atomic structure in a provided .cif file. Projections are computed with the abTEM library,
        to simulate a real STEM experiment. Two modes are available: HAADF and iDPC.

        Args:
            cif_file (str): The atomic structure from which the projections are taken.
            Nth (int): The number of projections to compute.
            angles_range (str): The range of projection angles in proj_files stacks, either 'full' (-90° to 90° projections) or 'tem'
                (-70° to 70° projections).
            cell_repetition (tuple[int, int, int], optional): By how much the cell from the .cif file should be duplicated in each direction. Defaults to (1,1,1).
            mode (str, optional): The mode to compute the projections, either 'haadf' or 'idpc'. Defaults to 'haadf'.
            padding (bool, optional): Wether or not to add padding on the outside of the cell. The padding width is equal to one eigth of the width of the
                cell (after the multiplication by cell_repetition step). Default to False.
            nb_frozen_phonons (int, optional): The number of frozen phonons. If None, the frozen phonons model is not applied.
            dose_per_area_noise (Optional[float], optional): The dose_per_area for Poisson noise. If None, no noise is added.
            gaussian_filter (Optional[float], optional): The standard deviation for Gaussian blur. If None, the filter is not applied.
            energy (float, optional): The energy of the electrons (in eV). Defaults to 300e3.
            prism_interpolator (tuple[int, int], optional): The interpolation factors for the PRISM algorithm. Defaults to (16,16).
                See https://abtem.readthedocs.io/en/latest/user_guide/tutorials/prism.html for more informations.
            custom_id (Optional[str], optional): Custom identifiant for the stack, used for the automatic generation of files names. If None, a default
                identifiant is given.
            allow_file_retrieval (bool, optionnal): Whether or not to allow the retrieval of the projections from a previously saved file with the same
                id. Default to True.

        Returns:
            ProjectionStack: The projection stack object.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if custom_id is None:
                id = cif_file[:-4].split('/')[-1] + f"-{cell_repetition[0]}{cell_repetition[1]}{cell_repetition[2]}-{mode}{Nth}proj"
                if padding:
                    id += '-padd'
                if gaussian_filter is not None:
                    id += '-blur'
                if dose_per_area_noise is not None:
                    id += '-nois'
                if nb_frozen_phonons is not None:
                    id += '-frph'
            else:
                id = custom_id

            if allow_file_retrieval and os.path.isfile(DATA_FOLDER / f"projection_files/{id}.mrc"):
                return cls.retrieve(id, angles_range)


            abtem.config.set({"device": "gpu"})
            abtem.config.set({"dask.chunk-size-gpu": "1024 MB"})
            dask.config.set({"num_workers": 1})

            atoms = ase.io.read(cif_file) * cell_repetition

            if padding:
                base_cell_size = atoms.cell.lengths()[0]
                atoms.center(vacuum=base_cell_size/8)

            if mode == 'idpc':
                detector = abtem.PixelatedDetector()
            elif mode == 'haadf':
                detector = abtem.AnnularDetector(inner=50, outer=200)
            else:
                raise ValueError(f"mode has value {mode} but must be either 'idpc' or 'haadf'.")

            sampling = abtem.core.energy.energy2wavelength(energy) / 3 / 200e-3 / 1.05
            
            projs = []
            tilt_angles = cls._get_tilt_serie(Nth, angles_range, 'deg')
            for angle in progressbar(tilt_angles, "Computation of projections: "):
                
                tilted_cell = atoms.copy()# * cell_repetition
                tilted_cell.rotate(angle, 'y', rotate_cell=False, center='COP')

                if nb_frozen_phonons is not None:
                    tilted_cell = abtem.FrozenPhonons(tilted_cell, num_configs=nb_frozen_phonons, sigmas=.1)

                tilted_potential = abtem.Potential(tilted_cell, sampling=sampling, projection='infinite')

                s_matrix = abtem.SMatrix(potential=tilted_potential, energy=energy, semiangle_cutoff=22, interpolation=prism_interpolator)
                nyquist_sampling = abtem.transfer.nyquist_sampling(s_matrix.semiangle_cutoff, s_matrix.energy)

                grid_scan = abtem.GridScan(
                    start=(0, 0),
                    end=(1, 1),
                    sampling=nyquist_sampling,
                    fractional=True,
                    potential=tilted_potential,
                )
                measurement = s_matrix.scan(scan=grid_scan, detectors=detector).compute(progress_bar=False)

                if mode == 'idpc':
                    dataset = py4DSTEM.DataCube(measurement.array)
                    dpc = py4DSTEM.process.phase.DPC(energy = energy, datacube = dataset, verbose=False)
                    dpc.preprocess(force_com_rotation=False, plot_rotation=False, plot_center_of_mass=False)
                    dpc.reconstruct(reset=True, progress_bar=False)
                    measurement = abtem.measurements.Images(dpc.object_phase, nyquist_sampling)

                #intensity = measurement.interpolate(sampling=sampling)

                # add gaussian blur
                if gaussian_filter is not None:
                    measurement = measurement.gaussian_filter(gaussian_filter)
                    
                # add poisson noise
                if dose_per_area_noise is not None:
                    measurement = measurement.poisson_noise(dose_per_area=dose_per_area_noise)

                projs.append(measurement.array)

            stack = np.stack(projs)

            return cls(stack, angles_range, id, axes_convention='imod')
    
    @classmethod
    def from_mrc_file(cls, projection_file: str, angles_range: str = 'tem', custom_id: Optional[str] = None) -> 'ProjectionStack':
        """Creation of a projection stack object, from a .ali or .mrc file.

        Args:
            projection_file (str): the .mrc file path which represents a stack of projections.
            angles_range (str): The range of projection angles in proj_files stacks, either 'full' (-90° to 90° projections) or 'tem'
                (-70° to 70° projections).
            custom_id (Optional[str], optional): Custom identifiant for the stack, used for the automatic generation of files names. If None, a default
                identifiant is given.

        Returns:
            ProjectionStack: the projection stack object.
        """

        with mrcfile.open(projection_file, permissive=True) as mrc:
            stack = mrc.data

        if custom_id is None :
            id = projection_file.split('.')[0].split('/')[-1]
        else:
            id = custom_id
        
        return cls(stack, angles_range, id, axes_convention='imod')


    def get_proj_subset(self, nb_proj_subset: int) -> 'ProjectionStack':
        """Computes a subset of the projection stack to simulate TEM reconstruction with less projection angles. nb_proj_subset should be set so that
        the resulting subset has a regular spacing between projection angles. When angles_range == 'tem', the constraint of being able to include both
        projections at -70° and 70° is added.

        Args:
            nb_proj_subset (int): The number of projections in the subset of proj_stack.
            
        Returns:
            ProjectionStack: The subset.
        """
        if self.angles_range == 'full':
            if self.Nth % nb_proj_subset == 0:
                new_stack = self.stack[:, 0 :: (self.Nth // nb_proj_subset)]
            else:
                raise ValueError("When angles_range == 'full', nb_proj_subset should be a divisor of the number of projections in proj_stack.")
        elif self.angles_range == 'tem':
            if (self.Nth-1) % (nb_proj_subset-1) == 0:
                new_stack = self.stack[:, 0 :: ((self.Nth-1) // (nb_proj_subset-1))]
            else:
                raise ValueError("When angles_range == 'tem',(nb_proj_subset - 1) should be a divisor of the number of projections in proj_stack - 1, \
                                since both projections at angles -70° and 70° are kept.")
            
        new_id = f"{self.id}-sub{nb_proj_subset}"
            
        return ProjectionStack(new_stack, self.angles_range, new_id)
    
    def convert_full_tem(self) -> 'ProjectionStack':
        """Returns a new projection stack with a 'tem' limited angle range (projections from -70° to 70°). The range of the original stack should be 'full'
        (angles from -90° to 90°).
        """
        if self.angles_range != 'full':
            raise ValueError("The angles_range of the stack should be 'full' to apply this method.")
        if self.Nth*7 % 9 != 0:
            raise ValueError("The conversion cannot be made: the original stack doesn't contain projections at -70° and 70°.")
        new_Nth = self.Nth*7//9 + 1
        if (self.Nth-new_Nth)//2 == 0:
            new_stack = self.stack[:, 1:]
        else:
            new_stack = self.stack[:, (self.Nth-new_Nth)//2 + 1 : - ((self.Nth-new_Nth)//2)]
        new_id = f"{self.id}-subtem{new_Nth}"
        return ProjectionStack(new_stack, 'tem', new_id)

    def get_resized_proj(self, new_size: Union[int, tuple[int, int]]) -> 'ProjectionStack':
        """Optionnaly performs a centered crop of the projections so that the aspect ratio is the same as the one determined by new_size, then
        resizes the projections to the new size. Returns the new stack of projections.

        Args:
            new_size (Union[int, tuple[int, int]]): If int, the stack is resized so that the amount of horizontal detectors (Nd) is set to new_size.
                If tuple, the stack is cropped first to have the same aspect ratio as new_size, then resized.

        Returns:
            ProjectionStack: The resized projection stack.
        """
        a, b = self.Nd, self.Nz

        if type(new_size) is int:
            images = [PIL.Image.fromarray(proj).resize((new_size*b//a, new_size)) for proj in self._get_imod_stack()]
            new_id = f"{self.id}-resized{new_size}"


        else:
            c, d = new_size
            if a/b < c/d:
                box_crop = (0, (b - a*d/c)//2, a, (b + a*d/c)//2)
            elif a/b > c/d:
                box_crop = ((a - b*c/d)//2, 0, (a + b*c/d)//2, b)
            else:
                box_crop = (0, 0, a, b)

            images = [PIL.Image.fromarray(proj).crop(box_crop).resize(new_size) for proj in self._get_imod_stack()]
            new_id = f"{self.id}-resized[{new_size[0]}-{new_size[1]}]"



        new_stack = np.stack([np.array(image) for image in images])

        return ProjectionStack(new_stack, self.angles_range, new_id, 'imod')

    @empty_cached_gpu_memory
    def get_SIRT_reconstruction(self, min_eps_var: float = 0.001, n_iter: Optional[int] = None, print_n_iter: bool = False, force_positive_values: bool = True,
                                allow_file_retrieval: bool = True) -> Union[Volume, tuple[Volume, list]]:
        """Computes the SIRT reconstruction, using ASTRA toolbox. Two modes: the SIRT reconstruction can be computed given a precise number of
        iterations (by providing the parameter n_iter), or the reconstruction is stopped when the variation of eps[i] = ||Ax[i]-b||, defined by
        (eps[i-1] - eps[i])/eps[i-1] is smaller than min_eps_var; where A is the forward projection operator, x[i] is the current SIRT reconstruction,
        and b is the provided sinogram / stack of projections. 

        Args:
            min_eps_var (float, optionnal): If n_iter is not given, minimum value for eps[i] before stopping. Default to 0.001.
            n_iter (Optional[int], optionnal): If given, fixed number of iterations for the SIRT algorithm.
            print_n_iter (bool, optionnal): Whether or not to print the final number of SIRT iterations. If the volume is retrieved from a file (with
                allow_file_retrieval set to True), nothing is printed either way. Default to False.
            force_positive_values (bool, optionnal): Whether or not to force SIRT to output positive values. Default to True
            allow_file_retrieval (bool, optionnal): Whether or not to allow the retrieval of the reconstruction from a previously saved file with the same
                id. Default to True.
            
        Returns:
            Volume: The reconstructed volume.
        """

        reconstruction_id = f"sirt_{self.id}"
        if allow_file_retrieval and os.path.isfile(DATA_FOLDER / f"volume_files/{reconstruction_id}.mrc"):
                return Volume.retrieve(reconstruction_id)

        proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, self.Nz, self.Nd, self.tilt_angles)
        proj_id = astra.data3d.create('-sino', proj_geom, self.stack)

        vol_geom = astra.create_vol_geom(self.Nd, self.Nd, self.Nz)
        recon_id = astra.data3d.create('-vol', vol_geom, data=0)  # Initialize with zeros

        cfg = astra.astra_dict('SIRT3D_CUDA')
        cfg['ReconstructionDataId'] = recon_id
        cfg['ProjectionDataId'] = proj_id
        if force_positive_values:
            cfg['option'] = {'MinConstraint': 0}
        alg_id = astra.algorithm.create(cfg)

        if n_iter is None:
            current_epsilon = None
            last_epsilon = None
            cp_stack = cp.asarray(self.stack)
            n_iter = 0
            while last_epsilon is None or (last_epsilon - current_epsilon)/last_epsilon > min_eps_var:
                last_epsilon = current_epsilon
                astra.algorithm.run(alg_id, 1)
                n_iter += 1
                current_rec = astra.data3d.get(recon_id)
                _, forward_proj = astra.create_sino3d_gpu(current_rec, proj_geom, vol_geom)
                cp_forward_proj = cp.asarray(forward_proj)
                current_epsilon = float(cp.sum(cp.square(cp_forward_proj - cp_stack)))
            reconstruction = current_rec

        else:
            astra.algorithm.run(alg_id, n_iter)
            reconstruction = astra.data3d.get(recon_id) 

        if print_n_iter:
            print(f"SIRT reconstruction computed with {n_iter} iterations.")
        return Volume(reconstruction, reconstruction_id)
    
    @empty_cached_gpu_memory
    def get_FBP_reconstruction(self, allow_file_retrieval: bool = True) -> Volume:
        """Computes the FBP reconstruction, using ASTRA toolbox. The computation is done by doing 2D FBP for each layer of the volume to reconstruct.
        
        Args:
            allow_file_retrieval (bool, optionnal): Whether or not to allow the retrieval of the reconstruction from a previously saved file with the same
                id. Default to True.
        
        Returns:
            Volume: the reconstructed volume.
        """
        reconstruction_id = f"fbp_{self.id}"
        if allow_file_retrieval and os.path.isfile(DATA_FOLDER / f"volume_files/{reconstruction_id}.mrc"):
                return Volume.retrieve(reconstruction_id)
        
        vol_geom = astra.create_vol_geom(self.Nd, self.Nd)
        proj_geom = astra.create_proj_geom('parallel', 1.0, self.Nd, self.tilt_angles)
        proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
        rec_id = astra.data2d.create('-vol', vol_geom)

        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['option'] = {'FilterType': 'Ram-Lak'}

        reconstruction = np.zeros((self.Nz, self.Nd, self.Nd), dtype=np.float32)

        for layer_index in range(self.Nz):

            proj_id = astra.data2d.create('-sino', proj_geom, self.stack[layer_index])
            
            cfg['ProjectionDataId'] = proj_id
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)

            reconstruction[layer_index] = astra.data2d.get(rec_id)

            astra.data2d.delete(proj_id)
            astra.algorithm.delete(alg_id)
        
        return Volume(np.flip(reconstruction, axis=1), reconstruction_id)
    
    @torch.no_grad
    def get_NNFBP_reconstruction(self, nn_model: 'NNFBP', empty_cached_memory: bool = True, show_progressbar: bool = True, # type: ignore  # noqa: F821
                                 allow_file_retrieval: bool = True) -> Volume:
        """Computes the neural network reconstruction, using the NN-FBP algorithm.

        Args:
            nn_model (NNFPB): The trained model used for the reconstruction.
            empty_cached_memory (bool, optional): Whether or not to delete the gpu cached memory generated by cupy and torch (which cannot be used
                by other libraries like astra).
            show_progressbar (bool, optionnal): Whether or not to show progressbar. Default to True.
            allow_file_retrieval (bool, optionnal): Whether or not to allow the retrieval of the reconstruction from a previously saved file with the same
                id. Default to True.

        Returns:
            Volume: the reconstructed volume.
        """
        reconstruction_id = f"{nn_model.id}_{self.id}"
        if allow_file_retrieval and os.path.isfile(DATA_FOLDER / f"volume_files/{reconstruction_id}.mrc"):
                return Volume.retrieve(reconstruction_id)

        if nn_model.Nth != self.interp_Nth or nn_model.Nd != self.Nd:
            raise ValueError("The model wasn't trained for this volume shape and number of projections.")

        nn_model.eval()

        ### Insure that voxels values are between 0 and 1 ###
        proj_stack = cp.asarray(np.transpose(self.interp_stack, (1,0,2)))

        ### Voxels coordinates ###
        X = (cp.arange(self.Nd) - (self.Nd-1)/2).astype(cp.float32)
        Y = (cp.arange(self.Nd) - (self.Nd-1)/2).astype(cp.float32)
        Z = cp.arange(self.Nz, dtype=cp.int32)
        XX, YY = cp.meshgrid(X, Y)

        Nh = nn_model.Nh

        ### Algorithm part 1 ###

        raw_input_size = 2*self.Nd-1
        fft_out_size = raw_input_size + self.Nd - 1
        T_convolve = cp.linspace(-(fft_out_size-1)/2, (fft_out_size-1)/2, fft_out_size, dtype = cp.float32)

        weights = nn_model.get_FBP_weights()
        weights_fft = cp.fft.rfft(cp.asarray(weights), fft_out_size, axis=1)

        b = cp.asarray(nn_model.W.bias)

        fbps = np.zeros((self.Nd, self.Nd, self.Nz, Nh), dtype = np.float32)

        if show_progressbar:
            iterable = progressbar(range(Nh), "Reconstruction part 1/2: ")
        else:
            iterable = range(Nh)
        for h in iterable:
            fbp_h = cp.zeros((self.Nd, self.Nd, self.Nz), dtype = cp.float32)
            for i, theta in enumerate(self.interp_tilt_angles):
                proj_fft = cp.fft.rfft(proj_stack[i], fft_out_size, axis=1)
                prod_fft = proj_fft * weights_fft[h].reshape(1, fft_out_size//2+1)
                convol = cp.fft.irfft(prod_fft, fft_out_size, axis=1).astype(cp.float32)
                UU = XX*cp.cos(theta) + YY*cp.sin(theta)
                fbp_h += custom_interp_reconstruction(UU, Z, T_convolve, convol)
            
            fbps[:,:,:,h] = (fbp_h + b[h]).get()
        

        ### Algorithm part 2-3-4 ###

        reconstruction = np.zeros((self.Nd, self.Nd, self.Nz), dtype = np.float32)
        
        nn_model.to('cuda:0')
        if show_progressbar:
            iterable = progressbar(range(Nh), "Reconstruction part 2/2: ")
        for h in iterable:
            input_nn = torch.from_numpy(fbps[:,:,h*self.Nz//Nh:(h+1)*self.Nz//Nh,:]).to('cuda:0')
            reconstruction[:,:,h*self.Nz//Nh:(h+1)*self.Nz//Nh] = nn_model.end_forward(input_nn).cpu().numpy()
        nn_model.to('cpu')


        if empty_cached_memory:
            del(X, Y, Z, proj_stack, XX, YY, T_convolve, weights, weights_fft, b, fbp_h, proj_fft, prod_fft, convol, UU, input_nn)
            cp.get_default_memory_pool().free_all_blocks()
            torch.cuda.empty_cache()
        
        reconstruction = np.rot90(np.rot90(reconstruction, axes=(2,0)), axes=(1,2))

        reconstruction = (reconstruction - nn_model.b)/nn_model.a

        return Volume(reconstruction, reconstruction_id)
    
    @torch.no_grad
    def get_MSDNET_reconstruction(self, nn_model: 'MSDNET', empty_cached_memory: bool = True, show_progressbar: bool = True, # type: ignore # noqa: F821
                                  allow_file_retrieval: bool = True) -> Volume:
        """Computes the MSDNET reconstruction.

        Args:
            nn_model (MSDNET): The trained model used for the reconstruction.
            empty_cached_memory (bool, optional): Whether or not to delete the gpu cached memory generated by cupy and torch (which cannot be used
                by other libraries like astra).
            show_progressbar (bool, optionnal): Whether or not to show progressbar. Default to True.
            allow_file_retrieval (bool, optionnal): Whether or not to allow the retrieval of the reconstruction from a previously saved file with the same
                id. Default to True.

        Returns:
            Volume: the reconstructed volume.
        """
        reconstruction_id = f"{nn_model.id}_{self.id}"
        if allow_file_retrieval and os.path.isfile(DATA_FOLDER / f"volume_files/{reconstruction_id}.mrc"):
                return Volume.retrieve(reconstruction_id)

        if nn_model.Nth != self.Nth or nn_model.Nd != self.Nd or nn_model.angles_range != self.angles_range:
            raise ValueError("The model wasn't trained for this volume shape, number of projections and range of projection angles.")
        
        nn_model.eval()
        nn_model.to('cuda:0')

        fbp_volume = self.get_FBP_reconstruction().volume
        reconstruction = np.zeros_like(fbp_volume)

        if show_progressbar:
            iterable = progressbar(range(len(fbp_volume)), "MSDNET forward: ")
        else:
            iterable = range(len(fbp_volume))
        for slice in iterable:
            gpu_slice = torch.from_numpy(fbp_volume[slice]).view(1,fbp_volume.shape[1], fbp_volume.shape[2]).to('cuda:0')
            reconstruction[slice] = nn_model.forward(gpu_slice).cpu().numpy()
        
        nn_model.to('cpu')

        if empty_cached_memory:
            del(gpu_slice)
            torch.cuda.empty_cache()

        return Volume(reconstruction, reconstruction_id)



    def _get_astra_stack(self) -> np.ndarray:
        """Returns the numpy array stack in the ASTRA axes convention."""
        return self.stack

    def _get_imod_stack(self) -> np.ndarray:
        """Returns the numpy array stack in the IMOD axes convention."""
        return np.transpose(self.stack, (1,2,0))

    @staticmethod
    def _get_tilt_serie(Nth: int, angles_range: str, unit: str) -> np.ndarray:
        """Computes a tilt serie, provided a number of projections, an angle range and a unit.

        Args:
            Nth (int): The number of projections.
            angles_range (str): The range of angles, either 'full' or 'tem'.
            unit (str): Either 'rad' or 'deg'.

        Returns:
            np.ndarray: The tilt serie.
        """

        if angles_range == 'full':
            tilt_angles = np.linspace(-90, 90, Nth, endpoint=False, dtype=np.float32)
        elif angles_range == 'tem':
            tilt_angles = np.linspace(-70, 70, Nth, endpoint=True, dtype=np.float32)
        else:
            raise ValueError(f"angles_range must be in ('tem', 'full') but has value '{angles_range}'")
        
        if unit == 'rad':
            tilt_angles *= np.pi/180
        elif unit != 'deg':
            raise ValueError(f"unit must be in ('rad', 'deg') but has value '{unit}'")
        
        return tilt_angles


    def save(self) -> None:
        """Saving of the projection stack in the folder projection_files, in the IMOD axes convention."""

        print("Saving projections...\r")
        with mrcfile.new(self.file_path, overwrite=True) as mrc:
            mrc.set_data(self._get_imod_stack())
        print(f"File saved at {self.file_path}.\n ID: {self.id}")

    @classmethod
    def retrieve(clas, id: str, angles_range: str) -> 'ProjectionStack':
        """Retrieves the projection stack in the folder projection_files, given an provided id.

        Args:
            id (str): The id of the stack to retrieve.
            angles_range (str): The range of projection angles in proj_files stacks, either 'full' (-90° to 90° projections) or 'tem'
                (-70° to 70° projections).

        Returns:
            ProjectionStack: The retrieved stack.
        """

        file_path = DATA_FOLDER / f"projection_files/{id}.mrc"
        with mrcfile.open(file_path) as mrc:
            stack = mrc.data
        return clas(stack, angles_range, id, 'imod')

