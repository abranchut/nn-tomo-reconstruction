import math

import cupy as cp

from nntomo import GPU_BLOCK_SIZE


def custom_interp_dataset_init(UU: cp.ndarray, Z: cp.ndarray, T: cp.ndarray, proj_data: cp.ndarray, Nz: int) -> cp.ndarray:
    """ One-dimensional linear interpolation, over multiple axis at the same time. Used for the dataset initialisation, which requires to compute
    equation (18) of [Pelt 2013] on multiple voxels at the same time.

    Args:
        UU (cp.ndarray): a 3D array of horizontal coordinates on which the interpolation is performed.
        Z (cp.ndarray): a 1D array of vertical coordinates so that Z[i] is associated with UU[i,:,:].
        T (cp.nd_array): a 1D array of horizontal coordinates on which the function values (proj_data) are known.
        proj_data (cp.ndarray): a 3D array containing the function values at the points T, for each value of theta and z (in Z).
        Nz (int) : Vertical size of the volume. Z takes its values in range(Nz).


    Returns:
        cp.ndarray: The interpolated values, same shape as UU.
    """

    # Ideally, the inputs are already in the good format
    UU = cp.ascontiguousarray(UU, dtype=cp.float32)
    Z = cp.ascontiguousarray(Z, dtype=cp.int32)
    T = cp.ascontiguousarray(T, dtype=cp.float32)
    proj_data = cp.ascontiguousarray(proj_data, dtype=cp.float32)

    n, Nth, input_size = UU.shape
    Nd = T.shape
    n = cp.int32(n)
    Nth = cp.int32(Nth)
    Nd = cp.int32(Nd)
    input_size = cp.int32(input_size)
    Nz = cp.int32(Nz)
    output = cp.zeros_like(UU, dtype=cp.float32)
    idx = cp.searchsorted(T, UU, side='right').astype(cp.int32)

    kern = cp.RawKernel(r'''
        extern "C" __global__
        void interp_kernel(const float* UU, const int* Z, const int* idx, const float* T, const float* proj_data, const int n, const int Nz, const int Nth,
                           const int input_size, const int Nd, float* out) {
            // idx from search_sorted(right) in n dimemsions
            int i = blockIdx.x * blockDim.x + threadIdx.x; // index of the random pixel
            int j = blockIdx.y * blockDim.y + threadIdx.y; // theta
            int k = blockIdx.z * blockDim.z + threadIdx.z; // detector
                        
            if (i<n && j<Nth && k<input_size) {
                long long index_3d_grid = k + j*input_size + i*input_size*Nth;

                int right_idx = idx[index_3d_grid];
                int left_idx = right_idx - 1;
                int z_idx = Z[i];
                
                if (UU[index_3d_grid] == T[Nd - 1]) {
                    out[index_3d_grid] = proj_data[Nd - 1 + z_idx*Nd + j*Nd*Nz];
                }
                else if (left_idx < 0 || right_idx > Nd-1) {
                    out[index_3d_grid] = 0.;
                }
                else {
                    const float slope = (proj_data[right_idx + z_idx*Nd + j*Nd*Nz] - proj_data[left_idx + z_idx*Nd + j*Nd*Nz]) / (T[right_idx] - T[left_idx]);
                    out[index_3d_grid] = slope * (UU[index_3d_grid] - T[left_idx]) + proj_data[left_idx + z_idx*Nd + j*Nd*Nz];
                }
            }
        }
        ''', 'interp_kernel')
    
    grid_size = (math.ceil(n/GPU_BLOCK_SIZE[0]), math.ceil(Nth/GPU_BLOCK_SIZE[1]), math.ceil(input_size/GPU_BLOCK_SIZE[2]))
    
    kern(grid_size, GPU_BLOCK_SIZE, (UU, Z, idx, T, proj_data, n, Nz, Nth, input_size, Nd, output))
    return output


def custom_interp_reconstruction(UU, Z, T, convol):
    """ One-dimensional linear interpolation, over multiple axis at the same time. Used for reconstruction.

    Args:
        UU (cupy.ndarray): a 2D array of horizontal coordinates on which the interpolation is performed.
        Z (cupy.ndarray): a 1D array of vertical coordinates (on which the interpolation is not performed).
        T (cupy.nd_array): a 1D array of horizontal coordinates on which the function values (convol) are known.
        convol (cupy.ndarray): a 2D array containing the function values at the points T, for each value of z (in ZZ).

    Returns:
        cupy.ndarray: The interpolated values, same shape as UU.
    """

    # Ideally, the inputs are already in the good format
    UU = cp.ascontiguousarray(UU, dtype=cp.float32)
    Z = cp.ascontiguousarray(Z, dtype=cp.int32)
    T = cp.ascontiguousarray(T, dtype=cp.float32)
    convol = cp.ascontiguousarray(convol, dtype=cp.float32)
    Nd = cp.int32(len(T))

    Nx, Ny = UU.shape
    Nz = Z.shape[0]
    output = cp.zeros((Nx,Ny,Nz), dtype=cp.float32)
    Nx = cp.int32(Nx)
    Ny = cp.int32(Ny)
    Nz = cp.int32(Nz)
    idx = cp.searchsorted(T, UU, side='right').astype(cp.int32)

    kern = cp.RawKernel(r'''
        extern "C" __global__
        void interp_kernel(const float* UU, const int* Z, const int* idx, const float* T, const float* convol, const int Nd, const int Nx, const int Ny,
                           const int Nz, float* out) {
            // idx from search_sorted(right) in n dimemsions
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;
            int k = blockIdx.z * blockDim.z + threadIdx.z;
            
            if (i<Nx && j<Ny && k<Nz) {
                long long index_3d_grid = k + j*Nz + i*Ny*Nz;
                long long index_uu = j + i*Ny;

                int right_idx = idx[index_uu];
                int left_idx = right_idx - 1;
                int z_idx = Z[k];

                if (UU[index_uu] == T[Nd - 1]) {
                    out[index_3d_grid] = convol[Nd - 1 + Nd*z_idx];
                }
                else if (left_idx < 0 || right_idx > Nd-1) {
                    out[index_3d_grid] = 0.;
                }
                else {
                    const float slope = (convol[right_idx + Nd*z_idx] - convol[left_idx + Nd*z_idx]) / ( T[right_idx] - T[left_idx]);
                    out[index_3d_grid] = slope * (UU[index_uu] - T[left_idx]) + convol[left_idx + Nd*z_idx];
                }
            }
        }
        ''', 'interp_kernel')
    
    grid_size = (math.ceil(Nx/GPU_BLOCK_SIZE[0]), math.ceil(Ny/GPU_BLOCK_SIZE[1]), math.ceil(Nz/GPU_BLOCK_SIZE[2]))

    kern(grid_size, GPU_BLOCK_SIZE, (UU, Z, idx, T, convol, Nd, Nx, Ny, Nz, output))
    return output