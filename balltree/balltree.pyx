#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
# distutils: language=c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: language_level=3

import torch
import numpy as np
cimport numpy as cnp
from libcpp cimport bool
from libc.math cimport ceil, log2
from libc.stdlib cimport malloc, free
from cython.parallel cimport parallel, prange
from openmp cimport omp_get_max_threads, omp_set_num_threads

cnp.import_array()

ctypedef cnp.intp_t intp_t
ctypedef cnp.float64_t float64_t

cdef extern from "balltree.h" nogil:
    void build_tree(
        const float64_t* data,      # Input data array
        intp_t* idx_array,          # Working array for indices
        intp_t* output_indices,     # Output array for tree
        bool* output_mask,          # Output array for mask (dummy indicators)
        intp_t idx_start,           # Start index in working array
        intp_t idx_end,             # End index in working array
        intp_t output_start,        # Start index in output array
        intp_t num_features,        # Dimensionality of data
        intp_t current_level,       # Current tree level
        intp_t max_level            # Maximum tree depth
    )

cdef struct BatchMetadata:
    intp_t start          # Start index in original data array
    intp_t end            # End index (exclusive) in original data array
    intp_t tree_offset    # Start index in output array for this tree
    intp_t tree_size      # Number of slots allocated for this tree
    intp_t num_leaves     # Number of leaves in the tree

cdef intp_t compute_tree_depth(intp_t num_points) nogil:
    """Compute the depth needed for a complete binary tree.
    
    Parameters
    ----------
    num_points : intp_t
        Number of points in the tree
        
    Returns
    -------
    intp_t
        Depth of the complete binary tree needed to store the points
    """
    return <intp_t>ceil(log2(num_points)) - 1

cdef int process_batch(
    const float64_t* data,       # Input data array
    BatchMetadata batch,         # Batch metadata
    intp_t* output,              # Output array for tree structure
    bool* mask,                  # Output array for mask
    intp_t num_features          # Dimensionality of data
) nogil:
    """Process a single batch to build its ball tree.
    
    Thread-safe function that builds a complete binary tree for a subset of points.
    
    Parameters
    ----------
    data : float64_t*
        Pointer to the start of the data array
    batch : BatchMetadata
        Metadata for the current batch
    output : intp_t*
        Pointer to the output array
    num_features : intp_t
        Number of features in the data
        
    Returns
    -------
    int
        0 on success, negative values on error
    """
    cdef intp_t batch_size = batch.end - batch.start
    cdef intp_t max_level = compute_tree_depth(batch_size)
    
    # Allocate an intermediate array for indices
    cdef intp_t* idx_array = <intp_t*>malloc(batch_size * sizeof(intp_t))
    if idx_array == NULL:
        return -1
        
    # Initialize indices contiguouslly
    cdef intp_t i
    for i in range(batch_size):
        idx_array[i] = i
        
    build_tree(
        data + batch.start * num_features,  # Point to batch's data
        idx_array,                          # Temporary working array
        output + batch.tree_offset,         # Point to batch's output location
        mask + batch.tree_offset,           # Point to batch's mask location
        0,                                  # Start at beginning of batch
        batch_size,                         # Process whole batch
        0,                                  # Start at beginning of output
        num_features,                       
        0,                                  # Start at root level
        max_level                          
    )
    
    # Adjust indices to account for batch offset
    cdef intp_t num_leaves = 1 << max_level
    for i in range(num_leaves * 2):
        if i < batch.tree_size:
            output[batch.tree_offset + i] += batch.start
    
    free(idx_array)
    return 0

def build_balltree_with_offsets(data, batch_offsets):
    """Build ball trees for multiple subarrays in parallel.
    
    For each subarray defined by batch_offsets, constructs a complete binary tree
    where each leaf node contains 2 points. Trees are built in parallel using OpenMP.
    
    Parameters
    ----------
    data : ndarray of shape (num_samples, num_features)
        Input data points. Must be contiguous and float64.
    batch_offsets : ndarray of shape (num_batches + 1,)
        Indices that define the boundaries of subarrays in data.
        Each subarray i contains points from batch_offsets[i] to batch_offsets[i+1].
        
    Returns
    -------
    ndarray
        Flattened array containing all ball trees. Each leaf node occupies 2 slots,
        either containing two point indices or duplicating a single point index.
        Tree structure can be inferred from batch sizes.
        
    Notes
    -----
    - Input arrays must be contiguous
    - Builds complete binary trees with fixed depth based on subarray size
    - Thread-safe and GIL-free during tree construction
    
    Raises
    ------
    ValueError
        If input arrays have incorrect shapes or invalid batch offsets
    RuntimeError
        If tree construction fails
    """
    # Input validation
    if not isinstance(data, np.ndarray) or not isinstance(batch_offsets, np.ndarray):
        raise ValueError("Both data and batch_offsets must be numpy arrays")

    if data.ndim != 2:
        raise ValueError(
            f"Data must be 2D array of shape (num_samples, num_features), "
            f"got shape {data.shape}"
        )

    if batch_offsets.ndim != 1:
        raise ValueError(
            f"Batch offsets must be 1D array of shape (num_batches + 1,), "
            f"got shape {batch_offsets.shape}"
        )
        
    if len(batch_offsets) < 2:
        raise ValueError("Batch offsets must contain at least 2 elements")
    
    cdef float64_t[:, ::1] data_arr = np.asarray(data, dtype=np.float64)
    cdef intp_t[::1] batch_offsets_arr = np.asarray(batch_offsets, dtype=np.intp)
    
    cdef intp_t num_batches = len(batch_offsets) - 1
    cdef intp_t num_features = data.shape[1]
    
    # Prepare meradata for each subarray
    cdef BatchMetadata[::1] batch_metadata = np.zeros(num_batches, dtype=[
        ('start', np.intp),
        ('end', np.intp),
        ('tree_offset', np.intp),
        ('tree_size', np.intp),
        ('num_leaves', np.intp)
    ])
    
    cdef intp_t total_size = 0
    cdef intp_t i, num_points
    cdef BatchMetadata* metadata
    
    for i in range(num_batches):
        # Get pointer to current batch metadata
        metadata = &batch_metadata[i]
        
        metadata.start = batch_offsets_arr[i]
        metadata.end = batch_offsets_arr[i + 1]
        
        if metadata.start >= metadata.end:
            raise ValueError(f"Invalid batch offsets at index {i}")
        if metadata.start < 0 or metadata.end > data.shape[0]:
            raise ValueError(f"Batch offset {i} out of bounds")
        
        # Calculate tree structure
        num_points = metadata.end - metadata.start
        metadata.num_leaves = 1 << compute_tree_depth(num_points)  # 2^depth
        metadata.tree_size = metadata.num_leaves * 2  # Each leaf gets 2 slots
        metadata.tree_offset = total_size
        
        total_size += metadata.tree_size
    
    # Allocate memory for the output arrays
    cdef intp_t[::1] output = np.empty(total_size, dtype=np.intp)
    cdef bool[::1] mask = np.empty(total_size, dtype=np.bool_)
    
    # Configure parallel processing of batches
    cdef int num_threads = omp_get_max_threads()
    cdef int chunk_size = max(1, num_batches // num_threads)
    omp_set_num_threads(num_threads)

    cdef int ret
    
    with nogil, parallel(num_threads=num_threads):
        for i in prange(num_batches, schedule='static', chunksize=chunk_size):
            ret = process_batch(
                &data_arr[0,0],
                batch_metadata[i],
                &output[0],
                &mask[0],
                num_features
            )
            if ret != 0:
                with gil:
                    raise RuntimeError(f"Error processing batch {i}: error code {ret}")
    
    return np.asarray(output), np.asarray(mask)


def build_balltree(data, batch_idx):
    """Build ball trees for multiple subarrays in parallel using batch indices.
    
    Similar to build_balltree_with_offsets but takes batch indices instead of offsets.
    Batch indices must be contiguous integers starting from 0.
    
    Parameters
    ----------
    data : ndarray of shape (num_samples, num_features)
        Input data points. Must be contiguous and float64.
    batch_idx : ndarray of shape (num_samples,)
        Integer array assigning each point to a batch.
        Must be contiguous integers starting from 0.
        
    Returns
    -------
    ndarray
        Flattened array containing all ball trees. Each leaf node occupies 2 slots,
        either containing two point indices or duplicating a single point index.
    """
    if not isinstance(batch_idx, np.ndarray):
        raise ValueError("batch_idx must be a numpy array")
        
    if batch_idx.ndim != 1:
        raise ValueError(f"batch_idx must be 1D array, got shape {batch_idx.shape}")
        
    if batch_idx.shape[0] != data.shape[0]:
        raise ValueError(
            f"batch_idx length ({batch_idx.shape[0]}) must match "
            f"number of data points ({data.shape[0]})"
        )
    
    if len(batch_idx) == 0:
        raise ValueError("batch_idx is empty")
    
    # Check if first index is 0
    if batch_idx[0] != 0:
        raise ValueError("Batch indices must start from 0")
    
    n = len(batch_idx)
    
    # Compute offsets by finding where batch index changes
    change_points = np.where(batch_idx[1:n] != batch_idx[0:n-1])[0] + 1
    num_batches = batch_idx[n-1] + 1

    batch_offsets = np.zeros(num_batches + 1, dtype=np.intp)
    batch_offsets[1:num_batches] = change_points
    batch_offsets[num_batches] = n
    
    return build_balltree_with_offsets(data, batch_offsets)


def build_balltree_torch(data: object, batch_idx: object):
    """Build ball trees for multiple subarrays in parallel using PyTorch tensors.
    
    Parameters
    ----------
    data : torch.Tensor of shape (num_samples, num_features)
        Input data points. Can be on CPU or CUDA.
    batch_idx : torch.Tensor of shape (num_samples,)
        Integer tensor assigning each point to a batch.
        Must be contiguous integers starting from 0.
        
    Returns
    -------
    tuple
        (indices, mask) tuple where both elements are torch.Tensor
        on the same device as the input.
    """
    # Validate inputs are torch tensors
    if not isinstance(data, torch.Tensor) or not isinstance(batch_idx, torch.Tensor):
        raise ValueError("Inputs must be PyTorch tensors")
        
    if data.dim() != 2:
        raise ValueError(f"Data must be 2D tensor, got shape {tuple(data.shape)}")
        
    if batch_idx.dim() != 1:
        raise ValueError(f"batch_idx must be 1D tensor, got shape {tuple(batch_idx.shape)}")
        
    if batch_idx.shape[0] != data.shape[0]:
        raise ValueError("batch_idx length must match number of data points")

    # Store original device
    device = data.device
    
    # Convert to numpy arrays
    cdef cnp.ndarray data_np = data.detach().cpu().double().numpy()
    cdef cnp.ndarray batch_idx_np = batch_idx.detach().cpu().long().numpy()
    
    # Call original function
    cdef tuple result = build_balltree(data_np, batch_idx_np)
    
    # Convert back to torch and return
    return (
        torch.from_numpy(result[0]).to(device),
        torch.from_numpy(result[1]).to(device)
    )


def build_balltree_with_offsets_torch(data: object, batch_offsets: object):
    """Build ball trees using PyTorch tensors with explicit batch offsets.
    
    Parameters
    ----------
    data : torch.Tensor of shape (num_samples, num_features)
        Input data points. Can be on CPU or CUDA.
    batch_offsets : torch.Tensor of shape (num_batches + 1,)
        Indices that define batch boundaries.
        
    Returns
    -------
    tuple
        (indices, mask) tuple where both elements are torch.Tensor
        on the same device as the input.
    """
    # Validate inputs are torch tensors
    if not isinstance(data, torch.Tensor) or not isinstance(batch_offsets, torch.Tensor):
        raise ValueError("Inputs must be PyTorch tensors")
        
    if data.dim() != 2:
        raise ValueError(f"Data must be 2D tensor, got shape {tuple(data.shape)}")
        
    if batch_offsets.dim() != 1:
        raise ValueError(f"batch_offsets must be 1D tensor, got shape {tuple(batch_offsets.shape)}")

    # Store original device
    device = data.device
    
    # Convert to numpy arrays
    cdef cnp.ndarray data_np = data.detach().cpu().double().numpy()
    cdef cnp.ndarray batch_offsets_np = batch_offsets.detach().cpu().long().numpy()
    
    # Call original function
    cdef tuple result = build_balltree_with_offsets(data_np, batch_offsets_np)
    
    return (
        torch.from_numpy(result[0]).to(device),
        torch.from_numpy(result[1]).to(device)
    )