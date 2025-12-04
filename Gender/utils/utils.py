import os
import torch
import numpy as np
from typing import Optional, Literal, Union, Tuple, Dict, List
import logging
import random

logger = logging.getLogger(__name__)

def set_random_seed(seed: int) -> None: 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _get_checkpoint_dir(epoch: int, dedupe: str = '') -> str:
    """
    Generate the checkpoint directory path for a given epoch.

    Args:
    ---------
        epoch: Epoch number for the checkpoint
        dedupe: Optional string identifier for deduplication/experiment naming

    Returns:
        String path to the checkpoint directory
    """
    if dedupe:
        return f'models/{dedupe}/model_{epoch}'
    else:
        return f'models/model_{epoch}'

def save_model(model, tokenizer, epoch: int, dedupe: str = '') -> str:
    """
    Save a model and tokenizer checkpoint to disk.

    Args:
    ---------
        model: The model to save
        tokenizer: The tokenizer to save
        epoch: Current epoch number
        dedupe: Optional string identifier for deduplication/experiment naming

    Returns:
        Path to the directory where the model was saved
    """
    output_dir = _get_checkpoint_dir(epoch, dedupe=dedupe)
    logger.info(f'Saving model at {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir

def _get_all_indices(shape: Tuple[int, ...], dim_to_agg: int) -> List[Tuple]:
    """
    Generate all index combinations for parameter partitioning.

    Recursively generates index tuples for accessing parameter slices,
    treating one dimension (dim_to_agg) as a range for aggregation.

    Args:
    ---------
        shape: Shape tuple of the parameter tensor
        dim_to_agg: Dimension to aggregate over (-1 for last dim, -2 for second-to-last)

    Returns:
        List of index tuples, where the aggregation dimension contains a range object
    """
    dims = len(shape)

    def _get_all_indices_helper(curr_inds, permute_from_dim):
        # Base case: reached the end of dimensions
        if permute_from_dim == dims:
            if dim_to_agg == -1:
                # Remove the range from the last dimension
                return [curr_inds[:-1]]
            else:
                return [curr_inds]
        else:
            # Generate all possible values for this dimension
            ind_vals = range(shape[permute_from_dim])

            # If this is the aggregation dimension, use a range instead of individual indices
            if permute_from_dim == dims + dim_to_agg:
                ind_vals = [ind_vals]

            # Recursively generate all combinations
            all_permutations = []
            for outer_perm_val in ind_vals:
                curr_inds = curr_inds[:permute_from_dim] + (outer_perm_val,) + curr_inds[permute_from_dim+1:]
                all_permutations.extend(_get_all_indices_helper(curr_inds, permute_from_dim+1))
            return all_permutations

    init_inds = (0,)*dims
    return _get_all_indices_helper(init_inds, 0)

def create_param_partition(params: Dict[str, torch.nn.Parameter], dim_to_agg: int = -1) -> List[Tuple[str, Optional[Tuple]]]:
    """
    Partition model parameters into vectors for gradient-based selection.

    Creates a list of (param_name, indices) tuples where each entry represents
    a parameter slice to be treated as a unit for gradient calculations.

    Args:
    ---------
        params: Dictionary mapping parameter names to parameter tensors
        dim_to_agg: Dimension to aggregate over
                    -1 for "input" aggregation (last dimension)
                    -2 for "output" aggregation (second-to-last dimension)

    Returns:
        List of tuples (param_name, indices) representing parameter partitions

    Note:
        Can only guarantee aggregation for dims -1, -2 (other dims might not exist).
        Code works for other dims if they exist.
    """
    assert dim_to_agg in {-1, -2}, 'can only guarantee aggregation for dims -1, -2 (other dims might not exist). Code works for other dims if they exist..'

    param_partition = []
    for param_name, param in params.items():
        # Skip parameters that don't require gradients
        if not param.requires_grad:
            continue

        dims = len(param.shape)
        if dims > 1:
            # Split multi-dimensional parameters into vectors based on aggregation dimension
            for indices in _get_all_indices(param.shape, dim_to_agg=dim_to_agg):
                param_partition.append((param_name, indices))
        elif dims == 1:
            # 1D parameters are already vectors, no aggregation needed
            param_partition.append((param_name, None))
        else:
            # 0D parameters (scalars) shouldn't exist in neural networks
            raise ValueError(f'param {param_name} has dimension of 0, shape is: {param.shape}')

    return param_partition

def get_params_map(model) -> Dict[str, torch.nn.Parameter]:
    """
    Create a dictionary mapping parameter names to parameter tensors.

    Args:
    ---------
        model: PyTorch model to extract parameters from

    Returns:
        Dictionary mapping parameter names (strings) to parameter tensors
    """
    params_map = {}
    for param_name, param in model.named_parameters():
        params_map[param_name] = param
    return params_map

def get_all_model_grads(model, clear_grads_after: bool = False) -> Dict[str, Optional[torch.Tensor]]:
    """
    Extract all gradients from a model's parameters.

    Collects gradients for all parameters that require gradients, handling cases
    where gradients may not exist (e.g., unused parameters).

    Args:
    ---------
        model: PyTorch model to extract gradients from
        clear_grads_after: If True, zero out gradients after extraction

    Returns:
        Dictionary mapping parameter names to gradient tensors (on CPU)
        Parameters without gradients map to None
    """
    params = list(model.named_parameters())
    param_grads = {}
    problems = []

    for i, (param_name, param) in enumerate(params):
        if param.requires_grad:
            try:
                # Detach and move gradient to CPU for storage
                param_grad = param.grad.detach().cpu()
            except (AttributeError, RuntimeError):
                # Gradient doesn't exist (parameter wasn't used in forward/backward pass)
                # AttributeError: param.grad is None
                # RuntimeError: gradient computation issues
                problems.append(f'{i}: {param_name}')
                param_grad = None
            param_grads[param_name] = param_grad

    if problems:
        logger.debug(f'Parameters without gradients: {problems}')

    if clear_grads_after:
        model.zero_grad()

    return param_grads

def accumulate_grad(accumulator_grad: Optional[Dict[str, Optional[torch.Tensor]]], grad: Dict[str, Optional[torch.Tensor]]) -> Dict[str, Optional[torch.Tensor]]:
    """
    Accumulate gradients across multiple batches.

    Adds new gradients to an accumulator dictionary, handling initialization
    and ensuring consistency of None values across parameters.

    Args:
    ---------
        accumulator_grad: Dictionary of accumulated gradients (or None for first batch)
        grad: Dictionary of new gradients to add

    Returns:
        Updated accumulator dictionary with summed gradients

    Raises:
        RuntimeError: If a parameter has a gradient in one dict but not the other
    """
    # Initialize accumulator on first call
    if accumulator_grad is None:
        return grad

    # Add gradients element-wise for each parameter
    for param_name, curr_grad_val in accumulator_grad.items():
        new_grad_val = grad[param_name]

        # Ensure both dicts have consistent None/non-None gradients
        if (curr_grad_val is None) != (new_grad_val is None):
            raise RuntimeError(f'Inconsistent gradient existence for param {param_name}')

        # Sum the gradients if they exist, otherwise keep as None
        if curr_grad_val is not None:
            accumulator_grad[param_name] = (curr_grad_val + new_grad_val).detach()
        # If both are None, no action needed (already None in accumulator)

    return accumulator_grad