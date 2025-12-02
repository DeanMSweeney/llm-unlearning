"""
Main entry point for bias mitigation through gradient-based unlearning.
Uses WinoGender dataset to reduce gender bias in language models by selectively
updating parameters to reduce the model's preference for stereotypical gender associations.
"""

from sqlite3 import NotSupportedError
import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModelForMaskedLM
import numpy as np
import random
import argparse
import logging
import os
from tqdm import tqdm
import json
import sys
from .utils.build_dataset import WGDataset
from .utils.utils import set_random_seed, get_params_map, get_all_model_grads, accumulate_grad, create_param_partition
from .utils.consts import PAD_TOKEN, MASK_TOKEN
from trainer import Unbias

logger = logging.getLogger(__name__)

def main(
    model_path_or_name: str,  # HuggingFace model name or local path
    num_epochs: int = 5,  # Number of training epochs
    is_mlm: bool = True,  # Whether model is masked language model (BERT-style)
    k: int = 10000,  # Number of top parameters to update per batch (for selective updating)
    sim_batch_size: int = -1,  # Batch size for similarity computation (-1 = use per-batch params)
    use_advantaged_for_grad: bool=True,  # If True, minimize advantaged gender; if False, maximize disadvantaged
    agg_input: bool=True,  # If True, aggregate gradients at input layer; if False, at output layer
    proportion_dev=0.75,  # Proportion of data used for dev set
    do_dynamic_gradient_selection: bool=False,  # Dynamically choose which gender to target
    lr: float = 1e-5,  # Learning rate
    momentum: float = 0.9,  # SGD momentum
    batch_size: int = 16,  # Training batch size
    seed: int = 89793,  # Random seed for reproducibility
    num_workers: int = 4,  # DataLoader workers
    start_at_epoch: int = 0,  # Resume from checkpoint epoch
    dedupe=''  # Experiment identifier for saving checkpoints
):
    """
    Main training function for bias mitigation.

    The approach:
    1. Load a pre-trained masked language model
    2. Use WinoGender dataset to identify gender bias
    3. Selectively update model parameters to reduce bias
    4. Target either the advantaged or disadvantaged gender based on configuration
    """ 

    # Set random seeds for reproducibility
    logger.info(f'Seed is {seed}')
    set_random_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer (RoBERTa needs special handling for spaces)
    if is_mlm:
        if 'roberta' in model_path_or_name:
            tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, add_prefix_space=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, pad_token=PAD_TOKEN, mask_token=MASK_TOKEN)
        raise ValueError('No non-mlms currently')

    # Load pre-trained model
    model = AutoModelForPreTraining.from_pretrained(model_path_or_name)
    model.resize_token_embeddings(len(tokenizer))

    model.train()
    model.to(device)

    # Load WinoGender dataset for bias evaluation and mitigation
    dataset = WGDataset('../data/wg.tsv', '../data/wg_stats.tsv', tokenizer)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, collate_fn=WGDataset.collate_batch_creator(tokenizer),
                                num_workers=num_workers,
                                )

    # Use SGD optimizer (simpler optimization for controlled updates)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Configure gradient aggregation dimension
    # -1 = input aggregation (aggregate over input tokens)
    # -2 = output aggregation (aggregate over output features)
    agg_dim = -1 if agg_input else -2

    # Configure which gradient to use:
    # "negative" = minimize advantaged gender (default)
    # "positive" = maximize disadvantaged gender
    which_grad = "negative" if use_advantaged_for_grad else "positive"

    logger.info('Retraining now')

    # Initialize and run the Unbias trainer
    rt = Unbias(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        device=device,
        batch_size=batch_size,
        k=k,  # Number of top-k parameters to update
        is_mlm=is_mlm,
        num_epochs=num_epochs,
        sim_batch_size=sim_batch_size,
        which_grad=which_grad,  # Which gradient direction to use
        proportion_dev=proportion_dev,
        do_dynamic_gradient_selection=do_dynamic_gradient_selection,
        agg_dim=agg_dim,  # Dimension for gradient aggregation
        start_at_epoch=start_at_epoch,
        dedupe=dedupe,  # Experiment name for checkpointing
        model_name=model_path_or_name
    )

if __name__=='__main__':
    # ============================================================================
    # Command-line argument parsing
    # ============================================================================
    parser = argparse.ArgumentParser(description='Bias mitigation through gradient-based unlearning on WinoGender dataset')

    # Model configuration
    parser.add_argument('-m', type=str, required=True, dest='model_path_or_name',
                        help='HuggingFace model name or path (e.g., bert-base-uncased)')

    # Optimization parameters
    parser.add_argument('-l', type=float, default=1e-5, dest='lr',
                        help='Learning rate for SGD optimizer')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')

    # Parameter selection strategy
    parser.add_argument('-k', type=int, default=10000, dest='k',
                        help='Number of top-k parameters to update per batch (selective updating)')
    parser.add_argument('--use-full-grad', dest='k', action='store_const', const=None,
                        help='Use full gradient instead of top-k parameter selection')

    # Training configuration
    parser.add_argument('-n', type=int, default=5, dest='num_epochs',
                        help='Total number of epochs to train')
    parser.add_argument('-b', type=int, default=16, dest='batch_size',
                        help='Batch size for training')
    parser.add_argument('--start-at', type=int, default=0,
                        help='Resume from checkpoint epoch (e.g., 1 will train remaining epochs)')

    # Experiment tracking
    parser.add_argument('--dedupe', type=str, default='',
                        help='Experiment name (models saved to sim_checkpoints/{dedupe}/model_{epoch})')

    # Bias mitigation strategy
    parser.add_argument('--output-agg', dest='aggregation', action='store_const', const='output', default='input',
                        help='Use output layer aggregation instead of input layer (default: input)')
    parser.add_argument('--dynamic_gradient_selection', dest='dynamic_gradient_selection', action='store_true', default=False,
                        help='Dynamically choose advantaged/disadvantaged each batch (default: static based on WG stats)')
    parser.add_argument('--use-disadvantaged', dest='use_advantaged_for_grad', action='store_false', default=True,
                        help='Maximize disadvantaged group instead of minimizing advantaged (default: minimize advantaged)')
    parser.add_argument('--use-same-params', dest='sim_batch_size', action='store_const', const=None, default=-1,
                        help='Use same parameter set each epoch instead of selecting per batch (default: per-batch)')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Determine model type (currently only MLMs supported)
    is_mlm = 'bert' in args.model_path_or_name

    # Extract configuration from args
    sim_batch_size = args.sim_batch_size
    use_advantaged_for_grad = args.use_advantaged_for_grad
    agg_input = args.aggregation=='input'

    # Determine gradient direction strategy label for experiment naming
    if args.dynamic_gradient_selection:
        direction_selection = 'dynamic'
    elif args.use_advantaged_for_grad:
        direction_selection = 'adv'  # Minimize advantaged group
    else:
        direction_selection = 'disadv'  # Maximize disadvantaged group

    # Determine parameter selection strategy label for experiment naming
    if args.k is None:
        partition_usage = 'full_grad'  # Update all parameters
    elif args.sim_batch_size is None:
        partition_usage = 'all'  # Use all parameter partitions across epochs
    else:
        partition_usage = 'notall'  # Use subset of parameters per batch

    # Build experiment identifier for checkpoint organization
    # Format: {model}/{param_strategy}/{aggregation}/{direction}/{lr}/64/{k}
    dedupe_model_name = args.model_path_or_name.split('/')[-1]
    dedupe = f"{dedupe_model_name}/{partition_usage}/{'inp' if agg_input else 'outp'}/{direction_selection}/{args.lr}/64/{args.k}"

    # Run main training loop
    main(args.model_path_or_name, num_epochs=args.num_epochs, is_mlm=is_mlm, k=args.k,
        proportion_dev=0.5, do_dynamic_gradient_selection=args.dynamic_gradient_selection,
        sim_batch_size=sim_batch_size, use_advantaged_for_grad=use_advantaged_for_grad, agg_input=agg_input,
        lr=args.lr, momentum=args.momentum, batch_size=args.batch_size, start_at_epoch=args.start_at, dedupe=dedupe)


