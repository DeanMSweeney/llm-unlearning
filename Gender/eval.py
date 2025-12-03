"""
Evaluation script for measuring model bias using the StereoSet benchmark.

This script evaluates masked language models (MLMs) on the StereoSet dataset across
multiple domains (gender, profession, race, religion). It computes stereotype scores (SS),
language modeling scores (LMS), and idealized context association test scores (ICAT).
"""

import torch
import torch.utils.data as data
from transformers import AutoTokenizer, AutoModelForMaskedLM
import argparse
import logging
import os
import json
from tqdm import tqdm
from utils.build_dataset import StereoSetDataset
from utils.eval_ss import compute_stereoset_scores, eval_mlm_given_dataloader

logger = logging.getLogger(__name__)

# Bias domains to evaluate on
DOMAINS = ['gender', 'profession', 'race', 'religion']

def main(model_type, model_name, ss_dev_proportion=0.5, models_loc=f'./models', pretrained_only=False, shuffled=True):
    """
    Main evaluation pipeline for StereoSet benchmark.

    Args:
        model_type: Base model type (e.g., 'bert-base-cased', 'roberta-base')
        model_name: Name identifier for the model being evaluated
        ss_dev_proportion: Proportion of StereoSet to use for dev set (rest is test)
        models_loc: Base directory where model checkpoints are stored
        pretrained_only: If True, only evaluate the pretrained model without fine-tuned versions
        shuffled: Whether to use shuffled or unshuffled data
    """
    # Create unique identifier for result files based on data configuration
    dedupe = f"{ss_dev_proportion}"
    if not shuffled:
        dedupe += "_unsh"

    # Set up device and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # RoBERTa requires special tokenization with prefix space
    if 'roberta' in model_type:
        tokenizer = AutoTokenizer.from_pretrained(model_type, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_type)

    logger.info(f'Initialized tokenizer')

    # Evaluate all models and construct hierarchical results map
    results_map = construct_results_map(model_name, tokenizer, ss_dev_proportion, device, models_loc=models_loc, pretrained_only=pretrained_only, shuffled=shuffled)

    # For pretrained-only evaluation, save and return early
    if pretrained_only:
        save_result(results_map, f'results/pretrained_{model_name}_results_map.json', dedupe=dedupe)
        return

    # Save full results map
    save_result(results_map, f'results/{model_name}_results_map.json', dedupe=dedupe)
    logger.info(f'Got results map {json.dumps(results_map, indent=4)}')

    # Find best model at each hyperparameter level based on dev set performance
    best_results = best_results_per_level(results_map)
    logger.info(f'Got best results {json.dumps(best_results, indent=4)}')

    # Extract all individual model results (removing the hierarchical structure)
    all_results_list = remove_final_level_results(best_results)
    all_results = {result['path']: result for result in all_results_list}
    logger.info(f'Got all results {json.dumps(all_results, indent=4)}')

    # Save best models per level and all individual results
    save_result(best_results, f'results/best_{model_name}_results.json', dedupe=dedupe)
    save_result(all_results, f'results/all_{model_name}_results.json', dedupe=dedupe)

def construct_ss_dataloaders(tokenizer, ss_dev_proportion, shuffled=True):
    """
    Create dataloaders for StereoSet evaluation across all domains.

    Args:
        tokenizer: Tokenizer for encoding text
        ss_dev_proportion: Proportion of data to use for dev split
        shuffled: Whether to use shuffled data files

    Returns:
        Dictionary mapping domains to their dev/test dataloaders and dataset counts
    """
    # Batch size must be divisible by 3 (stereo, anti-stereo, unrelated triplets)
    batch_size = 63

    dataloader_map = {}
    for domain in DOMAINS:
        dataloader_map[domain] = {}

        # Load domain-specific dataset (shuffled or unshuffled version)
        dataset_path = f'../data/stereoset_eval/{domain}{"" if shuffled else "_unsh"}.txt'

        # Create dev and test datasets/dataloaders
        dev_dataset = StereoSetDataset(dataset_path, tokenizer, dev=True, proportion_dev=ss_dev_proportion)
        dev_dataloader = data.DataLoader(dev_dataset, batch_size=batch_size,
                                        collate_fn=StereoSetDataset.collate_batch_creator(tokenizer))
        test_dataset = StereoSetDataset(dataset_path, tokenizer, dev=False, proportion_dev=ss_dev_proportion)
        test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size,
                                        collate_fn=StereoSetDataset.collate_batch_creator(tokenizer))

        # Store dataloaders and counts for this domain
        dataloader_map[domain]['dev'] = dev_dataloader
        dataloader_map[domain]['test'] = test_dataloader
        dataloader_map[domain]['dev_count'] = len(dev_dataset)
        dataloader_map[domain]['test_count'] = len(test_dataset)

    return dataloader_map

def construct_results_map(model_name, tokenizer, ss_dev_proportion, device, models_loc, pretrained_only=False, shuffled=True):
    """
    Evaluate all models and organize results in a hierarchical structure by hyperparameters.

    Args:
        model_name: Base name for the model
        tokenizer: Tokenizer for the model
        ss_dev_proportion: Proportion for dev/test split
        device: Device to run evaluation on
        models_loc: Base directory containing models
        pretrained_only: Whether to only evaluate the pretrained model
        shuffled: Whether to use shuffled data

    Returns:
        Hierarchical dictionary of results organized by hyperparameter directories
    """
    base_dir = f'{models_loc}/{model_name}/'

    # Find all model checkpoints (leaf directories in the model hierarchy)
    if not pretrained_only:
        model_paths = [
            x[0]
            for x in os.walk(base_dir)
            if not x[1] # only leaf directories (no subdirectories)
        ]
    else:
        # For pretrained, just use the model name directly
        model_paths = [model_name]

    logger.info(f'Found {len(model_paths)} models to evaluate.')
    logger.debug(model_paths)

    logger.info('Constructing StereoSet dataloaders')

    # Create dataloaders for all domains (reused across all models)
    dataloader_map = construct_ss_dataloaders(tokenizer, ss_dev_proportion, shuffled=shuffled)

    logger.info('Constructed StereoSet dataloaders')

    # Build hierarchical results tree organized by hyperparameters
    results_map = {}

    # Evaluate each model checkpoint
    for i, model_path in enumerate(tqdm(model_paths)):
        logger.info(f'Evaluating model at {model_path}')
        logger.info('='*50 + f'{i+1}/{len(model_paths)}' + '='*50)

        # Load model and set to eval mode
        model = AutoModelForMaskedLM.from_pretrained(model_path)
        model.eval()
        model.to(device)

        # Store results for this model
        result = {
            'path' : model_path,
        }

        # Evaluate on each domain (gender, profession, race, religion)
        for domain, dataloaders in dataloader_map.items():
            ss_dev_dataloader = dataloaders['dev']
            ss_test_dataloader = dataloaders['test']

            # Compute StereoSet scores on both dev and test sets
            result[domain] = {
                'dev' : generate_ss_result(model, tokenizer, device, ss_dev_dataloader),
                'test' : generate_ss_result(model, tokenizer, device, ss_test_dataloader),
                'dev_count': dataloaders['dev_count'],
                'test_count': dataloaders['test_count'],
            }

        logger.info(f'Got result: {json.dumps(result, indent=4)}')
        logger.info('='*100)

        # Organize results hierarchically by hyperparameter directory structure
        curr_map = results_map
        if not pretrained_only:
            model_name = os.path.basename(model_path)
            # Parse directory path to extract hyperparameter hierarchy
            split_path = os.path.split(model_path[len(base_dir):])[0].split('/')
            # Navigate/create nested dictionaries for each hyperparameter level
            for hyperparam in split_path:
                if hyperparam not in curr_map:
                    curr_map[hyperparam] = {}
                curr_map = curr_map[hyperparam]
        curr_map[model_name] = result

    return results_map
 
def generate_ss_result(model, tokenizer, device, ss_dataloader):
    """
    Evaluate a model on StereoSet data and compute bias metrics.

    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        device: Device to run on
        ss_dataloader: DataLoader with StereoSet examples

    Returns:
        Dictionary with SS (stereotype score), LMS (language modeling score),
        and ICAT (idealized CAT score) metrics
    """
    # Get raw preference counts from the model
    results = eval_mlm_given_dataloader(model, tokenizer, device, ss_dataloader)
    # Compute normalized bias scores
    ss_score, lms_score, icat_score = compute_stereoset_scores(*results)
    return {
        'ss': ss_score,  # Stereotype score (0.5 = unbiased, 1.0 = max bias)
        'lms': lms_score,  # Language modeling score (1.0 = best)
        'icat': icat_score,  # Idealized CAT (combines SS and LMS)
    }   

def default_result():
    """Initialize default result structure with no best model."""
    return {
        'best_model': None
    }

def score_result(result, stage):
    """
    Compute a criterion score for model selection based on stereotype scores.

    Args:
        result: Model evaluation results
        stage: 'dev' or 'test' - which split to score

    Returns:
        Negative weighted average SS score across gender and profession domains
        (more negative = better, as we want SS closer to 0.5)
    """
    unnormalized_score = 0
    total_count = 0
    # Only use gender and profession domains for model selection
    domains = ['gender', 'profession']
    for domain in domains:
        # Weight each domain's SS score by its dataset size
        unnormalized_score += result[domain][stage]['ss']*result[domain][f'{stage}_count']
        total_count += result[domain][f'{stage}_count']
    score = unnormalized_score/total_count
    return -score # wrong version but it's what was used for most, fix result outputs eventually..
    # return -abs(score-0.5) # correct version: maximize distance from bias

def result_better_than(result_1, result_2, scorer=score_result):
    """
    Compare two model results to determine which is better based on dev set performance.

    Args:
        result_1: First model's results
        result_2: Second model's results
        scorer: Function to compute criterion score

    Returns:
        True if result_1 is better than result_2, False otherwise
    """
    scores = []
    for result in [result_1, result_2]:
        if result is None:
            scores.append(-float('inf'))
            continue
        # Compute and cache criterion scores if not already present
        if 'dev_criterion_score' not in result:
            for stage in ['dev', 'test']:
                score = scorer(result, stage)
                result[f'{stage}_criterion_score'] = score
        scores.append(result['dev_criterion_score'])
    # Compare based on dev set criterion score (higher is better)
    return scores[0] > scores[1]

def best_results_per_level(results_map, key_name='', scorer=score_result):
    """
    Recursively find the best model at each level of the hyperparameter hierarchy.

    This function traverses the hierarchical results structure and identifies the
    best-performing model at each level based on dev set performance. The hierarchy
    typically reflects hyperparameter choices (e.g., learning_rate/batch_size/model_checkpoint).

    Args:
        results_map: Hierarchical dictionary of results
        key_name: Current key name (used to detect leaf nodes)
        scorer: Function to score model results

    Returns:
        Dictionary with 'best_model' at root and nested structure preserving hierarchy
    """
    # Base case: reached a leaf node (actual model checkpoint)
    if 'model_' in key_name:
        return {
            'best_model': results_map,
        }
    else:
        # Recursive case: internal node (hyperparameter level)
        output_map = default_result()
        for child in results_map:
            # Recursively get best results for this child
            child_best = best_results_per_level(results_map[child], key_name=child, scorer=scorer)
            output_map[child] = child_best
            # Update best model at this level if child's best is better
            if result_better_than(child_best['best_model'], output_map['best_model'], scorer=scorer):
                output_map['best_model'] = child_best['best_model']
        return output_map

def remove_final_level_results(best_results_map, removed_results=[]):
    """
    Extract all individual model results from the hierarchical structure.

    This flattens the hierarchy by removing the final level (actual model checkpoints)
    while preserving the hierarchical structure of hyperparameter levels.

    Args:
        best_results_map: Hierarchical results from best_results_per_level()
        removed_results: Accumulator for removed model results

    Returns:
        List of all individual model results
    """
    to_remove_keys = []
    for child, child_map in best_results_map.items():
        if child=='best_model':
            continue  # Skip the 'best_model' metadata field
        elif 'model_' in child:
            # Found a leaf (actual model), mark for removal
            to_remove_keys.append(child)
        else:
            # Recursively process hyperparameter levels
            remove_final_level_results(child_map, removed_results=removed_results)

    # Extract and collect all leaf model results
    for child in to_remove_keys:
        removed_results.append(best_results_map.pop(child)['best_model'])

    return removed_results

def save_result(result, output_file, dedupe=None):
    """
    Save evaluation results to a JSON file.

    Args:
        result: Results dictionary to save
        output_file: Output file path
        dedupe: Optional prefix for the filename (used for different data configurations)
    """
    import os
    # Add dedupe prefix if provided (for tracking different data splits/configurations)
    if dedupe:
        file_name = f'{dedupe}{output_file}'
    else:
        file_name = f'{output_file}'

    # Ensure output directory exists
    os.makedirs(os.path.split(file_name)[0], exist_ok=True)

    # Write results as formatted JSON
    with open(file_name, 'w') as f:
        json.dump(result, f, indent=4)

if __name__=='__main__': 
    MODEL_CHOICES = [
        'bert-base-cased', 
        'bert-base-uncased', 
        'albert-base-v2', 
        'roberta-base', 
    ]
    parser = argparse.ArgumentParser(description = 'This script uses prompts to sample sentences from an LM')
    parser.add_argument('-t', type=str, required=True, dest='model_type', choices=MODEL_CHOICES, help='the model type to evaluate')
    parser.add_argument('-m', type=str, required=True, dest='model_name', help='the model "name" to evaluate')
    parser.add_argument('--ss_dev_proportion', type=float, default=0.5, help='proportion of the original StereoSet dev set to use as our dev set')
    parser.add_argument('--models_base_loc', type=str, default='./models', help='relative path to the root of all the model checkpoints')
    parser.add_argument('--pretrained_only', dest='pretrained_only', action='store_true', default=False, help='to only evaluate the pretrained model (default: evaluate only all tuned models)')
    parser.add_argument('--unshuffled', dest='shuffled', action='store_false', default=True, help='to use unshuffled data (default: uses shuffled data)')

    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/evaluate_models_{args.model_name}.log" if not args.pretrained_only else f"logs/evaluate_pretrained_{args.model_name}.log"
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_filename), 
            # logging.StreamHandler(sys.stdout), 
        ],
    )
    
    main(args.model_type, args.model_name, ss_dev_proportion=args.ss_dev_proportion, models_loc=args.models_base_loc, pretrained_only=args.pretrained_only, shuffled=args.shuffled)