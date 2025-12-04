"""
Dataset builders for bias evaluation benchmarks.
Includes loaders and PyTorch Dataset classes for StereoSet, Crows-Pairs, and WinoGender datasets.
"""

from typing import List, Tuple
import os
import csv
import torch
import torch.nn as nn
import torch.utils.data as data
import math
from utils.consts import PAD_VALUE

# ============================================================================
# File I/O Utility Functions
# ============================================================================

def save_sampled_sents(filename: str, sents: List[str]):
    """Save list of sentences to a file, one per line."""
    dir = os.path.split(filename)[0]
    if dir:
        os.makedirs(dir, exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as f:
        for sent in sents:
            if sent:
                f.write(sent)
                f.write('\n')

def load_sampled_sents(filename: str) -> List[str]:
    """Load sentences from a file, one per line."""
    with open(filename, 'r', encoding='utf-8') as f:
        sents = [line.strip() for line in f if line]

    return sents

def load_sampled_sents_2(filename: str) -> List[Tuple]:
    """Load tab-separated sentence pairs (sentence, target_word) from file."""
    with open(filename, 'r', encoding='utf-8') as f:
        examples = [tuple(line.strip().split('\t')) for line in f if line]

    return examples 

def load_wg_stats(filename):
    """
    Load WinoGender occupation statistics (gender proportions).
    Returns dict mapping occupation -> percentage (using Bergsma proportion).
    """
    pcts = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines()[1:]:  # Skip header
            if line:
                split_line = line.strip().split('\t')
                pcts[split_line[0]] = float(split_line[1])  # using the bergsma proportion
    return pcts

def load_sents_wg(filename: str) -> List[Tuple]:
    """
    Load WinoGender sentences from file.
    Returns tuples of (male_sent, female_sent, neutral_sent) and corresponding occupations.
    Each example has 3 variants differing by gendered pronouns.
    """
    sents = []
    occs = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines()[1:]:  # Skip header
            if line:
                split_line = line.strip().split('\t')
                sents.append(split_line[1])
                occs.append(split_line[0].split('.')[0])

    # Group sentences into triples (male, female, neutral)
    sent_triples = []
    trip_occ = []
    for i in range(0, len(sents), 3):
        sent_triples.append((sents[i], sents[i+1], sents[i+2]))
        trip_occ.append(occs[i])

    return sent_triples, trip_occ

def load_sents_crows(filename: str):
    """
    Load Crows-Pairs sentence pairs from CSV file.
    Returns list of (more_stereotypical_sent, less_stereotypical_sent) tuples.
    """
    with open(filename, 'r', encoding='utf-8', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        lines = list(csv_reader)
    sent_pairs = [(row[1], row[2]) for row in lines]
    return sent_pairs

# ============================================================================
# Dataset Classes
# ============================================================================

class StereoSetDataset(data.Dataset):
    """
    StereoSet dataset for measuring stereotypical biases in language models.
    Each example is a sentence with a masked target word that the model must predict.
    Examples come in triples: (stereotypical, anti-stereotypical, unrelated).
    """
    def __init__(self, dataset_file, tokenizer, max_len=64, dev=True, proportion_dev=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        examples = load_sampled_sents_2(dataset_file)
        
        tokenized_mask = tokenizer.mask_token_id
        tokenized_unk = tokenizer.unk_token_id

        sents, target_words = [list(unzipped) for unzipped in zip(*examples)]

        tokenized_sents = [tokenizer(sent, add_special_tokens=False)['input_ids'][:max_len] for sent in sents]
        tokenized_target_words = [tokenizer(target, add_special_tokens=False)['input_ids'] for target in target_words]

        # Find the index of target word in each sentence
        # Filter out examples where target word is not a single token or appears multiple times
        inds = [-1]*len(tokenized_target_words)
        for i, (tokenized_sent, tokenized_target_word) in enumerate(zip(tokenized_sents, tokenized_target_words)):
            if len(tokenized_target_word)!=1:  # Target must be single token
                inds[i] = -2
                continue

            tokenized_target = tokenized_target_word[0]

            if tokenized_target==tokenized_unk:  # Skip unknown tokens
                continue

            token_indices = [i for i, token in enumerate(tokenized_sent) if token==tokenized_target]
            if len(token_indices)!=1:  # Target must appear exactly once
                inds[i] = -3
                continue

            inds[i] = token_indices[0]

        # Remove entire triples if any example in the triple is invalid
        for i in range(2, len(inds), 3):
            if inds[i]<0 or inds[i-1]<0 or inds[i-2]<0:
                inds[i] = -4
                inds[i-1] = -4
                inds[i-2] = -4


        # Keep only valid examples
        kept_sents = [tokenized_sent for i, tokenized_sent in enumerate(tokenized_sents) if inds[i]>=0]
        kept_inds = [ind for ind in inds if ind>=0]
        kept_target_tokens = [tokens[0] for i, tokens in enumerate(tokenized_target_words) if inds[i]>=0]

        # Replace target word with [MASK] token
        for sent, ind in zip(kept_sents, kept_inds):
            sent[ind] = tokenized_mask

        examples = list(zip(kept_sents, kept_inds, kept_target_tokens))
        example_count = len(examples)//3
        if example_count*3!=len(examples):
            raise ValueError('malformed stereoset input file?')

        # Split into dev/test sets
        cutoff = math.floor(example_count * proportion_dev)*3
        if dev:
            self.examples = examples[:cutoff]
        else:
            self.examples = examples[cutoff:]

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        return len(self.examples)

    @staticmethod
    def collate_batch_creator(tokenizer):
        """
        Creates a collate function for DataLoader that pads sequences and creates attention masks.
        Returns: (padded_seqs, attention_masks, target_indices, target_tokens, labels)
        """
        def collate_batch(batch):
            sents, inds, target_tokens = [list(unzipped) for unzipped in zip(*batch)]

            seqs = [torch.tensor(sent) for sent in sents]
            inds = torch.LongTensor(inds)
            target_tokens = torch.LongTensor(target_tokens)

            # Pad sequences to same length
            padded_seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=tokenizer.pad_token_id)

            # Create attention masks (1 for real tokens, 0 for padding)
            attention_masks = torch.ones_like(padded_seqs)
            attention_masks[padded_seqs==tokenizer.pad_token_id] = 0

            # Create labels (padded positions set to PAD_VALUE for loss computation)
            labels = padded_seqs.detach().clone()
            labels[padded_seqs==tokenizer.pad_token_id] = PAD_VALUE

            return padded_seqs, attention_masks, inds, target_tokens, labels
        return collate_batch

class CrowsDataset(data.Dataset):
    """
    Crows-Pairs dataset for measuring stereotypical biases.
    Each example is a pair of sentences that differ by a single word (e.g., 'man' vs 'woman').
    The model's preference for one sentence over the other reveals bias.
    """
    def __init__(self, dataset_file, tokenizer, max_len=24):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        examples = load_sents_crows(dataset_file)
        
        tokenized_mask = tokenizer('[MASK]', add_special_tokens=False)['input_ids'][0]
        self.tokenized_unk = tokenizer('[UNK]', add_special_tokens=False)['input_ids'][0]

        # Process sentence pairs and find the differing token
        sent_pairs = []
        inds = []
        target_token_pairs = []
        skipped_count = 0
        for first_sent, second_sent in examples:
            ind, (t_first, first), (t_second, second) = self._tokenize_and_find_diff(first_sent, second_sent)
            if ind is None:  # Skip pairs that don't meet criteria (single token diff)
                skipped_count += 1
                continue
            sent_pairs.append((t_first, t_second))
            inds.append(ind)
            target_token_pairs.append((first, second))
        print(f'Skipped {skipped_count}/{len(examples)}')

        # Mask the differing token in each sentence
        for (first, second), ind in zip(sent_pairs, inds):
            first[ind] = tokenized_mask
            second[ind] = tokenized_mask

        self.examples = list(zip(sent_pairs, inds, target_token_pairs))

    def _tokenize_and_find_diff(self, first, second):
        """
        Tokenize sentence pair and find the single differing token.
        Returns None if sentences differ by more than one token or if tokens are unknown.
        """
        tokenized_first = self.tokenizer(first, add_special_tokens=False)['input_ids'][:self.max_len]
        tokenized_second = self.tokenizer(second, add_special_tokens=False)['input_ids'][:self.max_len]

        if len(tokenized_first)==len(tokenized_second):
            found_i, found_f, found_s = None, None, None
            for i, (f, s) in enumerate(zip(tokenized_first, tokenized_second)):
                if f==s:
                    continue
                else:
                    if found_i is not None:  # Only accept single token differences
                        return None, (None, None), (None, None)
                    found_i = i
                    found_f = f
                    found_s = s
            # Reject if no difference found or if either token is unknown
            if found_i is None or found_f==self.tokenized_unk or found_s==self.tokenized_unk:
                return None, (None, None), (None, None)
            return i, (tokenized_first, found_f), (tokenized_second, found_s)
        return None, (None, None), (None, None)

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        return len(self.examples)

    @staticmethod
    def collate_batch_creator(tokenizer):
        """
        Creates collate function that flattens sentence pairs into single batch.
        Interleaves first and second sentences: [sent1_ex1, sent2_ex1, sent1_ex2, sent2_ex2, ...]
        """
        def collate_batch(batch):
            sent_pairs, inds, target_token_pairs = [list(unzipped) for unzipped in zip(*batch)]

            # Flatten pairs into single list
            seqs = []
            for first_sent, second_sent in sent_pairs:
                seqs.append(torch.tensor(first_sent))
                seqs.append(torch.tensor(second_sent))

            # Duplicate indices for both sentences in pair
            full_inds = []
            for ind in inds:
                full_inds.append(ind)
                full_inds.append(ind)
            inds = torch.LongTensor(full_inds)

            # Flatten target tokens
            target_tokens = []
            for first_target, second_target in target_token_pairs:
                target_tokens.append(first_target)
                target_tokens.append(second_target)
            target_tokens = torch.LongTensor(target_tokens)

            padded_seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=tokenizer.pad_token_id)
            attention_masks = torch.ones_like(padded_seqs)
            attention_masks[padded_seqs==tokenizer.pad_token_id] = 0

            labels = padded_seqs.detach().clone()
            labels[padded_seqs==tokenizer.pad_token_id] = PAD_VALUE

            return padded_seqs, attention_masks, inds, target_tokens, labels
        return collate_batch

class WGDataset(data.Dataset):
    """
    WinoGender (WG) dataset for measuring gender bias in occupation-related contexts.
    Each example has male/female/neutral variants. The dataset identifies which gender
    is disadvantaged for each occupation (based on real-world statistics) and measures
    whether the model is biased toward the advantaged gender.
    """
    def __init__(self, dataset_file, stats_file, tokenizer, max_len=24):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        sent_triples, occupations = load_sents_wg(dataset_file)
        stat_pcts = load_wg_stats(stats_file)
        
        tokenized_mask = tokenizer('[MASK]', add_special_tokens=False)['input_ids'][0]

        # Process sentence triples and determine disadvantaged/advantaged genders
        sent_trips = []
        inds = []
        target_token_pairs = []
        for sent_trip, occ in zip(sent_triples, occupations):
            male, female, neutral = sent_trip
            ind, (t_male, m), (t_female, f), (t_neutral, n) = self._tokenize_and_find_diff(male, female, neutral)
            if ind is None:
                continue

            # Determine which gender is disadvantaged based on occupation statistics
            # If occupation is <50% one gender, that gender is disadvantaged
            """
                if stat_pcts[occ]<50:
                disadvantaged = t_female
                disadvantaged_token = f
                advantaged = t_male
                advantaged_token = m
            else:
                disadvantaged = t_male
                disadvantaged_token = m
                advantaged = t_female
                advantaged_token = f
            
            """
        
            sent_trips.append((t_male, t_female, t_neutral))
            inds.append(ind)
            target_token_pairs.append((m, f, n))

        # Replace gendered pronouns with [MASK]
        for (t_male, t_female, t_neutral), ind in zip(sent_trips, inds):
            t_male[ind] = tokenized_mask
            t_female[ind] = tokenized_mask
            t_neutral[ind] = tokenized_mask

        self.examples = list(zip(sent_trips, inds, target_token_pairs))
    
    def _tokenize_and_find_diff(self, male, female, neutral):
        """
        Tokenize sentence triple and find the position where they differ (gendered pronoun).
        Returns the index and the differing tokens for each variant.
        """
        tokenized_male = self.tokenizer(male, add_special_tokens=False)['input_ids'][:self.max_len]
        tokenized_female = self.tokenizer(female, add_special_tokens=False)['input_ids'][:self.max_len]
        tokenized_neutral = self.tokenizer(neutral, add_special_tokens=False)['input_ids'][:self.max_len]

        if len(tokenized_male)==len(tokenized_female)==len(tokenized_neutral):
            for i, (m, f, n) in enumerate(zip(tokenized_male, tokenized_female, tokenized_neutral)):
                if m==f==n:
                    continue
                else:  # Found the differing position (gendered pronoun)
                    return i, (tokenized_male, m), (tokenized_female, f), (tokenized_neutral, n)
        return None, None, None, None

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        return len(self.examples)

    @staticmethod
    def collate_batch_creator(tokenizer):
        """
        Creates collate function that keeps male/female/neutral sentences separate.
        Returns tuple triples for sequences, masks, target tokens, and labels.
        This allows comparing model predictions across all three gender variants.
        """
        def collate_batch(batch):
            sent_triples, inds, target_token_triples = [list(unzipped) for unzipped in zip(*batch)]
            male_sents, female_sents, neutral_sents = [list(unzipped) for unzipped in zip(*sent_triples)]
            male_targets, female_targets, neutral_targets = [list(unzipped) for unzipped in zip(*target_token_triples)]

            male_seqs = [torch.tensor(sent) for sent in male_sents]
            female_seqs = [torch.tensor(sent) for sent in female_sents]
            neutral_seqs = [torch.tensor(sent) for sent in neutral_sents]
            inds = torch.LongTensor(inds)
            male_target_tokens = torch.LongTensor(male_targets)
            female_target_tokens = torch.LongTensor(female_targets)
            neutral_target_tokens = torch.LongTensor(neutral_targets)

            # Pad sequences separately for each gender variant
            padded_male_seqs = nn.utils.rnn.pad_sequence(male_seqs, batch_first=True, padding_value=tokenizer.pad_token_id)
            padded_female_seqs = nn.utils.rnn.pad_sequence(female_seqs, batch_first=True, padding_value=tokenizer.pad_token_id)
            padded_neutral_seqs = nn.utils.rnn.pad_sequence(neutral_seqs, batch_first=True, padding_value=tokenizer.pad_token_id)

            # Create attention masks
            male_attention_masks = torch.ones_like(padded_male_seqs)
            female_attention_masks = torch.ones_like(padded_female_seqs)
            neutral_attention_masks = torch.ones_like(padded_neutral_seqs)
            male_attention_masks[padded_male_seqs==tokenizer.pad_token_id] = 0
            female_attention_masks[padded_female_seqs==tokenizer.pad_token_id] = 0
            neutral_attention_masks[padded_neutral_seqs==tokenizer.pad_token_id] = 0

            # Create labels
            male_labels = padded_male_seqs.detach().clone()
            male_labels[padded_male_seqs==tokenizer.pad_token_id] = PAD_VALUE
            female_labels = padded_female_seqs.detach().clone()
            female_labels[padded_female_seqs==tokenizer.pad_token_id] = PAD_VALUE
            neutral_labels = padded_neutral_seqs.detach().clone()
            neutral_labels[padded_neutral_seqs==tokenizer.pad_token_id] = PAD_VALUE

            return (padded_male_seqs, padded_female_seqs, padded_neutral_seqs), \
                    (male_attention_masks, female_attention_masks, neutral_attention_masks), \
                    inds, \
                    (male_target_tokens, female_target_tokens, neutral_target_tokens), \
                    (male_labels, female_labels, neutral_labels)
        return collate_batch