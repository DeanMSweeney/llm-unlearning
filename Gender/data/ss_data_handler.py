"""
StereoSet Data Handler

Processes the StereoSet dataset to extract sentences with bias labels and target words.
Organizes data by bias type (gender, profession, race, religion) and writes to files
for evaluation purposes.
"""

import json
import os
import numpy as np

# Load the StereoSet development dataset
with open('./stereoset_dev.json', 'r') as f:
    dataset = json.load(f)

# Configuration variables
BLANK = 'BLANK'  # Token placeholder in sentences
TARGET_BLANK = True  # If True, extract the word that replaces BLANK; if False, use the specified target
BIAS_TYPE = ''  # Filter by bias type: 'gender', 'profession', 'race', 'religion', or empty for all
SHUFFLE = True  # Whether to shuffle the dataset order

# Specify which sentence labels to include in the output
LABELS_TO_KEEP = ['stereotype']
if TARGET_BLANK:

    LABELS_TO_KEEP.append('anti-stereotype')
    LABELS_TO_KEEP.append('unrelated')

def extract_target_word_inds(context, target_word=BLANK):
    """
    Find the start and end indices of the target word in the context sentence.

    Args:
        context: The context sentence containing the target word
        target_word: The word to locate (default: BLANK token)

    Returns:
        Tuple of (start_index, end_index) for the target word

    Raises:
        ValueError: If the target word is not found in the context
    """
    context = context.upper()
    target_word = target_word.upper()
    # inefficient but whatever, 1 time thing only for stereoset
    for i in range(len(context)):
        if context[i:i+len(target_word)] == target_word:
            return (i, i+len(target_word))
    raise ValueError(f'Context sentence "{context}" does not contain "{target_word}"')

def extract_target_word(sentence, context, target_word_inds, target_blank=True):
    """
    Extract the actual target word from a sentence based on indices from the context.

    Args:
        sentence: The complete sentence with the actual word (not BLANK)
        context: The template context sentence containing BLANK or target word
        target_word_inds: Tuple of (start, end) indices for the target word location
        target_blank: If True, extract from sentence at BLANK position; if False, use context target

    Returns:
        The extracted target word string, or None if extraction fails
    """
    start_ind = target_word_inds[0]
    end_ind = target_word_inds[1]-len(context)
    if end_ind==0:
        end_ind = len(sentence)

    if target_blank:
        # Extract the word that replaced BLANK in the sentence
        target_word = sentence[start_ind:end_ind]
    else:
        # Use the specified target word from context, trying various case transformations
        target_word = context[start_ind:end_ind]
        if ' ' in target_word and target_word not in sentence:
            target_word_split = target_word.split(' ')
            target_word = ' '.join([target_word_split[0]] + [word_part.lower() for word_part in target_word_split[1:]])
        if target_word not in sentence:
            target_word = target_word.lower()
        if target_word not in sentence:
            target_word = target_word.capitalize()
        if target_word not in sentence:
            print(f'Could not extract: {target_word} | {sentence} | {context}')
            target_word = None
    return target_word

# Extract intrasentence items from the dataset
all_items = dataset['data']['intrasentence']

# Create indices for dataset iteration, optionally shuffling for randomness
indices = np.arange(len(dataset['data']['intrasentence']))
if SHUFFLE:
    rng = np.random.default_rng(14159)  # Fixed seed for reproducibility
    rng.shuffle(indices)

# Initialize data structures for collecting processed sentences
could_not_extract_count = 0
all_sents = []  # All sentences across all bias types
sents_per_domain = {  # Sentences organized by bias domain
    'gender': [],
    'profession': [],
    'race': [],
    'religion': [],
}
# Process each item in the dataset
for index in indices:
    item = all_items[index]
    bias_type = item['bias_type']
    context = item['context']
    target_word = BLANK if TARGET_BLANK else item['target']
    target_word_inds = extract_target_word_inds(context, target_word=target_word)

    # Create a dictionary mapping labels to their corresponding sentences
    sentences = {}
    for sent_obj in item['sentences']:
        sentences[sent_obj['gold_label']] = sent_obj['sentence']

    # Process each label type we want to keep (stereotype, anti-stereotype, unrelated)
    for label in LABELS_TO_KEEP:
        sentence = sentences[label]
        target_word = extract_target_word(sentence, context, target_word_inds, target_blank=TARGET_BLANK)
        if target_word is None:
            could_not_extract_count += 1
            continue
        # Store sentence-target pairs in both general and domain-specific lists
        all_sents.append((sentence,target_word))
        sents_per_domain[bias_type].append((sentence,target_word))

# Validate that all sentence-target pairs are non-empty
for sent, target in all_sents:
    if not (sent or target):
        raise ValueError

def write_sents_to_file(filename, sents):
    """
    Write sentence-target pairs to a tab-separated file.

    Args:
        filename: Path to the output file
        sents: List of (sentence, target_word) tuples to write

    Output format: Each line contains "sentence\ttarget_word\n"
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for sent, target in sents:
            if sent and target:
                f.write(sent)
                f.write('\t')
                f.write(target)
                f.write('\n')



# Write all sentences to a combined file
# Filename includes "_unsh" suffix if data is not shuffled
write_sents_to_file(f'../data/stereoset_eval/all{"" if SHUFFLE else "_unsh"}.txt', all_sents)

# Write domain-specific files for each bias type (gender, profession, race, religion)
for domain, sents in sents_per_domain.items():
    write_sents_to_file(f'../data/stereoset_eval/{domain}{"" if SHUFFLE else "_unsh"}.txt', sents)