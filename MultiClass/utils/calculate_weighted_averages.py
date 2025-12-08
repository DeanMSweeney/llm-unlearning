import json
import re
import argparse

def calculate_weighted_averages(log_file_path):
    """
    Calculate weighted averages of SS, LMS, and ICAT scores for the test set
    across gender, profession, race, and religion domains.

    Scores are weighted by test_count in each domain.
    """

    # Read the log file
    with open(log_file_path, 'r') as f:
        content = f.read()

    # Find the "Got best results" section
    # Look for the pattern and extract the JSON-like structure
    pattern = r'Got best results (\{[\s\S]*?\n\})'
    match = re.search(pattern, content)

    if not match:
        print("Could not find 'Got best results' section")
        return

    # Extract and parse the JSON
    json_str = match.group(1)
    data = json.loads(json_str)

    # Extract test scores and counts for each domain
    domains = ['gender', 'profession', 'race', 'religion']
    best_model = data['best_model']

    scores = {
        'ss': [],
        'lms': [],
        'icat': []
    }
    weights = []

    print("Individual Domain Results (Test Set):")
    print("-" * 70)

    for domain in domains:
        if domain in best_model:
            test_data = best_model[domain]['test']
            test_count = best_model[domain]['test_count']

            print(f"\n{domain.capitalize()}:")
            print(f"  Test Count: {test_count}")
            print(f"  SS:   {test_data['ss']:.6f}")
            print(f"  LMS:  {test_data['lms']:.6f}")
            print(f"  ICAT: {test_data['icat']:.6f}")

            scores['ss'].append(test_data['ss'])
            scores['lms'].append(test_data['lms'])
            scores['icat'].append(test_data['icat'])
            weights.append(test_count)

    # Calculate weighted averages
    total_count = sum(weights)

    weighted_avg_ss = sum(s * w for s, w in zip(scores['ss'], weights)) / total_count
    weighted_avg_lms = sum(s * w for s, w in zip(scores['lms'], weights)) / total_count
    weighted_avg_icat = sum(s * w for s, w in zip(scores['icat'], weights)) / total_count

    # Print results
    print("\n" + "=" * 70)
    print("WEIGHTED AVERAGES (Test Set):")
    print("=" * 70)
    print(f"Total Test Count: {total_count}")
    print(f"\nWeighted Average SS:   {weighted_avg_ss:.6f}")
    print(f"Weighted Average LMS:  {weighted_avg_lms:.6f}")
    print(f"Weighted Average ICAT: {weighted_avg_icat:.6f}")
    print("=" * 70)

    # Print weights for verification
    print("\nWeights by domain:")
    for domain, weight in zip(domains, weights):
        percentage = (weight / total_count) * 100
        print(f"  {domain.capitalize():12s}: {weight:4d} ({percentage:5.2f}%)")

    return {
        'weighted_avg_ss': weighted_avg_ss,
        'weighted_avg_lms': weighted_avg_lms,
        'weighted_avg_icat': weighted_avg_icat,
        'total_count': total_count
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate weighted averages of SS, LMS, and ICAT scores from log file')
    parser.add_argument('log_file', type=str, help='Path to the log file to analyze')

    args = parser.parse_args()
    results = calculate_weighted_averages(args.log_file)
