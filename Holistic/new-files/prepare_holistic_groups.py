import argparse
import csv
import json
import os
from typing import Dict, List, Tuple


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_indefinite_article(noun: str) -> str:
    """
    Simple heuristic for 'a' vs 'an'.
    """
    if not noun:
        return "a"
    return "an" if noun[0].lower() in "aeiou" else "a"


def build_groups_from_local_jsons(
    holistic_root: str,
    axis: str,
    min_group_size: int = 2,
    max_groups: int | None = None,
) -> List[List[str]]:
    nouns_path = os.path.join(holistic_root, "../holistic-bias-data/nouns.json")
    standalone_np_path = os.path.join(holistic_root, "../holistic-bias-data/standalone_noun_phrases.json")
    sentence_templates_path = os.path.join(holistic_root, "../holistic-bias-data/sentence_templates.json")

    # Load JSONs
    nouns_json: Dict = load_json(nouns_path)
    standalone_np_json: Dict = load_json(standalone_np_path)
    sentence_templates_json: Dict = load_json(sentence_templates_path)

    # Collect a pool of singular nouns from nouns.json
    noun_singulars: List[str] = []
    for category, pairs in nouns_json.items():
        for pair in pairs:
            if not pair:
                continue
            singular = pair[0]
            noun_singulars.append(singular)

    if axis not in standalone_np_json:
        raise ValueError(
            f"Axis '{axis}' not found in standalone_noun_phrases.json. "
            f"Available axes: {list(standalone_np_json.keys())}"
        )

    base_sentence_template = "I'm {noun_phrase}."

    rows: List[List[str]] = []

    for entry in standalone_np_json[axis]:
        if "noun_phrase" not in entry:
            continue

        noun_phrase_template: str = entry["noun_phrase"]
        descriptor_pattern = noun_phrase_template  

        group_pairs: List[Tuple[str, str]] = []
        for noun in noun_singulars:
            article = choose_indefinite_article(noun)
            noun_phrase = noun_phrase_template.format(article=article, noun=noun)
            sentence = base_sentence_template.format(noun_phrase=noun_phrase)
            group_pairs.append((noun, sentence))

        if len(group_pairs) < min_group_size:
            continue

        # Construct TSV row:
        #   axis, descriptor_pattern, noun_1, sentence_1, noun_2, sentence_2, ...
        out_row: List[str] = [axis, descriptor_pattern]
        for noun, sentence in group_pairs:
            out_row.append(noun)
            out_row.append(sentence)

        rows.append(out_row)

        if max_groups is not None and len(rows) >= max_groups:
            break

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--holistic_root",
        type=str,
        required=True,
        help=(
            "Path to the holistic_bias folder inside the ResponsibleNLP repo, "
            "containing nouns.json, standalone_noun_phrases.json, sentence_templates.json, etc."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output TSV file, e.g. data/holistic_ability_groups.tsv",
    )
    parser.add_argument(
        "--axis",
        type=str,
        default="ability",
        help="Axis to build groups for (e.g., ability, race_ethnicity, religion...).",
    )
    parser.add_argument(
        "--min_group_size",
        type=int,
        default=2,
        help="Minimum number of nouns (classes) per group.",
    )
    parser.add_argument(
        "--max_groups",
        type=int,
        default=None,
        help="Optional max number of groups to generate.",
    )

    args = parser.parse_args()

    rows = build_groups_from_local_jsons(
        holistic_root=args.holistic_root,
        axis=args.axis,
        min_group_size=args.min_group_size,
        max_groups=args.max_groups,
    )

    if not rows:
        raise RuntimeError(
            f"No groups built for axis={args.axis}. "
            f"Check that the axis exists in standalone_noun_phrases.json."
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} groups to {args.output}")


if __name__ == "__main__":
    main()
