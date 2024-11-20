import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import pandas as pd
from beartype import beartype
from tqdm.auto import tqdm

from src.labels import ground_truth


class setEncoder(json.JSONEncoder):

    def default(self, obj):
        return list(obj)


@beartype
def split_events(events: list[dict], split_idx=None):
    test_events = ground_truth(deepcopy(events))
    if not split_idx:
        split_idx = random.randint(1, len(test_events))
    test_events = test_events[:split_idx]
    labels = test_events[-1]['labels']
    for event in test_events:
        del event['labels']
    return test_events, labels


@beartype
def create_kaggle_testset(sessions: pd.DataFrame, sessions_output: Path, labels_output: Path):
    last_labels = []
    splitted_sessions = []

    for _, session in tqdm(sessions.iterrows(), desc="Creating trimmed testset", total=len(sessions)):
        if len(session['events']) < 2:
            continue
        session = session.to_dict()
        splitted_events, labels = split_events(session['events'])
        last_labels.append({'session': session['session'], 'labels': labels})
        splitted_sessions.append({'session': session['session'], 'events': splitted_events})

    with open(sessions_output, 'w') as f:
        for session in splitted_sessions:
            f.write(json.dumps(session) + '\n')

    with open(labels_output, 'w') as f:
        for label in last_labels:
            f.write(json.dumps(label, cls=setEncoder) + '\n')

@beartype
def main(test_set: Path, output_path: Path, seed: int):
    random.seed(seed)
    test_sessions = pd.read_json(test_set, lines=True)
    test_sessions_file = output_path / 'test_sessions.jsonl'
    test_labels_file = output_path / 'test_labels.jsonl'
    create_kaggle_testset(test_sessions, test_sessions_file, test_labels_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-set', type=Path, required=True)
    parser.add_argument('--output-path', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args.test_set, args.output_path, args.seed)
