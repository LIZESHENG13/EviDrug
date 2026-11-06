"""Recompute metrics from local prediction arrays and an authorized cohort."""

import argparse
import json

import numpy as np

from evidrug.data.records import load_split
from evidrug.metrics import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--resources", default="resources")
    args = parser.parse_args()
    content = load_split(args.data, resources=args.resources)
    with np.load(args.predictions) as saved:
        print(json.dumps(metrics(saved["labels"], saved["probabilities"], content[4]), indent=2))


if __name__ == "__main__":
    main()
