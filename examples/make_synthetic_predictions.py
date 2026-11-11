#!/usr/bin/env python3
"""Generate an entirely synthetic five-fold CSV for the evaluation CLI."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import permutations
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    protocol = json.loads((Path(__file__).resolve().parents[1] / "configs/protocol.json").read_text(encoding="utf-8"))
    labels = list(protocol["all_labels"])
    events = [label for label in labels if label != protocol["persistent_base_class"]]
    roles = list(permutations(events, 2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["method", "fold", "role", "novel_class", "unknown_class", "k", "seed",
               "sample_id", "true_label", "closed_prediction", "open_prediction", "knownness"]
    row_count = 0
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for fold in protocol["folds"]:
            for role_index, (novel, unknown) in enumerate(roles):
                base = [label for label in labels if label not in {novel, unknown}]
                for k in protocol["k_values"]:
                    seeds = [-1] if k == 0 else protocol["support_seeds"]
                    for seed in seeds:
                        for class_index, actual in enumerate(labels):
                            closed = base[0] if actual == unknown else actual
                            # Deliberate, deterministic mistakes make fold summaries nontrivial.
                            if actual != unknown and (fold + role_index + k + seed + class_index) % 17 == 0:
                                closed = base[0] if actual != base[0] else base[1]
                            score = -0.8 if actual == unknown else (0.2 if closed != actual else 1.8)
                            final = closed
                            if k > 0 and (actual == unknown or (closed != actual and class_index % 2 == 0)):
                                final = unknown
                                # A Reject decision requires its logit to exceed
                                # the strongest enrolled-class logit.
                                if actual != unknown:
                                    score = -0.2
                            writer.writerow({"method": protocol["method"], "fold": fold,
                                             "role": f"novel={novel}|unknown={unknown}",
                                             "novel_class": novel, "unknown_class": unknown,
                                             "k": k, "seed": seed,
                                             "sample_id": f"synthetic_f{fold}_r{role_index}_{actual}",
                                             "true_label": actual, "closed_prediction": closed,
                                             "open_prediction": final, "knownness": score})
                            row_count += 1
    print(f"Wrote {row_count} synthetic query rows to {args.output}")


if __name__ == "__main__":
    main()
