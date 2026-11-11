#!/usr/bin/env python3
"""Recompute ACE-MG metrics from existing per-query predictions.

This script evaluates saved predictions. It does not fit or run the model.
Only Python's standard library is needed.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
from collections import defaultdict
from itertools import groupby, permutations
from pathlib import Path
from typing import Iterable, Mapping, Optional


REQUIRED_COLUMNS = {
    "method", "fold", "role", "novel_class", "unknown_class", "k", "seed",
    "sample_id", "true_label", "closed_prediction", "open_prediction", "knownness",
}
METRICS = (
    "base_macro_f1", "novel_f1_known_only", "unknown_auroc", "unknown_aupr", "oscr_auc",
)
EPISODE_KEY = ("method", "fold", "role", "novel_class", "unknown_class", "k", "seed")
PUBLIC_METHOD = "ACE-MG"


def metric_definition() -> dict[str, object]:
    """Document this helper's calculations without asserting manuscript agreement."""
    return {
        "schema_version": 1,
        "implementation": "ACE-MG supporting-code metric helper",
        "value_scale": "proportion_in_0_1",
        "f1_prediction": {"k_equals_0": "closed_prediction", "k_greater_than_0": "open_prediction"},
        "decision_rule_validation": {
            "enforced_for": "strict_ACE-MG_only",
            "knownness_definition": "highest_final_enrolled_logit_minus_final_Reject_logit",
            "k_greater_than_0_positive": "open_prediction_equals_closed_prediction",
            "k_greater_than_0_negative": "open_prediction_equals_Unknown_label",
            "k_greater_than_0_zero": "open_prediction_is_closed_prediction_or_Unknown_label",
            "k_equals_0": "no_open_prediction_constraint_because_hard_rejection_is_disabled",
            "partial_mode": "violations_reported_but_not_rejected",
        },
        "metrics": {
            "base_macro_f1": {
                "query_subset": "true_label_in_base_classes_only",
                "averaging": "equal_mean_of_F1_for_each_of_the_three_base_classes",
                "excluded_false_positives": "Novel_or_Unknown_queries_predicted_as_Base",
                "zero_denominator_f1": 0.0,
            },
            "novel_f1_known_only": {
                "query_subset": "true_label_in_Base_or_Novel_only",
                "positive_class": "Novel",
                "excluded_false_positives": "Unknown_queries_predicted_as_Novel",
                "zero_denominator_f1": 0.0,
            },
            "unknown_auroc": {
                "query_subset": "all_queries",
                "positive_class": "Unknown",
                "score": "negative_knownness",
                "ties": "half_credit",
            },
            "unknown_aupr": {
                "query_subset": "all_queries",
                "positive_class": "Unknown",
                "score": "negative_knownness",
                "integration": "average_precision_sum_of_recall_increments_times_precision",
                "thresholds": "distinct_scores_with_ties_grouped",
            },
            "oscr_auc": {
                "known_queries": "Base_and_Novel",
                "known_correctness": "closed_prediction_equals_true_label",
                "acceptance_score": "knownness",
                "x_axis": "accepted_Unknown_queries_divided_by_Unknown_query_count",
                "y_axis": "accepted_correct_known_queries_divided_by_known_query_count",
                "integration": "trapezoidal_area_at_distinct_score_thresholds_with_ties_grouped",
                "fixed_open_prediction_used": False,
            },
        },
        "aggregation": {
            "role_identity": ["novel_class", "unknown_class"],
            "within_role": "equal_mean_of_available_support_seed_episode_metrics",
            "within_fold": "equal_mean_of_available_actual_class_role_pair_means",
            "across_folds": "equal_mean_of_available_fold_means",
            "sd": "sample_standard_deviation_of_fold_means",
            "sd_ddof": 1,
            "single_fold_sd": 0.0,
            "partial_mode": "missing_roles_folds_or_seeds_are_not_imputed",
        },
        "verification_scope": "implementation_metadata_not_a_check_against_manuscript_tables",
    }


def binary_f1(y_true: Iterable[bool], y_pred: Iterable[bool]) -> float:
    pairs = list(zip(y_true, y_pred))
    tp = sum(actual and predicted for actual, predicted in pairs)
    fp = sum(not actual and predicted for actual, predicted in pairs)
    fn = sum(actual and not predicted for actual, predicted in pairs)
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2.0 * tp / denominator


def macro_f1(truth: list[str], predictions: list[str], labels: list[str]) -> float:
    return statistics.mean(
        binary_f1((value == label for value in truth),
                  (value == label for value in predictions))
        for label in labels
    )


def roc_auc(unknown_scores: list[float], is_unknown: list[bool]) -> float:
    """Unknown-positive ROC AUC, with half credit for tied scores."""
    pairs = sorted(zip(unknown_scores, is_unknown), key=lambda item: item[0])
    positive_count = sum(is_unknown)
    negative_count = len(is_unknown) - positive_count
    if not positive_count or not negative_count:
        raise ValueError("AUROC requires both known and Unknown queries")
    negatives_below = 0
    wins = 0.0
    for _, tied in groupby(pairs, key=lambda item: item[0]):
        group = list(tied)
        positives = sum(flag for _, flag in group)
        negatives = len(group) - positives
        wins += positives * (negatives_below + 0.5 * negatives)
        negatives_below += negatives
    return wins / (positive_count * negative_count)


def average_precision(unknown_scores: list[float], is_unknown: list[bool]) -> float:
    """Unknown-positive average precision at each distinct score threshold."""
    pairs = sorted(zip(unknown_scores, is_unknown), key=lambda item: item[0], reverse=True)
    positive_count = sum(is_unknown)
    if not positive_count:
        raise ValueError("Average precision requires Unknown queries")
    selected = 0
    selected_positive = 0
    area = 0.0
    for _, tied in groupby(pairs, key=lambda item: item[0]):
        group = list(tied)
        new_positive = sum(flag for _, flag in group)
        selected += len(group)
        selected_positive += new_positive
        area += (new_positive / positive_count) * (selected_positive / selected)
    return area


def oscr_auc(known_scores: list[float], known_correct: list[bool],
             unknown_scores: list[float]) -> float:
    """Area under correct-classification rate versus Unknown false-positive rate."""
    if not known_scores or not unknown_scores:
        raise ValueError("OSCR requires known and Unknown queries")
    items = [(score, False, correct) for score, correct in zip(known_scores, known_correct)]
    items.extend((score, True, False) for score in unknown_scores)
    items.sort(key=lambda item: item[0], reverse=True)
    accepted_unknown = 0
    accepted_correct = 0
    previous_fpr = 0.0
    previous_ccr = 0.0
    area = 0.0
    for _, tied in groupby(items, key=lambda item: item[0]):
        group = list(tied)
        accepted_unknown += sum(is_unknown for _, is_unknown, _ in group)
        accepted_correct += sum(correct for _, _, correct in group)
        fpr = accepted_unknown / len(unknown_scores)
        ccr = accepted_correct / len(known_scores)
        area += (fpr - previous_fpr) * (ccr + previous_ccr) / 2.0
        previous_fpr, previous_ccr = fpr, ccr
    return area


def evaluate_episode(rows: list[Mapping[str, str]], labels: list[str]) -> dict[str, object]:
    if not rows:
        raise ValueError("Empty episode")
    for row_number, row in enumerate(rows, start=1):
        invalid = [field for field in REQUIRED_COLUMNS
                   if not isinstance(row.get(field), str) or not row[field].strip()]
        if invalid:
            raise ValueError(f"Episode row {row_number} has missing or empty fields: {sorted(invalid)}")
    first = rows[0]
    key = tuple(first[field] for field in EPISODE_KEY)
    if any(tuple(row[field] for field in EPISODE_KEY) != key for row in rows):
        raise ValueError("Mixed episode keys")
    novel, unknown = first["novel_class"], first["unknown_class"]
    if novel == unknown or novel not in labels or unknown not in labels:
        raise ValueError(f"Invalid Novel/Unknown role: {novel}, {unknown}")
    base = [label for label in labels if label not in {novel, unknown}]
    enrolled = set(base + [novel])
    k = int(first["k"])
    if k < 0:
        raise ValueError("K must be nonnegative")

    sample_ids: set[str] = set()
    truth: list[str] = []
    closed: list[str] = []
    final: list[str] = []
    knownness: list[float] = []
    decision_rule_violations = 0
    for row in rows:
        sample_id = row["sample_id"]
        if not sample_id or sample_id in sample_ids:
            raise ValueError(f"Missing or duplicate sample ID in {key}: {sample_id}")
        sample_ids.add(sample_id)
        actual = row["true_label"]
        closed_prediction = row["closed_prediction"]
        open_prediction = row["open_prediction"]
        if actual not in labels or closed_prediction not in enrolled:
            raise ValueError(f"Invalid label or closed prediction in {key}")
        if open_prediction not in enrolled | {unknown}:
            raise ValueError(f"Invalid open prediction in {key}")
        score = float(row["knownness"])
        if not math.isfinite(score):
            raise ValueError(f"Non-finite knownness in {key}")
        if k > 0 and (
                (score > 0.0 and open_prediction != closed_prediction)
                or (score < 0.0 and open_prediction != unknown)
                or (score == 0.0 and open_prediction not in {closed_prediction, unknown})):
            decision_rule_violations += 1
        truth.append(actual)
        closed.append(closed_prediction)
        final.append(closed_prediction if k == 0 else open_prediction)
        knownness.append(score)

    base_indices = [index for index, value in enumerate(truth) if value in base]
    known_indices = [index for index, value in enumerate(truth) if value in enrolled]
    unknown_indices = [index for index, value in enumerate(truth) if value == unknown]
    if not base_indices or not known_indices or not unknown_indices:
        raise ValueError(f"Episode lacks Base, known, or Unknown queries: {key}")
    y_base = [truth[index] for index in base_indices]
    p_base = [final[index] for index in base_indices]
    y_known = [truth[index] for index in known_indices]
    p_known = [final[index] for index in known_indices]
    unknown_scores = [-score for score in knownness]
    unknown_targets = [value == unknown for value in truth]

    result: dict[str, object] = {
        "method": first["method"], "fold": int(first["fold"]), "role": first["role"],
        "novel_class": novel, "unknown_class": unknown, "base_classes": "+".join(base),
        "k": k, "seed": int(first["seed"]), "query_count": len(rows),
        "class_query_counts": {label: truth.count(label) for label in labels},
        "decision_rule_violations": decision_rule_violations,
        "base_macro_f1": macro_f1(y_base, p_base, base),
        "novel_f1_known_only": binary_f1(
            (value == novel for value in y_known), (value == novel for value in p_known)),
        "unknown_auroc": roc_auc(unknown_scores, unknown_targets),
        "unknown_aupr": average_precision(unknown_scores, unknown_targets),
        "oscr_auc": oscr_auc(
            [knownness[index] for index in known_indices],
            [closed[index] == truth[index] for index in known_indices],
            [knownness[index] for index in unknown_indices],
        ),
    }
    return result


def read_episodes(paths: list[Path]) -> Iterable[list[dict[str, str]]]:
    completed: set[tuple[str, ...]] = set()
    for path in paths:
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            try:
                fieldnames = reader.fieldnames or []
            except csv.Error as exc:
                raise ValueError(f"{path.name}: malformed CSV header: {exc}") from exc
            if any(not field or not field.strip() for field in fieldnames):
                raise ValueError(f"{path.name}: empty CSV column name")
            if len(fieldnames) != len(set(fieldnames)):
                raise ValueError(f"{path.name}: duplicate CSV column names")
            missing = REQUIRED_COLUMNS - set(fieldnames)
            if missing:
                raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
            current_key: Optional[tuple[str, ...]] = None
            current_rows: list[dict[str, str]] = []
            while True:
                try:
                    row = next(reader)
                except StopIteration:
                    break
                except csv.Error as exc:
                    raise ValueError(f"{path.name}: malformed CSV near line {reader.line_num}: {exc}") from exc
                location = f"{path.name}: line {reader.line_num}"
                if None in row or any(row[field] is None for field in fieldnames):
                    raise ValueError(f"{location}: row field count differs from CSV header")
                empty = [field for field in REQUIRED_COLUMNS if not row[field].strip()]
                if empty:
                    raise ValueError(f"{location}: empty required fields: {sorted(empty)}")
                for field in ("fold", "k", "seed"):
                    try:
                        int(row[field])
                    except ValueError as exc:
                        raise ValueError(f"{location}: {field} must be an integer") from exc
                try:
                    score = float(row["knownness"])
                except ValueError as exc:
                    raise ValueError(f"{location}: knownness must be a finite number") from exc
                if not math.isfinite(score):
                    raise ValueError(f"{location}: knownness must be a finite number")
                key = tuple(row[field] for field in EPISODE_KEY)
                if current_key is not None and key != current_key:
                    completed.add(current_key)
                    yield current_rows
                    current_rows = []
                if key in completed:
                    raise ValueError(f"Episode rows are not contiguous: {key}")
                current_key = key
                current_rows.append(row)
            if current_rows:
                completed.add(current_key)
                yield current_rows


def summarize(episodes: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    role_groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for episode in episodes:
        role_groups[(episode["method"], episode["fold"], episode["k"],
                     episode["novel_class"], episode["unknown_class"])].append(episode)
    fold_groups: dict[tuple[object, ...], list[dict[str, float]]] = defaultdict(list)
    for (method, fold, k, _novel, _unknown), group in role_groups.items():
        fold_groups[(method, fold, k)].append(
            {metric: statistics.mean(float(row[metric]) for row in group) for metric in METRICS})
    fold_rows: list[dict[str, object]] = []
    summary_groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for (method, fold, k), group in sorted(fold_groups.items()):
        row: dict[str, object] = {"method": method, "fold": fold, "k": k, "n_roles": len(group)}
        row.update({metric: statistics.mean(values[metric] for values in group) for metric in METRICS})
        fold_rows.append(row)
        summary_groups[(method, k)].append(row)
    summary_rows: list[dict[str, object]] = []
    for (method, k), group in sorted(summary_groups.items()):
        row = {"method": method, "k": k, "n_folds": len(group)}
        for metric in METRICS:
            values = [float(item[metric]) for item in group]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(row)
    return fold_rows, summary_rows


def validate_strict_protocol(episodes: list[dict[str, object]], protocol: Mapping[str, object]) -> None:
    fixed_settings = {
        "folds": {1, 2, 3, 4, 5},
        "k_values": {0, 1, 5, 10},
        "support_seeds": set(range(42, 62)),
    }
    for field, expected in fixed_settings.items():
        values = protocol[field]
        if (not isinstance(values, list) or any(type(value) is not int for value in values)
                or len(values) != len(expected) or set(values) != expected):
            raise ValueError(f"Strict ACE-MG protocol requires fixed {field}: {sorted(expected)}")
    folds = set(protocol["folds"])
    k_values = set(protocol["k_values"])
    expected_seeds = set(protocol["support_seeds"])
    labels = list(protocol["all_labels"])
    persistent_base = str(protocol["persistent_base_class"])
    if persistent_base != "normal" or persistent_base not in labels:
        raise ValueError("Strict ACE-MG protocol requires normal as the persistent Base class")
    event_labels = [label for label in labels if label != persistent_base]
    if len(labels) != 5 or len(set(labels)) != 5:
        raise ValueError("Strict protocol must list five distinct class labels")
    expected_pairs = set(permutations(event_labels, 2))
    if int(protocol["roles_per_fold"]) != len(expected_pairs):
        raise ValueError("roles_per_fold differs from the number of ordered Novel/Unknown pairs")
    if protocol["method"] != PUBLIC_METHOD:
        raise ValueError(f"Strict protocol method must be {PUBLIC_METHOD}")
    by_cell: dict[tuple[int, int], dict[tuple[str, str], set[int]]] = defaultdict(lambda: defaultdict(set))
    for row in episodes:
        if row["method"] != PUBLIC_METHOD:
            raise ValueError(f"Unexpected method: {row['method']}")
        fold, k, seed = int(row["fold"]), int(row["k"]), int(row["seed"])
        if fold not in folds or k not in k_values:
            raise ValueError(f"Unexpected fold/K: {fold}/{k}")
        pair = (str(row["novel_class"]), str(row["unknown_class"]))
        if pair not in expected_pairs:
            raise ValueError(f"Invalid strict Novel/Unknown class pair: {pair}")
        expected_role = f"novel={pair[0]}|unknown={pair[1]}"
        if row["role"] != expected_role:
            raise ValueError(f"Role string does not match actual Novel/Unknown pair: {row['role']}; expected {expected_role}")
        query_counts = row.get("class_query_counts")
        if (not isinstance(query_counts, dict) or set(query_counts) != set(labels)
                or any(not isinstance(query_counts[label], int) or query_counts[label] < 1 for label in labels)):
            raise ValueError(f"Strict episode must contain queries from every class: fold={fold}, K={k}, {expected_role}, seed={seed}")
        if row.get("decision_rule_violations") != 0:
            raise ValueError(f"Strict ACE-MG decision rule conflicts with knownness: fold={fold}, K={k}, {expected_role}, seed={seed}; violations={row.get('decision_rule_violations')}")
        role_seeds = by_cell[(fold, k)][pair]
        if seed in role_seeds:
            raise ValueError(f"Duplicate episode: fold={fold}, K={k}, role={row['role']}, seed={seed}")
        role_seeds.add(seed)
    for fold in folds:
        for k in k_values:
            role_map = by_cell.get((fold, k), {})
            if set(role_map) != expected_pairs:
                missing = sorted(expected_pairs - set(role_map))
                extra = sorted(set(role_map) - expected_pairs)
                raise ValueError(f"Fold {fold}, K={k}: class-role pairs differ from protocol; missing={missing}, extra={extra}")
            for pair, seeds in role_map.items():
                target = {-1} if k == 0 else expected_seeds
                if seeds != target:
                    raise ValueError(f"Fold {fold}, K={k}, pair={pair}: support seeds differ from protocol")


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {field: json.dumps(value, ensure_ascii=False, sort_keys=True)
             if isinstance(value, dict) else value for field, value in row.items()}
            for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", nargs="+", type=Path, help="Per-query CSV or CSV.GZ files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).resolve().parents[1] / "configs/protocol.json")
    parser.add_argument("--strict-protocol", action="store_true", help="Require all five folds, 12 roles and saved support seeds")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    labels = list(protocol["all_labels"])
    if len(labels) != 5 or len(set(labels)) != 5:
        raise ValueError("Protocol must list five distinct class labels")
    if any(not path.is_file() for path in args.predictions):
        raise FileNotFoundError("A predictions file is missing")
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output}")

    episodes = [evaluate_episode(group, labels) for group in read_episodes(args.predictions)]
    if not episodes:
        raise ValueError("No prediction rows")
    if args.strict_protocol:
        validate_strict_protocol(episodes, protocol)
    fold_rows, summary_rows = summarize(episodes)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "episode_metrics.csv", episodes,
              ["method", "fold", "role", "novel_class", "unknown_class", "base_classes", "k", "seed", "query_count", "class_query_counts", "decision_rule_violations", *METRICS])
    write_csv(output / "fold_means.csv", fold_rows, ["method", "fold", "k", "n_roles", *METRICS])
    summary_columns = ["method", "k", "n_folds"]
    for metric in METRICS:
        summary_columns.extend([f"{metric}_mean", f"{metric}_sd"])
    write_csv(output / "summary_mean_sd.csv", summary_rows, summary_columns)
    definition = metric_definition()
    definition["run"] = {
        "strict_protocol": args.strict_protocol,
        "observed_methods": sorted({str(row["method"]) for row in episodes}),
        "protocol_method": protocol["method"],
        "labels": labels,
        "persistent_base_class": protocol["persistent_base_class"],
    }
    (output / "metric_definition.json").write_text(
        json.dumps(definition, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": len(episodes), "fold_cells": len(fold_rows),
                      "strict_protocol": args.strict_protocol,
                      "metric_definition": str(output / "metric_definition.json"),
                      "output_dir": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
