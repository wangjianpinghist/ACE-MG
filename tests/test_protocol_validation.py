"""Regression checks for semantic protocol validation and metric provenance."""

from __future__ import annotations

import copy
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from itertools import permutations
from pathlib import Path

from evaluation.recompute_metrics import (
    METRICS,
    evaluate_episode,
    read_episodes,
    summarize,
    validate_strict_protocol,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((PACKAGE_ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
LABELS = list(PROTOCOL["all_labels"])


def query_rows(fold: int, novel: str, unknown: str, k: int, seed: int) -> list[dict[str, str]]:
    base = [label for label in LABELS if label not in {novel, unknown}]
    rows = []
    for index, actual in enumerate(LABELS):
        closed = base[0] if actual == unknown else actual
        final = unknown if k > 0 and actual == unknown else closed
        rows.append({
            "method": "ACE-MG", "fold": str(fold),
            "role": f"novel={novel}|unknown={unknown}", "novel_class": novel,
            "unknown_class": unknown, "k": str(k), "seed": str(seed),
            "sample_id": f"synthetic-{index}", "true_label": actual,
            "closed_prediction": closed, "open_prediction": final,
            "knownness": "-1" if actual == unknown else "1",
        })
    return rows


def complete_fixture() -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    event_labels = [label for label in LABELS if label != PROTOCOL["persistent_base_class"]]
    episodes, all_rows = [], []
    for fold in PROTOCOL["folds"]:
        for novel, unknown in permutations(event_labels, 2):
            for k in PROTOCOL["k_values"]:
                for seed in ([-1] if k == 0 else PROTOCOL["support_seeds"]):
                    rows = query_rows(fold, novel, unknown, k, seed)
                    episodes.append(evaluate_episode(rows, LABELS))
                    all_rows.extend(rows)
    return episodes, all_rows


class StrictProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.episodes, cls.rows = complete_fixture()

    def test_complete_five_fold_protocol_passes(self) -> None:
        self.assertEqual(len(self.episodes), 3660)
        validate_strict_protocol(self.episodes, PROTOCOL)

    def test_distinct_role_strings_cannot_hide_repeated_actual_pair(self) -> None:
        episodes = copy.deepcopy(self.episodes)
        for episode in episodes:
            episode["novel_class"], episode["unknown_class"] = "estrus", "squeal"
        with self.assertRaisesRegex(ValueError, "Role string|Duplicate episode"):
            validate_strict_protocol(episodes, PROTOCOL)

    def test_normal_cannot_be_novel_or_unknown(self) -> None:
        for field in ("novel_class", "unknown_class"):
            with self.subTest(field=field):
                episodes = copy.deepcopy(self.episodes)
                episodes[0][field] = "normal"
                episodes[0]["role"] = (
                    f"novel={episodes[0]['novel_class']}|unknown={episodes[0]['unknown_class']}")
                with self.assertRaisesRegex(ValueError, "Invalid strict Novel/Unknown"):
                    validate_strict_protocol(episodes, PROTOCOL)

    def test_role_cannot_change_actual_pair_across_support_seeds(self) -> None:
        episodes = copy.deepcopy(self.episodes)
        episode = next(row for row in episodes if row["k"] == 1 and row["seed"] == 43)
        episode["novel_class"], episode["unknown_class"] = "estrus", "squeal"
        with self.assertRaisesRegex(ValueError, "Role string"):
            validate_strict_protocol(episodes, PROTOCOL)

    def test_missing_actual_role_is_rejected(self) -> None:
        first = self.episodes[0]
        episodes = [row for row in self.episodes
                    if not (row["fold"] == first["fold"] and row["k"] == first["k"]
                            and row["novel_class"] == first["novel_class"]
                            and row["unknown_class"] == first["unknown_class"])]
        with self.assertRaisesRegex(ValueError, "class-role pairs differ"):
            validate_strict_protocol(episodes, PROTOCOL)

    def test_duplicate_episode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate episode"):
            validate_strict_protocol(self.episodes + [self.episodes[0]], PROTOCOL)

    def test_missing_support_draw_is_rejected(self) -> None:
        index = next(i for i, row in enumerate(self.episodes) if row["k"] == 1)
        episodes = self.episodes[:index] + self.episodes[index + 1:]
        with self.assertRaisesRegex(ValueError, "support seeds differ"):
            validate_strict_protocol(episodes, PROTOCOL)

    def test_strict_episode_requires_each_of_five_query_classes(self) -> None:
        first = self.episodes[0]
        for missing in LABELS:
            with self.subTest(missing=missing):
                rows = query_rows(first["fold"], first["novel_class"], first["unknown_class"],
                                  first["k"], first["seed"])
                rows = [row for row in rows if row["true_label"] != missing]
                try:
                    incomplete = evaluate_episode(rows, LABELS)
                except ValueError:
                    # Missing the sole Unknown already fails the metric precondition.
                    continue
                episodes = [incomplete] + self.episodes[1:]
                with self.assertRaisesRegex(ValueError, "queries from every class"):
                    validate_strict_protocol(episodes, PROTOCOL)

    def test_unexpected_method_is_not_silently_renamed(self) -> None:
        rows = query_rows(1, "estrus", "squeal", 0, -1)
        for row in rows:
            row["method"] = "OtherMethod"
        episode = evaluate_episode(rows, LABELS)
        self.assertEqual(episode["method"], "OtherMethod")
        with self.assertRaisesRegex(ValueError, "Unexpected method"):
            validate_strict_protocol([episode] + self.episodes[1:], PROTOCOL)

    def test_modified_protocol_cannot_redefine_paper_folds_k_or_seeds(self) -> None:
        variants = [
            ("folds", [1, 2, 3, 4]),
            ("folds", [1, 2, 3, 4, 6]),
            ("folds", [1, 2, 3, 4, 5, 5]),
            ("folds", [True, 2, 3, 4, 5]),
            ("k_values", [0, 1, 5]),
            ("k_values", [0, 1, 5, 20]),
            ("support_seeds", list(range(42, 61))),
            ("support_seeds", list(range(43, 63))),
        ]
        for field, values in variants:
            with self.subTest(field=field, values=values):
                protocol = copy.deepcopy(PROTOCOL)
                protocol[field] = values
                with self.assertRaisesRegex(ValueError, f"requires fixed {field}"):
                    validate_strict_protocol(self.episodes, protocol)

    def replace_fixture_episode(self, rows: list[dict[str, str]]) -> list[dict[str, object]]:
        replacement = evaluate_episode(rows, LABELS)
        match_fields = ("fold", "k", "seed", "novel_class", "unknown_class")
        index = next(index for index, episode in enumerate(self.episodes)
                     if all(episode[field] == replacement[field] for field in match_fields))
        episodes = list(self.episodes)
        episodes[index] = replacement
        return episodes

    def test_strict_rejects_open_decisions_inconsistent_with_knownness(self) -> None:
        # The first query has closed_prediction=cough; eat is another enrolled class.
        variants = [("1", "squeal"), ("-1", "cough"), ("1", "eat"), ("0", "eat")]
        for score, prediction in variants:
            with self.subTest(knownness=score, open_prediction=prediction):
                rows = query_rows(1, "estrus", "squeal", 1, 42)
                rows[0]["knownness"], rows[0]["open_prediction"] = score, prediction
                episodes = self.replace_fixture_episode(rows)
                self.assertEqual(evaluate_episode(rows, LABELS)["decision_rule_violations"], 1)
                with self.assertRaisesRegex(ValueError, "decision rule conflicts with knownness"):
                    validate_strict_protocol(episodes, PROTOCOL)

    def test_zero_knownness_allows_closed_prediction_or_reject(self) -> None:
        for prediction in ("cough", "squeal"):
            with self.subTest(open_prediction=prediction):
                rows = query_rows(1, "estrus", "squeal", 1, 42)
                rows[0]["knownness"], rows[0]["open_prediction"] = "0", prediction
                episodes = self.replace_fixture_episode(rows)
                self.assertEqual(evaluate_episode(rows, LABELS)["decision_rule_violations"], 0)
                validate_strict_protocol(episodes, PROTOCOL)

    def test_k0_open_prediction_is_not_forced_by_knownness(self) -> None:
        for score, prediction in [("1", "squeal"), ("-1", "cough"), ("0", "eat")]:
            with self.subTest(knownness=score, open_prediction=prediction):
                rows = query_rows(1, "estrus", "squeal", 0, -1)
                rows[0]["knownness"], rows[0]["open_prediction"] = score, prediction
                episodes = self.replace_fixture_episode(rows)
                self.assertEqual(evaluate_episode(rows, LABELS)["decision_rule_violations"], 0)
                validate_strict_protocol(episodes, PROTOCOL)

    def test_complete_cli_writes_explicit_definition_and_class_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions, protocol_path = root / "predictions.csv", root / "protocol.json"
            with predictions.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
            protocol_path.write_text(json.dumps(PROTOCOL), encoding="utf-8")
            run = subprocess.run([
                sys.executable, "-B", str(PACKAGE_ROOT / "evaluation/recompute_metrics.py"),
                str(predictions), "--output-dir", str(root / "output"),
                "--protocol", str(protocol_path), "--strict-protocol",
            ], capture_output=True, text=True, check=False)
            self.assertEqual(run.returncode, 0, run.stderr)
            status = json.loads(run.stdout)
            self.assertEqual(status["episodes"], 3660)
            self.assertTrue(status["strict_protocol"])
            definition = json.loads((root / "output/metric_definition.json").read_text(encoding="utf-8"))
            self.assertEqual(definition["metrics"]["base_macro_f1"]["query_subset"],
                             "true_label_in_base_classes_only")
            self.assertEqual(definition["metrics"]["novel_f1_known_only"]["query_subset"],
                             "true_label_in_Base_or_Novel_only")
            self.assertEqual(definition["aggregation"]["sd_ddof"], 1)
            self.assertEqual(definition["decision_rule_validation"]["enforced_for"], "strict_ACE-MG_only")
            with (root / "output/episode_metrics.csv").open(encoding="utf-8", newline="") as handle:
                first = next(csv.DictReader(handle))
            self.assertEqual(json.loads(first["class_query_counts"]), dict.fromkeys(LABELS, 1))
            self.assertEqual(int(first["decision_rule_violations"]), 0)

    def test_partial_other_method_cli_reports_conflicts_without_rejecting(self) -> None:
        rows = query_rows(1, "estrus", "squeal", 1, 42)
        for row in rows:
            row["method"] = "OtherMethod"
        rows[0]["open_prediction"] = "squeal"  # Positive knownness: diagnostic conflict only.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.csv"
            with predictions.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            run = subprocess.run([
                sys.executable, "-B", str(PACKAGE_ROOT / "evaluation/recompute_metrics.py"),
                str(predictions), "--output-dir", str(root / "output"),
            ], capture_output=True, text=True, check=False)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertFalse(json.loads(run.stdout)["strict_protocol"])
            with (root / "output/episode_metrics.csv").open(encoding="utf-8", newline="") as handle:
                episode = next(csv.DictReader(handle))
            self.assertEqual(episode["method"], "OtherMethod")
            self.assertEqual(int(episode["decision_rule_violations"]), 1)


class PartialAggregationTests(unittest.TestCase):
    def test_shared_role_string_does_not_merge_different_actual_tasks(self) -> None:
        episodes = []
        for novel, unknown, value in [("cough", "eat", 0.0), ("cough", "eat", 1.0),
                                      ("estrus", "squeal", 1.0)]:
            episode = {"method": "ACE-MG", "fold": 1, "k": 1, "role": "same-string",
                       "novel_class": novel, "unknown_class": unknown}
            episode.update(dict.fromkeys(METRICS, value))
            episodes.append(episode)
        folds, _ = summarize(episodes)
        self.assertEqual(folds[0]["n_roles"], 2)
        self.assertEqual(folds[0]["base_macro_f1"], 0.75)

    def test_different_aliases_for_same_pair_are_one_role(self) -> None:
        episodes = []
        for role, value in [("alias-A", 0.0), ("alias-B", 1.0)]:
            episode = {"method": "ACE-MG", "fold": 1, "k": 1, "role": role,
                       "novel_class": "cough", "unknown_class": "eat"}
            episode.update(dict.fromkeys(METRICS, value))
            episodes.append(episode)
        folds, _ = summarize(episodes)
        self.assertEqual(folds[0]["n_roles"], 1)
        self.assertEqual(folds[0]["base_macro_f1"], 0.5)


class PredictionCsvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = query_rows(1, "estrus", "squeal", 0, -1)
        self.columns = list(self.rows[0])

    def assert_invalid_csv(self, header: list[str], row: list[str], expected: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                writer.writerow(row)
            with self.assertRaisesRegex(ValueError, expected):
                list(read_episodes([path]))

    def test_duplicate_header_is_rejected(self) -> None:
        self.assert_invalid_csv(self.columns + ["knownness"], list(self.rows[0].values()) + ["1"],
                                "duplicate CSV column")

    def test_empty_header_is_rejected(self) -> None:
        self.assert_invalid_csv(self.columns + [""], list(self.rows[0].values()) + ["extra"],
                                "empty CSV column")

    def test_missing_required_column_is_rejected(self) -> None:
        self.assert_invalid_csv(self.columns[:-1], list(self.rows[0].values())[:-1], "missing columns")

    def test_short_and_long_rows_are_rejected_with_line_number(self) -> None:
        values = list(self.rows[0].values())
        for row in (values[:-1], values + ["extra"]):
            with self.subTest(length=len(row)):
                self.assert_invalid_csv(self.columns, row, "line 2: row field count")

    def test_whitespace_required_value_is_rejected(self) -> None:
        values = list(self.rows[0].values())
        values[self.columns.index("role")] = "  "
        self.assert_invalid_csv(self.columns, values, "line 2: empty required fields")

    def test_nonfinite_knownness_is_rejected_with_line_number(self) -> None:
        for value in ("NaN", "inf", "-inf", "text"):
            with self.subTest(value=value):
                values = list(self.rows[0].values())
                values[self.columns.index("knownness")] = value
                self.assert_invalid_csv(self.columns, values, "line 2: knownness must be a finite number")

    def test_invalid_integer_has_useful_message(self) -> None:
        values = list(self.rows[0].values())
        values[self.columns.index("seed")] = "42.5"
        self.assert_invalid_csv(self.columns, values, "line 2: seed must be an integer")

    def test_noncontiguous_episode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "interleaved.csv"
            other = query_rows(1, "cough", "eat", 0, -1)
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.columns)
                writer.writeheader()
                writer.writerows(self.rows + other + self.rows)
            with self.assertRaisesRegex(ValueError, "not contiguous"):
                list(read_episodes([path]))


class MetricScopeRegressionTests(unittest.TestCase):
    def test_existing_f1_query_subsets_remain_unchanged(self) -> None:
        rows = query_rows(1, "estrus", "squeal", 0, -1)
        for row in rows:
            if row["true_label"] in {"estrus", "squeal"}:
                row["closed_prediction"] = row["open_prediction"] = "cough"
        self.assertEqual(evaluate_episode(rows, LABELS)["base_macro_f1"], 1.0)
        rows = query_rows(1, "estrus", "squeal", 0, -1)
        for row in rows:
            if row["true_label"] == "squeal":
                row["closed_prediction"] = row["open_prediction"] = "estrus"
        self.assertEqual(evaluate_episode(rows, LABELS)["novel_f1_known_only"], 1.0)


if __name__ == "__main__":
    unittest.main()
