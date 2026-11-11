"""Checks with hand-computable cases and an in-memory CLAP-shaped NPZ."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.recompute_metrics import evaluate_episode, summarize
from inputs.validate_audio_embeddings import validate


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((PACKAGE_ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
LABELS = list(PROTOCOL["all_labels"])


def episode_rows(k: int, tied: bool = False) -> list[dict[str, str]]:
    rows = []
    novel, unknown = "estrus", "squeal"
    for index, actual in enumerate(LABELS):
        closed = "cough" if actual == unknown else actual
        final = "squeal" if k > 0 and actual in {"eat", "squeal"} else closed
        score = -1.0 if actual == unknown else (-0.1 if actual == "eat" else 4 - index)
        rows.append({"method": "ACE-MG", "fold": "1",
                     "role": "novel=estrus|unknown=squeal", "novel_class": novel,
                     "unknown_class": unknown, "k": str(k), "seed": "-1" if k == 0 else "42",
                     "sample_id": f"synthetic-{index}", "true_label": actual,
                     "closed_prediction": closed, "open_prediction": final,
                     "knownness": "0" if tied else str(score)})
    return rows


class EvaluationTests(unittest.TestCase):
    def test_k0_closed_f1_and_k1_reject_f1_but_same_oscr(self) -> None:
        k0 = evaluate_episode(episode_rows(0), LABELS)
        k1 = evaluate_episode(episode_rows(1), LABELS)
        self.assertEqual(k0["base_macro_f1"], 1.0)
        self.assertAlmostEqual(k1["base_macro_f1"], 2 / 3)
        self.assertEqual(k0["novel_f1_known_only"], 1.0)
        self.assertEqual(k1["novel_f1_known_only"], 1.0)
        for key in ("unknown_auroc", "unknown_aupr", "oscr_auc"):
            self.assertEqual(k0[key], 1.0)
            self.assertEqual(k1[key], 1.0)

    def test_tied_knownness_matches_rank_metric_conventions(self) -> None:
        result = evaluate_episode(episode_rows(0, tied=True), LABELS)
        self.assertEqual(result["unknown_auroc"], 0.5)
        self.assertEqual(result["unknown_aupr"], 0.2)
        self.assertEqual(result["oscr_auc"], 0.5)

    def test_fold_summary_uses_role_means_then_sample_sd(self) -> None:
        rows = [
            {"method": "ACE-MG", "fold": 1, "k": 1, "role": "A", "base_macro_f1": 0.0},
            {"method": "ACE-MG", "fold": 1, "k": 1, "role": "A", "base_macro_f1": 1.0},
            {"method": "ACE-MG", "fold": 1, "k": 1, "role": "B", "base_macro_f1": 1.0},
            {"method": "ACE-MG", "fold": 2, "k": 1, "role": "A", "base_macro_f1": 0.0},
            {"method": "ACE-MG", "fold": 2, "k": 1, "role": "B", "base_macro_f1": 0.0},
        ]
        for row in rows:
            novel, unknown = ("estrus", "squeal") if row["role"] == "A" else ("eat", "cough")
            row.update({"novel_class": novel, "unknown_class": unknown})
            row.update({key: row["base_macro_f1"] for key in
                        ("novel_f1_known_only", "unknown_auroc", "unknown_aupr", "oscr_auc")})
        folds, summary = summarize(rows)
        self.assertEqual([row["base_macro_f1"] for row in folds], [0.75, 0.0])
        self.assertEqual(summary[0]["base_macro_f1_mean"], 0.375)
        self.assertAlmostEqual(summary[0]["base_macro_f1_sd"], math.sqrt(0.28125))


class EmbeddingInputTests(unittest.TestCase):
    def test_valid_npz_and_absolute_path_rejection(self) -> None:
        matrix = np.zeros((2, 512), dtype=np.float32)
        matrix[0, 0] = 1.0
        matrix[1, 1] = 1.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio_embeddings.npz"
            fields = {"embeddings": matrix, "labels": np.array(["cough", "eat"]),
                      "sample_ids": np.array(["synthetic-1", "synthetic-2"]),
                      "paths": np.array(["clip1.wav", "clip2.wav"]),
                      "model_id": np.array("laion/clap-htsat-unfused"),
                      "audio_padding": np.array("pad"),
                      "analysis_window_seconds": np.array(1.0)}
            np.savez_compressed(path, **fields)
            report, _, _ = validate(path, PROTOCOL)
            self.assertEqual(report["rows"], 2)
            fields["paths"] = np.array(["C:\\private\\clip1.wav", "clip2.wav"])
            np.savez_compressed(path, **fields)
            with self.assertRaisesRegex(ValueError, "relative"):
                validate(path, PROTOCOL)


if __name__ == "__main__":
    unittest.main()
