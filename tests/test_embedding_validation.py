"""NPZ schema and split-identity regressions without real audio or embeddings."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from inputs.validate_audio_embeddings import validate


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


class EmbeddingValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def fields(self, count: int = 2) -> dict[str, np.ndarray]:
        matrix = np.zeros((count, 512), dtype=np.float32)
        matrix[:, 0] = 1.0
        return {
            "embeddings": matrix,
            "labels": np.array(["cough"] * count),
            "sample_ids": np.array([f"id-{index}" for index in range(count)]),
            "paths": np.array([f"clips/{index}.wav" for index in range(count)]),
            "model_id": np.array(PROTOCOL["expected_audio_model_id"]),
            "audio_padding": np.array(PROTOCOL["expected_audio_padding"]),
            "analysis_window_seconds": np.array(1.0),
        }

    def write(self, fields: dict[str, np.ndarray], name: str = "input.npz") -> Path:
        path = self.root / name
        np.savez_compressed(path, **fields)
        return path

    def cli(self, *paths: Path, disjoint: bool = True) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, "-B", str(ROOT / "inputs/validate_audio_embeddings.py")]
        command.extend(str(path) for path in paths)
        if disjoint:
            command.append("--disjoint")
        return subprocess.run(command, capture_output=True, text=True, check=False)

    def test_normalizes_separator_dot_and_case_without_reading_audio(self) -> None:
        fields = self.fields(1)
        fields["paths"] = np.array(["Clips\\.\\A.wav"])
        report, _, paths = validate(self.write(fields), PROTOCOL)
        self.assertEqual(paths, {"clips/a.wav"})
        self.assertEqual(report["path_identity_policy"], "portable_casefold")
        self.assertFalse(report["session_ids_present"])

    def test_rejects_path_alias_duplicates(self) -> None:
        for alias in ("clips/./a.wav", "clips\\a.wav", "CLIPS/A.wav", "clips//a.wav"):
            with self.subTest(alias=alias):
                fields = self.fields()
                fields["paths"] = np.array(["clips/a.wav", alias])
                with self.assertRaisesRegex(ValueError, "duplicate paths"):
                    validate(self.write(fields), PROTOCOL)

    def test_rejects_paths_outside_portable_root(self) -> None:
        paths = ("../a.wav", "clips/../a.wav", "C:a.wav", "C:/a.wav", "\\a.wav",
                 "\\\\server\\share\\a.wav", "/a.wav", ".", "./", "   ", "a.wav ",
                 "clips./a.wav", "clips/ a.wav", "a.wav:stream", "a\x00.wav")
        for path in paths:
            with self.subTest(path=path):
                fields = self.fields(1)
                fields["paths"] = np.array([path])
                with self.assertRaises(ValueError):
                    validate(self.write(fields), PROTOCOL)

    def test_allows_spaces_inside_file_names(self) -> None:
        fields = self.fields(1)
        fields["paths"] = np.array(["clips/my pig.wav"])
        self.assertEqual(validate(self.write(fields), PROTOCOL)[2], {"clips/my pig.wav"})

    def test_rejects_bad_text_array_shapes_types_and_values(self) -> None:
        for name in ("labels", "sample_ids", "paths", "session_ids"):
            for invalid in (np.array("cough"), np.array([["cough"], ["eat"]]),
                            np.array([1, 2]), np.array(["cough"]),
                            np.array(["", "eat"]), np.array(["cough", " "])):
                with self.subTest(name=name, shape=invalid.shape, dtype=str(invalid.dtype)):
                    fields = self.fields()
                    fields[name] = invalid
                    with self.assertRaises(ValueError):
                        validate(self.write(fields), PROTOCOL)

    def test_rejects_duplicate_ids_and_unknown_labels(self) -> None:
        for name, values in (("sample_ids", ["same", "same"]),
                             ("labels", ["cough", "not-a-class"])):
            with self.subTest(name=name):
                fields = self.fields()
                fields[name] = np.array(values)
                with self.assertRaises(ValueError):
                    validate(self.write(fields), PROTOCOL)

    def test_rejects_nonfloating_embeddings(self) -> None:
        for dtype in (np.complex64, np.int32, np.bool_):
            with self.subTest(dtype=dtype):
                fields = self.fields()
                fields["embeddings"] = fields["embeddings"].astype(dtype)
                with self.assertRaisesRegex(ValueError, "real floating-point"):
                    validate(self.write(fields), PROTOCOL)

    def test_rejects_nonfinite_and_unnormalized_embeddings(self) -> None:
        for value in (float("nan"), float("inf"), 2.0, 0.0):
            with self.subTest(value=value):
                fields = self.fields()
                fields["embeddings"][0, 0] = value
                with self.assertRaises(ValueError):
                    validate(self.write(fields), PROTOCOL)

    def test_rejects_wrong_embedding_shape(self) -> None:
        for shape in ((2, 511), (2, 512, 1), (512,), (0, 512)):
            with self.subTest(shape=shape):
                fields = self.fields()
                fields["embeddings"] = np.zeros(shape, dtype=np.float32)
                with self.assertRaises(ValueError):
                    validate(self.write(fields), PROTOCOL)

    def test_requires_scalar_string_model_and_padding_metadata(self) -> None:
        for name in ("model_id", "audio_padding"):
            for value in (np.array(1), np.array([PROTOCOL["expected_audio_model_id"]]),
                          np.array(""), np.array(" pad ")):
                with self.subTest(name=name, shape=value.shape):
                    fields = self.fields()
                    fields[name] = value
                    with self.assertRaises(ValueError):
                        validate(self.write(fields), PROTOCOL)

    def test_requires_matching_model_and_padding(self) -> None:
        for name, value in (("model_id", "another-model"), ("audio_padding", "repeat")):
            with self.subTest(name=name):
                fields = self.fields()
                fields[name] = np.array(value)
                with self.assertRaisesRegex(ValueError, "model ID or padding"):
                    validate(self.write(fields), PROTOCOL)

    def test_missing_and_nan_window_are_unknown_but_infinity_is_invalid(self) -> None:
        fields = self.fields()
        del fields["analysis_window_seconds"]
        self.assertIsNone(validate(self.write(fields), PROTOCOL)[0]["analysis_window_seconds"])
        fields["analysis_window_seconds"] = np.array(float("nan"))
        self.assertIsNone(validate(self.write(fields), PROTOCOL)[0]["analysis_window_seconds"])
        for value in (np.array(float("inf")), np.array(float("-inf")), np.array([1.0]),
                      np.array("1.0"), np.array(1.0j), np.array(True), np.array(2.0)):
            with self.subTest(value=value):
                fields["analysis_window_seconds"] = value
                with self.assertRaises(ValueError):
                    validate(self.write(fields), PROTOCOL)

    def test_repeated_session_ids_inside_one_split_are_valid(self) -> None:
        fields = self.fields()
        fields["session_ids"] = np.array(["session-a", "session-a"])
        report = validate(self.write(fields), PROTOCOL)[0]
        self.assertTrue(report["session_ids_present"])
        self.assertEqual(report["session_count"], 1)

    def split_pair(self, with_sessions: bool = False) -> tuple[Path, Path]:
        paths = []
        for split in ("train", "query"):
            fields = self.fields(1)
            fields["sample_ids"] = np.array([f"{split}-id"])
            fields["paths"] = np.array([f"{split}.wav"])
            if with_sessions:
                fields["session_ids"] = np.array([f"{split}-session"])
            paths.append(self.write(fields, f"{split}.npz"))
        return paths[0], paths[1]

    def test_cli_rejects_cross_split_alias_even_with_different_sample_ids(self) -> None:
        first, second = self.split_pair()
        fields = self.fields(1)
        fields["paths"] = np.array(["./TRAIN.wav"])
        fields["sample_ids"] = np.array(["query-id"])
        second = self.write(fields, second.name)
        result = self.cli(first, second)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Overlap with an earlier split", result.stderr)

    def test_cli_without_session_metadata_does_not_claim_session_separation(self) -> None:
        result = self.cli(*self.split_pair())
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["disjoint_checked"])
        self.assertFalse(report["session_disjoint_checked"])
        self.assertFalse(report["session_metadata_complete"])
        self.assertFalse(report["audio_content_checked"])

    def test_cli_reports_complete_session_check_only_for_multiple_inputs(self) -> None:
        first, second = self.split_pair(with_sessions=True)
        result = self.cli(first, second)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["session_disjoint_checked"])
        self.assertTrue(report["session_metadata_complete"])
        self.assertNotIn("train-session", result.stdout)
        result = self.cli(first)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)["session_disjoint_checked"])

    def test_cli_rejects_cross_split_session_overlap(self) -> None:
        first, second = self.split_pair(with_sessions=True)
        fields = self.fields(1)
        fields["paths"] = np.array(["query.wav"])
        fields["sample_ids"] = np.array(["query-id"])
        fields["session_ids"] = np.array(["train-session"])
        second = self.write(fields, second.name)
        result = self.cli(first, second)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Session overlap", result.stderr)

    def test_cli_partial_session_metadata_does_not_claim_complete_session_check(self) -> None:
        first, second = self.split_pair()
        fields = self.fields(1)
        fields["sample_ids"] = np.array(["third-id"])
        fields["paths"] = np.array(["third.wav"])
        fields["session_ids"] = np.array(["third-session"])
        third = self.write(fields, "third.npz")
        result = self.cli(first, second, third)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report["session_disjoint_checked"])
        self.assertFalse(report["session_metadata_complete"])

    def test_cli_without_disjoint_flag_does_not_claim_session_check(self) -> None:
        result = self.cli(*self.split_pair(with_sessions=True), disjoint=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report["session_disjoint_checked"])
        self.assertFalse(report["disjoint_checked"])


if __name__ == "__main__":
    unittest.main()
