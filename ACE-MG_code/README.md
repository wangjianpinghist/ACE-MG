# ACE-MG Supporting Code

Supporting utilities for **Asymmetric Candidate Evidence Calibration with Two-Margin Graph Inference (ACE-MG)**, described in the manuscript *Few-shot open-world pig vocalization recognition via asymmetric evidence calibration and two-margin graph inference*.

This package validates user-supplied, precomputed CLAP audio embeddings and recomputes evaluation metrics from saved per-query predictions. It provides no model fitting or inference. The complete ACE-MG implementation, study data, split manifests, pretrained checkpoints, and real predictions are not included; the paper's experimental results cannot be reproduced from this package alone.

## Installation

Use Python 3.9 or newer. The prediction evaluator and synthetic example use only the standard library. NumPy is required for the embedding validator and tests.

```text
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Run all commands below from the package root.

## Synthetic example

```text
python examples/make_synthetic_predictions.py --output demo/predictions.csv
python evaluation/recompute_metrics.py demo/predictions.csv --output-dir demo/evaluation --strict-protocol
```

The example generates **artificial data**, with 18,300 query rows and 3,660 episodes. Its metrics demonstrate the evaluator's operation and are not study results. The evaluator refuses to overwrite a nonempty output directory.

Successful evaluation creates `episode_metrics.csv`, `fold_means.csv`, `summary_mean_sd.csv`, and `metric_definition.json`. Metrics are proportions; multiply by 100 for percentages.

## Saved predictions

Pass one or more CSV or gzip-compressed CSV files:

```text
python evaluation/recompute_metrics.py fold_1.csv.gz fold_2.csv.gz fold_3.csv.gz fold_4.csv.gz fold_5.csv.gz --output-dir metric_check --strict-protocol
```

Required columns:

```text
method,fold,role,novel_class,unknown_class,k,seed,sample_id,true_label,closed_prediction,open_prediction,knownness
```

The public method identifier is `ACE-MG`. Internal class labels map to the manuscript as follows:

| CSV / NPZ label | Manuscript class |
|---|---|
| `normal` | Normal |
| `cough` | Cough |
| `eat` | Feeding |
| `estrus` | Oestrus |
| `squeal` | Scream |

The `role` must describe its actual fields, for example `novel=estrus|unknown=squeal`. `closed_prediction` is the highest-scoring enrolled class. In `open_prediction`, the current Unknown class label denotes a Reject decision. `knownness` is the saved continuous score, defined by the upstream method as the highest final enrolled-class logit minus the final Reject logit. This evaluator reads that score; it cannot reconstruct or verify its provenance.

Rows belonging to one episode must be contiguous and contained in one input file. An episode is identified by method, fold, role, Novel class, Unknown class, K and seed. A sample ID must occur only once within an episode; reuse across support draws or class-role configurations is expected.

`--strict-protocol` checks all five folds; the 12 actual ordered Novel/Unknown pairs from the four non-Normal classes; Normal remaining a Base class; K = 0, 1, 5, 10; seed -1 for K=0 and the 20 seeds 42–61 for K>0; matching role names; and the presence of all five query classes in each episode. These fold, K and seed sets must remain unchanged in a custom protocol file when strict mode is used. At K≥1 it also checks decision consistency: positive knownness requires `open_prediction == closed_prediction`; negative knownness requires Reject; zero permits either decision. Omit strict mode to evaluate an incomplete set of otherwise valid episodes. These checks do not prove that predictions came from ACE-MG, that support examples were correctly sampled, or that recording sessions were isolated.

## Metric definitions

The evaluator uses these explicit query subsets:

| Output | Definition |
|---|---|
| `base_macro_f1` | Macro-F1 over the three Base classes, using **Base-class queries only**. Novel or Unknown queries predicted as Base are excluded from this metric's false-positive counts. |
| `novel_f1_known_only` | Novel-class F1 using **Base and Novel queries only**. Unknown queries predicted as Novel are excluded from this metric's false-positive counts. |
| `unknown_auroc` | Unknown-positive ROC AUC using `-knownness`, with half credit for tied scores. |
| `unknown_aupr` | Unknown-positive **average precision (AP)** using `-knownness`, not trapezoidal PR-curve integration. |
| `oscr_auc` | Area under the CCR–FPR curve obtained by sweeping `knownness`; known-query correctness uses `closed_prediction`, independently of a fixed Reject decision. |

For F1, K=0 uses `closed_prediction`; K≥1 uses `open_prediction`. Metrics are averaged over support-seed episodes within each actual class-role pair, then equally over class-role pairs within a fold, and finally equally over folds. The SD is the **sample SD of fold means (`ddof=1`)**. A partial evaluation with one fold reports SD=0 as a software convention; it is not an estimate of between-fold variability.

The F1 query subsets are part of the metric definition, not an interchangeable implementation detail. Comparisons and manuscript descriptions must use the same subsets. See [evaluation protocol](docs/evaluation_protocol.md) for examples and the boundary of the paper-alignment checks.

## Precomputed audio embeddings

```text
python inputs/validate_audio_embeddings.py train/audio_embeddings.npz query/audio_embeddings.npz --disjoint
```

Required NPZ arrays are `embeddings`, `labels`, `sample_ids`, `paths`, `model_id`, and `audio_padding`. Embeddings must be a nonempty real floating-point matrix with 512 columns, finite values and unit L2 norm (absolute tolerance 1e-3). Labels, IDs and paths are one-dimensional string arrays of the same row count. `model_id` and `audio_padding` are scalar strings matching `laion/clap-htsat-unfused` and `pad`.

Paths are relative to a shared audio root. Path identities use a conservative portable rule: normalize slash direction and `.` components, and compare without case. Absolute, rooted, drive-relative and parent-traversal paths are rejected. Thus `clips/a.wav`, `clips/./a.wav` and `clips\\a.wav` are the same identity. This rule can conservatively reject distinct files differing only by case on a case-sensitive file system.

Optional `analysis_window_seconds` must be 1.0 when specified; absent or NaN metadata is reported as `null`. Infinite values are invalid. This metadata does not prove the duration of the source WAV. Optional `session_ids` is a per-row string array; recording sessions may contain multiple events. With `--disjoint`, cross-file sample IDs and normalized paths are compared, and available session IDs are also checked. `session_disjoint_checked` is true only when at least two inputs all provide session IDs and pass that comparison. Session IDs must use a globally consistent namespace across the supplied files.

These checks do not inspect WAV contents, identify duplicate recordings under unrelated IDs, or establish the correctness of the original recording-session split. The validator does not extract embeddings or run CLAP.

## Protocol and scope

`configs/protocol.json` records the main evaluation settings: GaussianTextMatch with text/support-audio Novel prototypes, nine-feature CSRC with candidate-set shift regularization λ=1000, and two-margin query-graph inference with α=0.35 and 8 iterations. It is documentation for the supporting utilities, not an executable fitting or inference configuration. The five-fold, 12-role and K settings correspond to the associated manuscript.

This package contains no event localization, one-second cutting, waveform normalization, Mel frontend, CLAP feature extraction, GaussianTextMatch/CSRC fitting, graph inference, model training, study recordings, real embeddings, historical results, or split manifests. A dataset, if distributed, is managed separately.

The synthetic checks and unit tests validate utility behavior. They do not verify the paper's numerical results.

## Rights and permissions

The code and documentation included in this supporting-code package are released under the **MIT License**. See `LICENSE`, `LICENSE_STATUS.md` and `THIRD_PARTY_NOTICES.md`. The complete ACE-MG method implementation, study data, split manifests, model weights and real predictions are not included; this release does not license those absent materials.

## 中文说明

本包提供 ACE-MG 的辅助工具，用于检查已有 CLAP 音频嵌入和重算已有预测的指标，不包含核心拟合与推理实现，不能单独复现论文成绩。代码明确区分 Base-only 和 Known-only 的 F1 统计范围；论文、实验结果与对比方法必须使用相同口径。示例数据全部为人工生成，不是论文数据或论文结果。

本包中包含的辅助代码和文档采用 MIT 许可证，完整条款见 `LICENSE`。未包含的核心实现、研究数据、划分清单、模型权重和真实预测不属于本次许可范围。
