from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from ukcat.evaluate import (
    _evaluate_ukcat_ovr_rows_with_scores,
    _evaluate_ukcat_regex_rows,
    _load_labelled_ukcat,
    _score_ukcat_multilabel,
)
from ukcat.ml_ukcat_hybrid import (
    HYBRID_RULE_CHOICES,
    HYBRID_RULE_LABEL_CONF_GATED_REGEX,
    combine_hybrid_predictions,
)
from ukcat.ml_ukcat_ovr import SGD_LOSS_CHOICES

# Evaluation model notes:
# - Regex is the hand-written baseline and emits UK-CAT codes directly from text rules.
# - OvR turns the charity text fields into one shared feature matrix, then trains one
#   binary classifier per UK-CAT code on that same matrix. For each charity row it
#   therefore returns one score per code: "how likely is this label to be present?"
# - The final OvR multilabel prediction is reconstructed by thresholding those
#   per-code scores independently, with optional top-k fallback if no label clears
#   the threshold.
# - Hybrid does not train a third model. It reuses regex outputs plus those same
#   OvR per-code scores, then applies a combination rule to decide the final labels.

# Grid defaults: use sequences for sweeps or single values for fixed runs.
DEFAULT_SAMPLE_FILES = ("data/sample.csv", "data/top2000.csv")
DEFAULT_RANDOM_STATES = (67, 2026, 42, 123, 321)

OVR_LOGISTIC_DEFAULT_THRESHOLDS = (0.10, 0.20, 0.30, 0.35, 0.40)
OVR_LOGISTIC_DEFAULT_NGRAM_MAX_VALUES = (1, 2)
OVR_LOGISTIC_DEFAULT_CHAR_NGRAM_MAX_VALUES = (0, 5)
OVR_LOGISTIC_DEFAULT_C_VALUES = (10.0, 20.0, 30.0)
OVR_LOGISTIC_DEFAULT_CLASS_WEIGHT_MODES = ("balanced",) # balanced and/or none
OVR_LOGISTIC_DEFAULT_TOP_K_FALLBACK_VALUES = (0, 1)

OVR_SVC_DEFAULT_THRESHOLDS = (-0.75, -0.50, -0.25)
OVR_SVC_DEFAULT_NGRAM_MAX_VALUES = (1, 2)
OVR_SVC_DEFAULT_CHAR_NGRAM_MAX_VALUES = (0, 5)
OVR_SVC_DEFAULT_C_VALUES = (0.3, 1.0, 2.0, 3.0)
OVR_SVC_DEFAULT_CLASS_WEIGHT_MODES = ("balanced", "none")
OVR_SVC_DEFAULT_TOP_K_FALLBACK_VALUES = (0, 1)

OVR_SGD_DEFAULT_THRESHOLDS = (0.22, 0.25, 0.275, 0.30, 0.325)
OVR_SGD_DEFAULT_NGRAM_MAX_VALUES = (2,)
OVR_SGD_DEFAULT_CHAR_NGRAM_MAX_VALUES = (0, 5)
OVR_SGD_DEFAULT_LOSSES = ("modified_huber",)
OVR_SGD_DEFAULT_ALPHA_VALUES = (0.0002, 0.0003, 0.0004, 0.0005, 0.00075)
OVR_SGD_DEFAULT_CLASS_WEIGHT_MODES = ("none",)
OVR_SGD_DEFAULT_TOP_K_FALLBACK_VALUES = (0, 1)

HYBRID_LOGISTIC_DEFAULT_THRESHOLDS = (0.125, 0.15, 0.20, 0.25)
HYBRID_LOGISTIC_DEFAULT_NGRAM_MAX_VALUES = (1, 2)
HYBRID_LOGISTIC_DEFAULT_CHAR_NGRAM_MAX_VALUES = (0, 5)
HYBRID_LOGISTIC_DEFAULT_C_VALUES = (2500, 3000, 3500, 4000)
HYBRID_LOGISTIC_DEFAULT_CLASS_WEIGHT_MODES = ("none",) # balanced and/or none
HYBRID_LOGISTIC_DEFAULT_TOP_K_FALLBACK_VALUES = (0, 1)
HYBRID_LOGISTIC_DEFAULT_LABEL_CONFIDENCE_THRESHOLDS = (0.00002, 0.00003, 0.00004, 0.00005, 0.000075)

HYBRID_SVC_DEFAULT_THRESHOLDS = (-0.4, -0.3, -0.2, -0.1, 0.0)
HYBRID_SVC_DEFAULT_NGRAM_MAX_VALUES = (2, )
HYBRID_SVC_DEFAULT_CHAR_NGRAM_MAX_VALUES = (0, 5)
HYBRID_SVC_DEFAULT_C_VALUES = (0.3, 0.5, 1.0)
HYBRID_SVC_DEFAULT_CLASS_WEIGHT_MODES = ("balanced", )
HYBRID_SVC_DEFAULT_TOP_K_FALLBACK_VALUES = (1, )
HYBRID_SVC_DEFAULT_LABEL_CONFIDENCE_THRESHOLDS = (0.125, 0.25, 0.35)

HYBRID_SGD_DEFAULT_THRESHOLDS = (0.24, 0.25, 0.26, 0.275, 0.29, 0.30, 0.325)
HYBRID_SGD_DEFAULT_NGRAM_MAX_VALUES = (2,)
HYBRID_SGD_DEFAULT_CHAR_NGRAM_MAX_VALUES = (0, 4, 5, 6, 7)
HYBRID_SGD_DEFAULT_LOSSES = ("log_loss",)
HYBRID_SGD_DEFAULT_ALPHA_VALUES = (0.000003, 0.000005, 0.0000075, 0.00001, 0.000015, 0.00002)
HYBRID_SGD_DEFAULT_CLASS_WEIGHT_MODES = ("none",)
HYBRID_SGD_DEFAULT_TOP_K_FALLBACK_VALUES = (0,)
HYBRID_SGD_DEFAULT_LABEL_CONFIDENCE_THRESHOLDS = (0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02)
DEFAULT_CLEAN_TEXT_MODE = "on" # can be set to 'on', 'off' or 'compare'
DEFAULT_FIELD_SETS = (
    ("name", "activities"),
    #("name", "activities", "objects"), # Commented out as objects consistently harms performance
)
# Top-level model toggles. These control both which approaches are searched and
# which approaches are displayed in comparison output.
DEFAULT_ENABLE_REGEX = True
DEFAULT_ENABLE_OVR_LOGISTIC = True
DEFAULT_ENABLE_OVR_SVC = False
DEFAULT_ENABLE_OVR_SGD = True
DEFAULT_ENABLE_HYBRID_LOGISTIC = True
DEFAULT_ENABLE_HYBRID_SVC = False
DEFAULT_ENABLE_HYBRID_SGD = True

DEFAULT_OPTIMISE_PRECISION_MICRO = False
DEFAULT_OPTIMISE_RECALL_MICRO = False
DEFAULT_OPTIMISE_F1_MICRO = False
DEFAULT_OPTIMISE_F1_MACRO = False
DEFAULT_OPTIMISE_SUBSET_ACCURACY = False
DEFAULT_OPTIMISE_JACCARD_SAMPLES = False
DEFAULT_OPTIMISE_HAMMING_LOSS = False
DEFAULT_OPTIMISE_WEIGHTED_PRIMARY = True

OPTIMISATION_METRIC_SPECS = (
    ("weighted_primary", DEFAULT_OPTIMISE_WEIGHTED_PRIMARY, "DEFAULT_OPTIMISE_WEIGHTED_PRIMARY"),
    ("subset_accuracy", DEFAULT_OPTIMISE_SUBSET_ACCURACY, "DEFAULT_OPTIMISE_SUBSET_ACCURACY"),
    ("precision_micro", DEFAULT_OPTIMISE_PRECISION_MICRO, "DEFAULT_OPTIMISE_PRECISION_MICRO"),
    ("recall_micro", DEFAULT_OPTIMISE_RECALL_MICRO, "DEFAULT_OPTIMISE_RECALL_MICRO"),
    ("f1_micro", DEFAULT_OPTIMISE_F1_MICRO, "DEFAULT_OPTIMISE_F1_MICRO"),
    ("f1_macro", DEFAULT_OPTIMISE_F1_MACRO, "DEFAULT_OPTIMISE_F1_MACRO"),
    ("jaccard_samples", DEFAULT_OPTIMISE_JACCARD_SAMPLES, "DEFAULT_OPTIMISE_JACCARD_SAMPLES"),
    ("hamming_loss", DEFAULT_OPTIMISE_HAMMING_LOSS, "DEFAULT_OPTIMISE_HAMMING_LOSS"),
)
OPTIMISATION_METRICS = tuple(metric for metric, _, _ in OPTIMISATION_METRIC_SPECS)

# Weighted objective coefficients used when DEFAULT_OPTIMISE_WEIGHTED_PRIMARY is True.
# Keep these non-negative and summing to 1.0.
WEIGHTED_PRIMARY_F1_MICRO = 0.35
WEIGHTED_PRIMARY_F1_MACRO = 0.40
WEIGHTED_PRIMARY_RECALL_MICRO = 0.15
WEIGHTED_PRIMARY_PRECISION_MICRO = 0.10

# Optional uardrails (applied to the candidate approach being ranked: OVR and/or Hybrid)
# Setting to 0 essentially turns that guardrail 'off'
GUARDRAIL_MIN_PRECISION_MICRO = 0
GUARDRAIL_MIN_F1_MICRO = 0.40
GUARDRAIL_MIN_SUBSET_ACCURACY = 0.10
GUARDRAIL_MAX_HAMMING_LOSS_EXCLUSIVE = 0.0160
GUARDRAIL_MIN_JACCARD_SAMPLES = 0.0

DISPLAY_METRICS = OPTIMISATION_METRICS
ALL_APPROACHES = ("regex", "ovr_logistic", "ovr_svc", "ovr_sgd", "hybrid_logistic", "hybrid_svc", "hybrid_sgd")
APPROACHES = tuple(
    approach
    for approach, enabled in (
        ("regex", DEFAULT_ENABLE_REGEX),
        ("ovr_logistic", DEFAULT_ENABLE_OVR_LOGISTIC),
        ("ovr_svc", DEFAULT_ENABLE_OVR_SVC),
        ("ovr_sgd", DEFAULT_ENABLE_OVR_SGD),
        ("hybrid_logistic", DEFAULT_ENABLE_HYBRID_LOGISTIC),
        ("hybrid_svc", DEFAULT_ENABLE_HYBRID_SVC),
        ("hybrid_sgd", DEFAULT_ENABLE_HYBRID_SGD),
    )
    if enabled
)
BASE_METRICS = ("rows",) + OPTIMISATION_METRICS
GRID_GROUP_COLUMNS = (
    "grid_approach",
    "fields_key",
    "clean_text",
    "model_family",
    "threshold",
    "ngram_max",
    "char_ngram_max",
    "model_c",
    "sgd_loss",
    "sgd_alpha",
    "class_weight_mode",
    "top_k_fallback",
    "hybrid_label_confidence_threshold",
)
AGGREGATED_METRIC_COLUMNS = tuple(
    f"{approach}_{metric}"
    for approach in ALL_APPROACHES
    for metric in BASE_METRICS
)
GUARDRAIL_SPECS = (
    ("precision_micro", ">=", GUARDRAIL_MIN_PRECISION_MICRO),
    ("f1_micro", ">=", GUARDRAIL_MIN_F1_MICRO),
    ("subset_accuracy", ">=", GUARDRAIL_MIN_SUBSET_ACCURACY),
    ("hamming_loss", "<", GUARDRAIL_MAX_HAMMING_LOSS_EXCLUSIVE),
    ("jaccard_samples", ">=", GUARDRAIL_MIN_JACCARD_SAMPLES),
)

MetricMap = dict[str, float]
FieldSet = tuple[str, ...]


@dataclass(frozen=True)
class GridParams:
    grid_approach: str
    fields: tuple[str, ...]
    clean_text: bool
    model_family: str
    threshold: float
    ngram_max: int
    char_ngram_max: int
    model_c: float
    sgd_loss: str
    sgd_alpha: float
    class_weight_mode: str
    top_k_fallback: int
    hybrid_label_confidence_threshold: float


def _parse_csv_list(value: str | int | float | Sequence[str]) -> list[str]:
    if isinstance(value, (tuple, list)):
        return [str(v) for v in value]
    if isinstance(value, (int, float)):
        return [str(value)]
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_int_csv(value: str) -> list[int]:
    return [int(v) for v in _parse_csv_list(value)]


def _parse_float_csv(value: str) -> list[float]:
    return [float(v) for v in _parse_csv_list(value)]


def _parse_field_sets(values: Sequence[str]) -> list[FieldSet]:
    if values:
        return [tuple(_parse_csv_list(value)) for value in values]
    return [tuple(v) for v in DEFAULT_FIELD_SETS]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run UKCAT compare-style evaluation over an OvR parameter grid and report "
            "averaged metrics across random states."
        )
    )
    parser.add_argument(
        "--sample-file",
        dest="sample_files",
        action="append",
        default=[],
        help="Labelled CSV file (repeatable). Defaults to data/sample.csv and data/top2000.csv.",
    )
    parser.add_argument(
        "--random-states",
        default=",".join(_parse_csv_list(DEFAULT_RANDOM_STATES)),
        help="Comma-separated random states for shared holdout splits.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Single random state override (use instead of --random-states).",
    )
    parser.add_argument(
        "--ovr-logistic-thresholds",
        default=",".join(_parse_csv_list(OVR_LOGISTIC_DEFAULT_THRESHOLDS)),
        help="Comma-separated OVR logistic thresholds to test.",
    )
    parser.add_argument(
        "--ovr-logistic-threshold",
        type=float,
        default=None,
        help="Single OVR logistic threshold override.",
    )
    parser.add_argument(
        "--ovr-logistic-ngram-max-values",
        default=",".join(_parse_csv_list(OVR_LOGISTIC_DEFAULT_NGRAM_MAX_VALUES)),
        help="Comma-separated OVR logistic maximum n-gram sizes (uses 1..N).",
    )
    parser.add_argument(
        "--ovr-logistic-ngram-max",
        type=int,
        default=None,
        help="Single OVR logistic n-gram max override.",
    )
    parser.add_argument(
        "--ovr-logistic-char-ngram-max-values",
        default=",".join(_parse_csv_list(OVR_LOGISTIC_DEFAULT_CHAR_NGRAM_MAX_VALUES)),
        help="Comma-separated OVR logistic maximum char_wb n-gram sizes (0 disables char features).",
    )
    parser.add_argument(
        "--ovr-logistic-char-ngram-max",
        type=int,
        default=None,
        help="Single OVR logistic char_wb n-gram max override.",
    )
    parser.add_argument(
        "--ovr-logistic-c-values",
        default=",".join(_parse_csv_list(OVR_LOGISTIC_DEFAULT_C_VALUES)),
        help="Comma-separated OVR LogisticRegression C values to test.",
    )
    parser.add_argument("--ovr-logistic-c", type=float, default=None, help="Single OVR LogisticRegression C override.")
    parser.add_argument(
        "--ovr-class-weight-modes",
        default=",".join(_parse_csv_list(OVR_LOGISTIC_DEFAULT_CLASS_WEIGHT_MODES)),
        help="Comma-separated OVR logistic class-weight modes to test (none, balanced).",
    )
    parser.add_argument(
        "--ovr-class-weight-mode",
        choices=("none", "balanced"),
        default=None,
        help="Single OVR logistic class-weight mode override.",
    )
    parser.add_argument(
        "--ovr-top-k-fallback-values",
        default=",".join(_parse_csv_list(OVR_LOGISTIC_DEFAULT_TOP_K_FALLBACK_VALUES)),
        help="Comma-separated OVR logistic fallback values (0 disables fallback).",
    )
    parser.add_argument(
        "--ovr-top-k-fallback",
        type=int,
        default=None,
        help="Single OVR logistic top-k fallback override.",
    )
    parser.add_argument(
        "--ovr-svc-thresholds",
        default=",".join(_parse_csv_list(OVR_SVC_DEFAULT_THRESHOLDS)),
        help="Comma-separated OVR LinearSVC decision thresholds to test.",
    )
    parser.add_argument("--ovr-svc-threshold", type=float, default=None, help="Single OVR SVC threshold override.")
    parser.add_argument(
        "--ovr-svc-ngram-max-values",
        default=",".join(_parse_csv_list(OVR_SVC_DEFAULT_NGRAM_MAX_VALUES)),
        help="Comma-separated OVR SVC maximum n-gram sizes (uses 1..N).",
    )
    parser.add_argument("--ovr-svc-ngram-max", type=int, default=None, help="Single OVR SVC n-gram max override.")
    parser.add_argument(
        "--ovr-svc-char-ngram-max-values",
        default=",".join(_parse_csv_list(OVR_SVC_DEFAULT_CHAR_NGRAM_MAX_VALUES)),
        help="Comma-separated OVR SVC maximum char_wb n-gram sizes (0 disables char features).",
    )
    parser.add_argument("--ovr-svc-char-ngram-max", type=int, default=None, help="Single OVR SVC char_wb n-gram max override.")
    parser.add_argument(
        "--ovr-svc-c-values",
        default=",".join(_parse_csv_list(OVR_SVC_DEFAULT_C_VALUES)),
        help="Comma-separated OVR LinearSVC C values to test.",
    )
    parser.add_argument("--ovr-svc-c", type=float, default=None, help="Single OVR LinearSVC C override.")
    parser.add_argument(
        "--ovr-svc-class-weight-modes",
        default=",".join(_parse_csv_list(OVR_SVC_DEFAULT_CLASS_WEIGHT_MODES)),
        help="Comma-separated OVR SVC class-weight modes to test (none, balanced).",
    )
    parser.add_argument(
        "--ovr-svc-class-weight-mode",
        choices=("none", "balanced"),
        default=None,
        help="Single OVR SVC class-weight mode override.",
    )
    parser.add_argument(
        "--ovr-svc-top-k-fallback-values",
        default=",".join(_parse_csv_list(OVR_SVC_DEFAULT_TOP_K_FALLBACK_VALUES)),
        help="Comma-separated OVR SVC fallback values (0 disables fallback).",
    )
    parser.add_argument("--ovr-svc-top-k-fallback", type=int, default=None, help="Single OVR SVC top-k fallback override.")
    parser.add_argument(
        "--ovr-sgd-thresholds",
        default=",".join(_parse_csv_list(OVR_SGD_DEFAULT_THRESHOLDS)),
        help="Comma-separated OVR SGD thresholds to test.",
    )
    parser.add_argument("--ovr-sgd-threshold", type=float, default=None, help="Single OVR SGD threshold override.")
    parser.add_argument(
        "--ovr-sgd-ngram-max-values",
        default=",".join(_parse_csv_list(OVR_SGD_DEFAULT_NGRAM_MAX_VALUES)),
        help="Comma-separated OVR SGD maximum n-gram sizes (uses 1..N).",
    )
    parser.add_argument("--ovr-sgd-ngram-max", type=int, default=None, help="Single OVR SGD n-gram max override.")
    parser.add_argument(
        "--ovr-sgd-char-ngram-max-values",
        default=",".join(_parse_csv_list(OVR_SGD_DEFAULT_CHAR_NGRAM_MAX_VALUES)),
        help="Comma-separated OVR SGD maximum char_wb n-gram sizes (0 disables char features).",
    )
    parser.add_argument("--ovr-sgd-char-ngram-max", type=int, default=None, help="Single OVR SGD char_wb n-gram max override.")
    parser.add_argument(
        "--ovr-sgd-losses",
        default=",".join(_parse_csv_list(OVR_SGD_DEFAULT_LOSSES)),
        help="Comma-separated OVR SGD losses to test.",
    )
    parser.add_argument(
        "--ovr-sgd-loss",
        choices=SGD_LOSS_CHOICES,
        default=None,
        help="Single OVR SGD loss override.",
    )
    parser.add_argument(
        "--ovr-sgd-alpha-values",
        default=",".join(_parse_csv_list(OVR_SGD_DEFAULT_ALPHA_VALUES)),
        help="Comma-separated OVR SGD alpha values to test.",
    )
    parser.add_argument("--ovr-sgd-alpha", type=float, default=None, help="Single OVR SGD alpha override.")
    parser.add_argument(
        "--ovr-sgd-class-weight-modes",
        default=",".join(_parse_csv_list(OVR_SGD_DEFAULT_CLASS_WEIGHT_MODES)),
        help="Comma-separated OVR SGD class-weight modes to test (none, balanced).",
    )
    parser.add_argument(
        "--ovr-sgd-class-weight-mode",
        choices=("none", "balanced"),
        default=None,
        help="Single OVR SGD class-weight mode override.",
    )
    parser.add_argument(
        "--ovr-sgd-top-k-fallback-values",
        default=",".join(_parse_csv_list(OVR_SGD_DEFAULT_TOP_K_FALLBACK_VALUES)),
        help="Comma-separated OVR SGD fallback values (0 disables fallback).",
    )
    parser.add_argument("--ovr-sgd-top-k-fallback", type=int, default=None, help="Single OVR SGD top-k fallback override.")
    parser.add_argument(
        "--hybrid-logistic-thresholds",
        default=",".join(_parse_csv_list(HYBRID_LOGISTIC_DEFAULT_THRESHOLDS)),
        help="Comma-separated hybrid logistic thresholds to test.",
    )
    parser.add_argument("--hybrid-logistic-threshold", type=float, default=None, help="Single hybrid logistic threshold override.")
    parser.add_argument(
        "--hybrid-logistic-ngram-max-values",
        default=",".join(_parse_csv_list(HYBRID_LOGISTIC_DEFAULT_NGRAM_MAX_VALUES)),
        help="Comma-separated hybrid logistic maximum n-gram sizes (uses 1..N).",
    )
    parser.add_argument("--hybrid-logistic-ngram-max", type=int, default=None, help="Single hybrid logistic n-gram max override.")
    parser.add_argument(
        "--hybrid-logistic-char-ngram-max-values",
        default=",".join(_parse_csv_list(HYBRID_LOGISTIC_DEFAULT_CHAR_NGRAM_MAX_VALUES)),
        help="Comma-separated hybrid logistic maximum char_wb n-gram sizes (0 disables char features).",
    )
    parser.add_argument("--hybrid-logistic-char-ngram-max", type=int, default=None, help="Single hybrid logistic char_wb n-gram max override.")
    parser.add_argument(
        "--hybrid-logistic-c-values",
        default=",".join(_parse_csv_list(HYBRID_LOGISTIC_DEFAULT_C_VALUES)),
        help="Comma-separated hybrid logistic C values to test.",
    )
    parser.add_argument("--hybrid-logistic-c", type=float, default=None, help="Single hybrid logistic C override.")
    parser.add_argument(
        "--hybrid-logistic-class-weight-modes",
        default=",".join(_parse_csv_list(HYBRID_LOGISTIC_DEFAULT_CLASS_WEIGHT_MODES)),
        help="Comma-separated hybrid logistic class-weight modes to test (none, balanced).",
    )
    parser.add_argument(
        "--hybrid-logistic-class-weight-mode",
        choices=("none", "balanced"),
        default=None,
        help="Single hybrid logistic class-weight mode override.",
    )
    parser.add_argument(
        "--hybrid-logistic-top-k-fallback-values",
        default=",".join(_parse_csv_list(HYBRID_LOGISTIC_DEFAULT_TOP_K_FALLBACK_VALUES)),
        help="Comma-separated hybrid logistic fallback values (0 disables fallback).",
    )
    parser.add_argument(
        "--hybrid-logistic-top-k-fallback",
        type=int,
        default=None,
        help="Single hybrid logistic top-k fallback override.",
    )
    parser.add_argument(
        "--hybrid-logistic-label-confidence-threshold-values",
        default=",".join(_parse_csv_list(HYBRID_LOGISTIC_DEFAULT_LABEL_CONFIDENCE_THRESHOLDS)),
        help="Comma-separated hybrid logistic label confidence thresholds for label_conf_gated_regex.",
    )
    parser.add_argument(
        "--hybrid-logistic-label-confidence-threshold",
        type=float,
        default=None,
        help="Single hybrid logistic label confidence threshold override.",
    )
    parser.add_argument(
        "--hybrid-svc-thresholds",
        default=",".join(_parse_csv_list(HYBRID_SVC_DEFAULT_THRESHOLDS)),
        help="Comma-separated hybrid SVC thresholds to test.",
    )
    parser.add_argument("--hybrid-svc-threshold", type=float, default=None, help="Single hybrid SVC threshold override.")
    parser.add_argument(
        "--hybrid-svc-ngram-max-values",
        default=",".join(_parse_csv_list(HYBRID_SVC_DEFAULT_NGRAM_MAX_VALUES)),
        help="Comma-separated hybrid SVC maximum n-gram sizes (uses 1..N).",
    )
    parser.add_argument("--hybrid-svc-ngram-max", type=int, default=None, help="Single hybrid SVC n-gram max override.")
    parser.add_argument(
        "--hybrid-svc-char-ngram-max-values",
        default=",".join(_parse_csv_list(HYBRID_SVC_DEFAULT_CHAR_NGRAM_MAX_VALUES)),
        help="Comma-separated hybrid SVC maximum char_wb n-gram sizes (0 disables char features).",
    )
    parser.add_argument("--hybrid-svc-char-ngram-max", type=int, default=None, help="Single hybrid SVC char_wb n-gram max override.")
    parser.add_argument(
        "--hybrid-svc-c-values",
        default=",".join(_parse_csv_list(HYBRID_SVC_DEFAULT_C_VALUES)),
        help="Comma-separated hybrid SVC C values to test.",
    )
    parser.add_argument("--hybrid-svc-c", type=float, default=None, help="Single hybrid SVC C override.")
    parser.add_argument(
        "--hybrid-svc-class-weight-modes",
        default=",".join(_parse_csv_list(HYBRID_SVC_DEFAULT_CLASS_WEIGHT_MODES)),
        help="Comma-separated hybrid SVC class-weight modes to test (none, balanced).",
    )
    parser.add_argument(
        "--hybrid-svc-class-weight-mode",
        choices=("none", "balanced"),
        default=None,
        help="Single hybrid SVC class-weight mode override.",
    )
    parser.add_argument(
        "--hybrid-svc-top-k-fallback-values",
        default=",".join(_parse_csv_list(HYBRID_SVC_DEFAULT_TOP_K_FALLBACK_VALUES)),
        help="Comma-separated hybrid SVC fallback values (0 disables fallback).",
    )
    parser.add_argument(
        "--hybrid-svc-top-k-fallback",
        type=int,
        default=None,
        help="Single hybrid SVC top-k fallback override.",
    )
    parser.add_argument(
        "--hybrid-svc-label-confidence-threshold-values",
        default=",".join(_parse_csv_list(HYBRID_SVC_DEFAULT_LABEL_CONFIDENCE_THRESHOLDS)),
        help="Comma-separated hybrid SVC label confidence thresholds for label_conf_gated_regex.",
    )
    parser.add_argument(
        "--hybrid-svc-label-confidence-threshold",
        type=float,
        default=None,
        help="Single hybrid SVC label confidence threshold override.",
    )
    parser.add_argument(
        "--hybrid-sgd-thresholds",
        default=",".join(_parse_csv_list(HYBRID_SGD_DEFAULT_THRESHOLDS)),
        help="Comma-separated hybrid SGD thresholds to test.",
    )
    parser.add_argument("--hybrid-sgd-threshold", type=float, default=None, help="Single hybrid SGD threshold override.")
    parser.add_argument(
        "--hybrid-sgd-ngram-max-values",
        default=",".join(_parse_csv_list(HYBRID_SGD_DEFAULT_NGRAM_MAX_VALUES)),
        help="Comma-separated hybrid SGD maximum n-gram sizes (uses 1..N).",
    )
    parser.add_argument("--hybrid-sgd-ngram-max", type=int, default=None, help="Single hybrid SGD n-gram max override.")
    parser.add_argument(
        "--hybrid-sgd-char-ngram-max-values",
        default=",".join(_parse_csv_list(HYBRID_SGD_DEFAULT_CHAR_NGRAM_MAX_VALUES)),
        help="Comma-separated hybrid SGD maximum char_wb n-gram sizes (0 disables char features).",
    )
    parser.add_argument("--hybrid-sgd-char-ngram-max", type=int, default=None, help="Single hybrid SGD char_wb n-gram max override.")
    parser.add_argument(
        "--hybrid-sgd-losses",
        default=",".join(_parse_csv_list(HYBRID_SGD_DEFAULT_LOSSES)),
        help="Comma-separated hybrid SGD losses to test.",
    )
    parser.add_argument(
        "--hybrid-sgd-loss",
        choices=SGD_LOSS_CHOICES,
        default=None,
        help="Single hybrid SGD loss override.",
    )
    parser.add_argument(
        "--hybrid-sgd-alpha-values",
        default=",".join(_parse_csv_list(HYBRID_SGD_DEFAULT_ALPHA_VALUES)),
        help="Comma-separated hybrid SGD alpha values to test.",
    )
    parser.add_argument("--hybrid-sgd-alpha", type=float, default=None, help="Single hybrid SGD alpha override.")
    parser.add_argument(
        "--hybrid-sgd-class-weight-modes",
        default=",".join(_parse_csv_list(HYBRID_SGD_DEFAULT_CLASS_WEIGHT_MODES)),
        help="Comma-separated hybrid SGD class-weight modes to test (none, balanced).",
    )
    parser.add_argument(
        "--hybrid-sgd-class-weight-mode",
        choices=("none", "balanced"),
        default=None,
        help="Single hybrid SGD class-weight mode override.",
    )
    parser.add_argument(
        "--hybrid-sgd-top-k-fallback-values",
        default=",".join(_parse_csv_list(HYBRID_SGD_DEFAULT_TOP_K_FALLBACK_VALUES)),
        help="Comma-separated hybrid SGD fallback values (0 disables fallback).",
    )
    parser.add_argument(
        "--hybrid-sgd-top-k-fallback",
        type=int,
        default=None,
        help="Single hybrid SGD top-k fallback override.",
    )
    parser.add_argument(
        "--hybrid-sgd-label-confidence-threshold-values",
        default=",".join(_parse_csv_list(HYBRID_SGD_DEFAULT_LABEL_CONFIDENCE_THRESHOLDS)),
        help="Comma-separated hybrid SGD label confidence thresholds for label_conf_gated_regex.",
    )
    parser.add_argument(
        "--hybrid-sgd-label-confidence-threshold",
        type=float,
        default=None,
        help="Single hybrid SGD label confidence threshold override.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout test fraction.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="OvR parallel jobs.",
    )
    parser.add_argument(
        "--fields",
        action="append",
        default=[],
        help=(
            "Field set used by OvR (repeatable, comma-separated). "
            "If omitted, defaults to two sets: name,activities and name,activities,objects."
        ),
    )
    parser.add_argument(
        "--clean-text-mode",
        choices=("off", "on", "compare"),
        default=DEFAULT_CLEAN_TEXT_MODE,
        help="Use raw text only, cleaned text only, or compare both in the grid.",
    )
    parser.add_argument(
        "--include-groups",
        action="store_true",
        help="Include parent/group codes in regex predictions.",
    )
    parser.add_argument(
        "--hybrid-rule",
        choices=HYBRID_RULE_CHOICES,
        default=HYBRID_RULE_LABEL_CONF_GATED_REGEX,
        help="Hybrid decision rule used when ranking the hybrid approach.",
    )
    parser.add_argument(
        "--show-top",
        type=int,
        default=0,
        help="Print the top N parameter combinations (0 disables). In dual mode this is applied to each model.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each parameter combination during grid evaluation instead of using a progress bar.",
    )
    parser.add_argument(
        "--save-summary",
        type=str,
        default=None,
        help="Optional CSV path for aggregated grid results.",
    )
    parser.add_argument(
        "--save-per-state",
        type=str,
        default=None,
        help="Optional CSV path for per-state/per-parameter results.",
    )
    return parser


def _resolve_optimise_metric_from_knobs() -> str:
    selected = [metric for metric, enabled, _ in OPTIMISATION_METRIC_SPECS if enabled]
    if len(selected) != 1:
        raise SystemExit(
            "Exactly one metric knob must be True: "
            + ", ".join(knob_name for _, _, knob_name in OPTIMISATION_METRIC_SPECS)
        )
    return selected[0]


def _resolve_selected_approaches_from_knobs() -> list[str]:
    selected_knobs = {
        "ovr_logistic": DEFAULT_ENABLE_OVR_LOGISTIC,
        "ovr_svc": DEFAULT_ENABLE_OVR_SVC,
        "ovr_sgd": DEFAULT_ENABLE_OVR_SGD,
        "hybrid_logistic": DEFAULT_ENABLE_HYBRID_LOGISTIC,
        "hybrid_svc": DEFAULT_ENABLE_HYBRID_SVC,
        "hybrid_sgd": DEFAULT_ENABLE_HYBRID_SGD,
    }
    selected = [name for name, enabled in selected_knobs.items() if enabled]
    if not selected:
        raise SystemExit(
            "At least one model selection knob must be True: "
            "DEFAULT_ENABLE_OVR_LOGISTIC, DEFAULT_ENABLE_OVR_SVC, "
            "DEFAULT_ENABLE_OVR_SGD, DEFAULT_ENABLE_HYBRID_LOGISTIC, "
            "DEFAULT_ENABLE_HYBRID_SVC, and/or DEFAULT_ENABLE_HYBRID_SGD."
        )
    return selected


def _aggregate_results(per_state_df: pd.DataFrame) -> pd.DataFrame:
    mean_df = per_state_df.groupby(list(GRID_GROUP_COLUMNS))[list(AGGREGATED_METRIC_COLUMNS)].mean().reset_index()
    std_df = per_state_df.groupby(list(GRID_GROUP_COLUMNS))[list(AGGREGATED_METRIC_COLUMNS)].std().reset_index()

    # Suffix columns so the output table is explicit about what is averaged.
    mean_df = mean_df.rename(columns={c: f"{c}_mean" for c in AGGREGATED_METRIC_COLUMNS})
    std_df = std_df.rename(columns={c: f"{c}_std" for c in AGGREGATED_METRIC_COLUMNS})

    return mean_df.merge(std_df, on=list(GRID_GROUP_COLUMNS), how="left")


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _decimal_places(token: str) -> int:
    value = token.strip()
    if not value:
        return 0
    # argparse may parse tiny floats as scientific notation (e.g., 1e-05).
    if "e" in value.lower():
        fixed = f"{float(value):.12f}".rstrip("0").rstrip(".")
        if "." not in fixed:
            return 0
        return len(fixed.split(".")[-1])
    if "." not in value:
        return 0
    return len(value.split(".")[-1])


def _resolve_hybrid_conf_precision(args: argparse.Namespace) -> int:
    precision = 0
    for raw_values in (
        getattr(args, "hybrid_logistic_label_confidence_threshold_values", ""),
        getattr(args, "hybrid_svc_label_confidence_threshold_values", ""),
        getattr(args, "hybrid_sgd_label_confidence_threshold_values", ""),
    ):
        if isinstance(raw_values, str) and raw_values.strip():
            precision = max(precision, max(_decimal_places(part) for part in raw_values.split(",") if part.strip()))
    for single_value in (
        getattr(args, "hybrid_logistic_label_confidence_threshold", None),
        getattr(args, "hybrid_svc_label_confidence_threshold", None),
        getattr(args, "hybrid_sgd_label_confidence_threshold", None),
    ):
        if single_value is not None:
            precision = max(precision, _decimal_places(str(single_value)))
    return max(4, precision)


def _fmt_hybrid_conf(value: float, precision: int) -> str:
    return f"{value:.{precision}f}"


def _weighted_primary(
    f1_micro: float,
    f1_macro: float,
    recall_micro: float,
    precision_micro: float,
) -> float:
    return (
        WEIGHTED_PRIMARY_F1_MICRO * f1_micro
        + WEIGHTED_PRIMARY_F1_MACRO * f1_macro
        + WEIGHTED_PRIMARY_RECALL_MICRO * recall_micro
        + WEIGHTED_PRIMARY_PRECISION_MICRO * precision_micro
    )


def _validate_weighted_primary_weights() -> None:
    total = (
        WEIGHTED_PRIMARY_F1_MICRO
        + WEIGHTED_PRIMARY_F1_MACRO
        + WEIGHTED_PRIMARY_RECALL_MICRO
        + WEIGHTED_PRIMARY_PRECISION_MICRO
    )
    if min(
        WEIGHTED_PRIMARY_F1_MICRO,
        WEIGHTED_PRIMARY_F1_MACRO,
        WEIGHTED_PRIMARY_RECALL_MICRO,
        WEIGHTED_PRIMARY_PRECISION_MICRO,
    ) < 0:
        raise SystemExit("Weighted objective coefficients must be non-negative.")
    if abs(total - 1.0) > 1e-9:
        raise SystemExit(
            "Weighted objective coefficients must sum to 1.0 "
            f"(currently {total:.6f})."
        )


def _predict_codes(
    prob_df: pd.DataFrame,
    threshold: float,
    top_k_fallback: int,
) -> list[list[str]]:
    # OvR produces one probability per label. The multilabel prediction is then
    # reconstructed by thresholding each label independently, with an optional
    # fallback that forces the top-k labels on rows where nothing clears the cut
    pred_binary = (prob_df.values >= threshold).astype(int)
    if top_k_fallback > 0 and pred_binary.shape[1] > 0:
        k = min(top_k_fallback, pred_binary.shape[1])
        for i in range(pred_binary.shape[0]):
            if pred_binary[i].sum() > 0:
                continue
            top_idx = prob_df.iloc[i].to_numpy().argsort()[-k:]
            pred_binary[i, top_idx] = 1

    labels = list(map(str, prob_df.columns))
    predicted_codes: list[list[str]] = []
    for row in pred_binary:
        row_labels = [labels[j] for j, value in enumerate(row) if value == 1]
        predicted_codes.append(sorted(set(row_labels)))
    return predicted_codes


def _build_eval(
    prob_df: pd.DataFrame,
    true_codes_by_org: pd.Series,
    threshold: float,
    top_k_fallback: int,
) -> pd.DataFrame:
    # At this point the OvR model has already produced a probability table with:
    # - one row per charity
    # - one column per UK-CAT code
    # This helper converts that dense score table back into the repo's standard
    # multilabel shape: a list of predicted codes per charity row.
    predicted_codes = _predict_codes(
        prob_df=prob_df,
        threshold=threshold,
        top_k_fallback=top_k_fallback,
    )
    org_ids = prob_df.index.astype(str)
    aligned_true_codes = true_codes_by_org.reindex(org_ids).tolist()
    return pd.DataFrame(
        {
            "org_id": org_ids,
            "true_codes": aligned_true_codes,
            "predicted_codes": predicted_codes,
            "prediction_source": "ml_model_holdout_ovr",
        }
    )


def _score_pair(
    regex_eval_df: pd.DataFrame,
    regex_metrics: MetricMap,
    ovr_probability_df: pd.DataFrame,
    true_codes_by_org: pd.Series,
    params: GridParams,
    hybrid_rule: str,
) -> dict:
    # "ovr" here is a pure One-vs-Rest multilabel model:
    # - input text is built from the selected charity fields, such as name and activities
    # - that text is vectorised once into a shared feature space
    # - one classifier is trained for each UK-CAT code using those shared features
    # The result is a probability per charity/per code, which is then thresholded
    # back into a set of predicted labels for the row.
    ovr_eval_df = _build_eval(
        prob_df=ovr_probability_df,
        true_codes_by_org=true_codes_by_org,
        threshold=params.threshold,
        top_k_fallback=params.top_k_fallback,
    )
    # Hybrid is a second decision layer, not a separate fit. It takes:
    # - regex labels, which encode strong hand-written rules
    # - OvR scores, which capture softer statistical evidence from text
    # and combines them into one final prediction set.
    pure_ovr_metrics = _score_ukcat_multilabel(ovr_eval_df)
    ovr_logistic_metrics = {metric: float("nan") for metric in BASE_METRICS}
    ovr_svc_metrics = {metric: float("nan") for metric in BASE_METRICS}
    ovr_sgd_metrics = {metric: float("nan") for metric in BASE_METRICS}
    if params.grid_approach == "ovr_logistic":
        ovr_logistic_metrics = pure_ovr_metrics
    elif params.grid_approach == "ovr_svc":
        ovr_svc_metrics = pure_ovr_metrics
    elif params.grid_approach == "ovr_sgd":
        ovr_sgd_metrics = pure_ovr_metrics
    hybrid_logistic_metrics = {metric: float("nan") for metric in BASE_METRICS}
    hybrid_svc_metrics = {metric: float("nan") for metric in BASE_METRICS}
    hybrid_sgd_metrics = {metric: float("nan") for metric in BASE_METRICS}
    if params.grid_approach in {"hybrid_logistic", "hybrid_svc", "hybrid_sgd"}:
        hybrid_eval_df = combine_hybrid_predictions(
            ovr_eval_df=ovr_eval_df,
            regex_eval_df=regex_eval_df,
            ovr_probability_df=ovr_probability_df,
            rule=hybrid_rule,
            label_confidence_threshold=params.hybrid_label_confidence_threshold,
        )
        if params.grid_approach == "hybrid_logistic":
            hybrid_logistic_metrics = _score_ukcat_multilabel(hybrid_eval_df)
        elif params.grid_approach == "hybrid_svc":
            hybrid_svc_metrics = _score_ukcat_multilabel(hybrid_eval_df)
        else:
            hybrid_sgd_metrics = _score_ukcat_multilabel(hybrid_eval_df)

    row = {
        "grid_approach": params.grid_approach,
        "fields_key": ",".join(params.fields),
        "clean_text": params.clean_text,
        "model_family": params.model_family,
        "threshold": params.threshold,
        "ngram_max": params.ngram_max,
        "char_ngram_max": params.char_ngram_max,
        "model_c": params.model_c,
        "sgd_loss": params.sgd_loss,
        "sgd_alpha": params.sgd_alpha,
        "class_weight_mode": params.class_weight_mode,
        "top_k_fallback": params.top_k_fallback,
        "hybrid_label_confidence_threshold": params.hybrid_label_confidence_threshold,
    }
    for approach, metrics in (
        ("regex", regex_metrics),
        ("ovr_logistic", ovr_logistic_metrics),
        ("ovr_svc", ovr_svc_metrics),
        ("ovr_sgd", ovr_sgd_metrics),
        ("hybrid_logistic", hybrid_logistic_metrics),
        ("hybrid_svc", hybrid_svc_metrics),
        ("hybrid_sgd", hybrid_sgd_metrics),
    ):
        for key, value in metrics.items():
            row[f"{approach}_{key}"] = value
    row["regex_weighted_primary"] = _weighted_primary(
        f1_micro=float(row["regex_f1_micro"]),
        f1_macro=float(row["regex_f1_macro"]),
        recall_micro=float(row["regex_recall_micro"]),
        precision_micro=float(row["regex_precision_micro"]),
    )
    row["ovr_logistic_weighted_primary"] = (
        _weighted_primary(
            f1_micro=float(row["ovr_logistic_f1_micro"]),
            f1_macro=float(row["ovr_logistic_f1_macro"]),
            recall_micro=float(row["ovr_logistic_recall_micro"]),
            precision_micro=float(row["ovr_logistic_precision_micro"]),
        )
        if params.grid_approach == "ovr_logistic"
        else float("nan")
    )
    row["ovr_svc_weighted_primary"] = (
        _weighted_primary(
            f1_micro=float(row["ovr_svc_f1_micro"]),
            f1_macro=float(row["ovr_svc_f1_macro"]),
            recall_micro=float(row["ovr_svc_recall_micro"]),
            precision_micro=float(row["ovr_svc_precision_micro"]),
        )
        if params.grid_approach == "ovr_svc"
        else float("nan")
    )
    row["ovr_sgd_weighted_primary"] = (
        _weighted_primary(
            f1_micro=float(row["ovr_sgd_f1_micro"]),
            f1_macro=float(row["ovr_sgd_f1_macro"]),
            recall_micro=float(row["ovr_sgd_recall_micro"]),
            precision_micro=float(row["ovr_sgd_precision_micro"]),
        )
        if params.grid_approach == "ovr_sgd"
        else float("nan")
    )
    row["hybrid_logistic_weighted_primary"] = (
        _weighted_primary(
            f1_micro=float(row["hybrid_logistic_f1_micro"]),
            f1_macro=float(row["hybrid_logistic_f1_macro"]),
            recall_micro=float(row["hybrid_logistic_recall_micro"]),
            precision_micro=float(row["hybrid_logistic_precision_micro"]),
        )
        if params.grid_approach == "hybrid_logistic"
        else float("nan")
    )
    row["hybrid_svc_weighted_primary"] = (
        _weighted_primary(
            f1_micro=float(row["hybrid_svc_f1_micro"]),
            f1_macro=float(row["hybrid_svc_f1_macro"]),
            recall_micro=float(row["hybrid_svc_recall_micro"]),
            precision_micro=float(row["hybrid_svc_precision_micro"]),
        )
        if params.grid_approach == "hybrid_svc"
        else float("nan")
    )
    row["hybrid_sgd_weighted_primary"] = (
        _weighted_primary(
            f1_micro=float(row["hybrid_sgd_f1_micro"]),
            f1_macro=float(row["hybrid_sgd_f1_macro"]),
            recall_micro=float(row["hybrid_sgd_recall_micro"]),
            precision_micro=float(row["hybrid_sgd_precision_micro"]),
        )
        if params.grid_approach == "hybrid_sgd"
        else float("nan")
    )

    return row


def _build_grid(
    *,
    approach: str,
    field_sets: Sequence[FieldSet],
    clean_values: Sequence[bool],
    model_family: str,
    thresholds: Sequence[float],
    ngram_max_values: Sequence[int],
    char_ngram_max_values: Sequence[int],
    model_c_values: Sequence[float],
    sgd_loss_values: Sequence[str],
    sgd_alpha_values: Sequence[float],
    class_weight_modes: Sequence[str],
    top_k_values: Sequence[int],
    hybrid_label_conf_values: Sequence[float],
) -> list[GridParams]:
    return [
        GridParams(
            grid_approach=approach,
            fields=field_set,
            clean_text=clean_text,
            model_family=model_family,
            threshold=t,
            ngram_max=n,
            char_ngram_max=char_n,
            model_c=c,
            sgd_loss=sgd_loss,
            sgd_alpha=sgd_alpha,
            class_weight_mode=class_weight_mode,
            top_k_fallback=k,
            hybrid_label_confidence_threshold=h,
        )
        for field_set, clean_text, t, n, char_n, c, sgd_loss, sgd_alpha, class_weight_mode, k, h in product(
            field_sets,
            clean_values,
            thresholds,
            ngram_max_values,
            char_ngram_max_values,
            model_c_values,
            sgd_loss_values,
            sgd_alpha_values,
            class_weight_modes,
            top_k_values,
            hybrid_label_conf_values,
        )
    ]


def _grid(
    params: argparse.Namespace,
    field_sets: Sequence[FieldSet],
    selected_approaches: Sequence[str],
) -> list[GridParams]:
    clean_values = _clean_values(params.clean_text_mode)
    grid_params: list[GridParams] = []
    if "ovr_logistic" in selected_approaches:
        grid_params.extend(
            _build_grid(
                approach="ovr_logistic",
                field_sets=field_sets,
                clean_values=clean_values,
                model_family="logistic",
                thresholds=[params.ovr_logistic_threshold]
                if params.ovr_logistic_threshold is not None
                else _parse_float_csv(params.ovr_logistic_thresholds),
                ngram_max_values=[params.ovr_logistic_ngram_max]
                if params.ovr_logistic_ngram_max is not None
                else _parse_int_csv(params.ovr_logistic_ngram_max_values),
                char_ngram_max_values=[params.ovr_logistic_char_ngram_max]
                if params.ovr_logistic_char_ngram_max is not None
                else _parse_int_csv(params.ovr_logistic_char_ngram_max_values),
                model_c_values=[params.ovr_logistic_c]
                if params.ovr_logistic_c is not None
                else _parse_float_csv(params.ovr_logistic_c_values),
                sgd_loss_values=[""],
                sgd_alpha_values=[-1.0],
                class_weight_modes=[params.ovr_class_weight_mode]
                if params.ovr_class_weight_mode is not None
                else _parse_csv_list(params.ovr_class_weight_modes),
                top_k_values=[params.ovr_top_k_fallback]
                if params.ovr_top_k_fallback is not None
                else _parse_int_csv(params.ovr_top_k_fallback_values),
                hybrid_label_conf_values=[-1.0],
            )
        )
    if "ovr_svc" in selected_approaches:
        grid_params.extend(
            _build_grid(
                approach="ovr_svc",
                field_sets=field_sets,
                clean_values=clean_values,
                model_family="linear_svc",
                thresholds=[params.ovr_svc_threshold]
                if params.ovr_svc_threshold is not None
                else _parse_float_csv(params.ovr_svc_thresholds),
                ngram_max_values=[params.ovr_svc_ngram_max]
                if params.ovr_svc_ngram_max is not None
                else _parse_int_csv(params.ovr_svc_ngram_max_values),
                char_ngram_max_values=[params.ovr_svc_char_ngram_max]
                if params.ovr_svc_char_ngram_max is not None
                else _parse_int_csv(params.ovr_svc_char_ngram_max_values),
                model_c_values=[params.ovr_svc_c]
                if params.ovr_svc_c is not None
                else _parse_float_csv(params.ovr_svc_c_values),
                sgd_loss_values=[""],
                sgd_alpha_values=[-1.0],
                class_weight_modes=[params.ovr_svc_class_weight_mode]
                if params.ovr_svc_class_weight_mode is not None
                else _parse_csv_list(params.ovr_svc_class_weight_modes),
                top_k_values=[params.ovr_svc_top_k_fallback]
                if params.ovr_svc_top_k_fallback is not None
                else _parse_int_csv(params.ovr_svc_top_k_fallback_values),
                hybrid_label_conf_values=[-1.0],
            )
        )
    if "ovr_sgd" in selected_approaches:
        grid_params.extend(
            _build_grid(
                approach="ovr_sgd",
                field_sets=field_sets,
                clean_values=clean_values,
                model_family="sgd",
                thresholds=[params.ovr_sgd_threshold]
                if params.ovr_sgd_threshold is not None
                else _parse_float_csv(params.ovr_sgd_thresholds),
                ngram_max_values=[params.ovr_sgd_ngram_max]
                if params.ovr_sgd_ngram_max is not None
                else _parse_int_csv(params.ovr_sgd_ngram_max_values),
                char_ngram_max_values=[params.ovr_sgd_char_ngram_max]
                if params.ovr_sgd_char_ngram_max is not None
                else _parse_int_csv(params.ovr_sgd_char_ngram_max_values),
                model_c_values=[-1.0],
                sgd_loss_values=[params.ovr_sgd_loss]
                if params.ovr_sgd_loss is not None
                else _parse_csv_list(params.ovr_sgd_losses),
                sgd_alpha_values=[params.ovr_sgd_alpha]
                if params.ovr_sgd_alpha is not None
                else _parse_float_csv(params.ovr_sgd_alpha_values),
                class_weight_modes=[params.ovr_sgd_class_weight_mode]
                if params.ovr_sgd_class_weight_mode is not None
                else _parse_csv_list(params.ovr_sgd_class_weight_modes),
                top_k_values=[params.ovr_sgd_top_k_fallback]
                if params.ovr_sgd_top_k_fallback is not None
                else _parse_int_csv(params.ovr_sgd_top_k_fallback_values),
                hybrid_label_conf_values=[-1.0],
            )
        )
    if "hybrid_logistic" in selected_approaches:
        grid_params.extend(
            _build_grid(
                approach="hybrid_logistic",
                field_sets=field_sets,
                clean_values=clean_values,
                model_family="logistic",
                thresholds=[params.hybrid_logistic_threshold]
                if params.hybrid_logistic_threshold is not None
                else _parse_float_csv(params.hybrid_logistic_thresholds),
                ngram_max_values=[params.hybrid_logistic_ngram_max]
                if params.hybrid_logistic_ngram_max is not None
                else _parse_int_csv(params.hybrid_logistic_ngram_max_values),
                char_ngram_max_values=[params.hybrid_logistic_char_ngram_max]
                if params.hybrid_logistic_char_ngram_max is not None
                else _parse_int_csv(params.hybrid_logistic_char_ngram_max_values),
                model_c_values=[params.hybrid_logistic_c]
                if params.hybrid_logistic_c is not None
                else _parse_float_csv(params.hybrid_logistic_c_values),
                sgd_loss_values=[""],
                sgd_alpha_values=[-1.0],
                class_weight_modes=[params.hybrid_logistic_class_weight_mode]
                if params.hybrid_logistic_class_weight_mode is not None
                else _parse_csv_list(params.hybrid_logistic_class_weight_modes),
                top_k_values=[params.hybrid_logistic_top_k_fallback]
                if params.hybrid_logistic_top_k_fallback is not None
                else _parse_int_csv(params.hybrid_logistic_top_k_fallback_values),
                hybrid_label_conf_values=[params.hybrid_logistic_label_confidence_threshold]
                if params.hybrid_logistic_label_confidence_threshold is not None
                else _parse_float_csv(params.hybrid_logistic_label_confidence_threshold_values),
            )
        )
    if "hybrid_svc" in selected_approaches:
        grid_params.extend(
            _build_grid(
                approach="hybrid_svc",
                field_sets=field_sets,
                clean_values=clean_values,
                model_family="linear_svc",
                thresholds=[params.hybrid_svc_threshold]
                if params.hybrid_svc_threshold is not None
                else _parse_float_csv(params.hybrid_svc_thresholds),
                ngram_max_values=[params.hybrid_svc_ngram_max]
                if params.hybrid_svc_ngram_max is not None
                else _parse_int_csv(params.hybrid_svc_ngram_max_values),
                char_ngram_max_values=[params.hybrid_svc_char_ngram_max]
                if params.hybrid_svc_char_ngram_max is not None
                else _parse_int_csv(params.hybrid_svc_char_ngram_max_values),
                model_c_values=[params.hybrid_svc_c]
                if params.hybrid_svc_c is not None
                else _parse_float_csv(params.hybrid_svc_c_values),
                sgd_loss_values=[""],
                sgd_alpha_values=[-1.0],
                class_weight_modes=[params.hybrid_svc_class_weight_mode]
                if params.hybrid_svc_class_weight_mode is not None
                else _parse_csv_list(params.hybrid_svc_class_weight_modes),
                top_k_values=[params.hybrid_svc_top_k_fallback]
                if params.hybrid_svc_top_k_fallback is not None
                else _parse_int_csv(params.hybrid_svc_top_k_fallback_values),
                hybrid_label_conf_values=[params.hybrid_svc_label_confidence_threshold]
                if params.hybrid_svc_label_confidence_threshold is not None
                else _parse_float_csv(params.hybrid_svc_label_confidence_threshold_values),
            )
        )
    if "hybrid_sgd" in selected_approaches:
        grid_params.extend(
            _build_grid(
                approach="hybrid_sgd",
                field_sets=field_sets,
                clean_values=clean_values,
                model_family="sgd",
                thresholds=[params.hybrid_sgd_threshold]
                if params.hybrid_sgd_threshold is not None
                else _parse_float_csv(params.hybrid_sgd_thresholds),
                ngram_max_values=[params.hybrid_sgd_ngram_max]
                if params.hybrid_sgd_ngram_max is not None
                else _parse_int_csv(params.hybrid_sgd_ngram_max_values),
                char_ngram_max_values=[params.hybrid_sgd_char_ngram_max]
                if params.hybrid_sgd_char_ngram_max is not None
                else _parse_int_csv(params.hybrid_sgd_char_ngram_max_values),
                model_c_values=[-1.0],
                sgd_loss_values=[params.hybrid_sgd_loss]
                if params.hybrid_sgd_loss is not None
                else _parse_csv_list(params.hybrid_sgd_losses),
                sgd_alpha_values=[params.hybrid_sgd_alpha]
                if params.hybrid_sgd_alpha is not None
                else _parse_float_csv(params.hybrid_sgd_alpha_values),
                class_weight_modes=[params.hybrid_sgd_class_weight_mode]
                if params.hybrid_sgd_class_weight_mode is not None
                else _parse_csv_list(params.hybrid_sgd_class_weight_modes),
                top_k_values=[params.hybrid_sgd_top_k_fallback]
                if params.hybrid_sgd_top_k_fallback is not None
                else _parse_int_csv(params.hybrid_sgd_top_k_fallback_values),
                hybrid_label_conf_values=[params.hybrid_sgd_label_confidence_threshold]
                if params.hybrid_sgd_label_confidence_threshold is not None
                else _parse_float_csv(params.hybrid_sgd_label_confidence_threshold_values),
            )
        )
    return grid_params


def _validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.test_size < 1:
        raise SystemExit("--test-size must be between 0 and 1")
    if args.n_jobs < 1:
        raise SystemExit("--n-jobs must be at least 1")
    if args.show_top < 0:
        raise SystemExit("--show-top must be 0 or greater")
    if getattr(args, "optimise_metric", None) not in OPTIMISATION_METRICS:
        raise SystemExit(f"Invalid optimise metric: {getattr(args, 'optimise_metric', None)}")
    if args.clean_text_mode not in {"off", "on", "compare"}:
        raise SystemExit("--clean-text-mode must be one of: off, on, compare")
    for label, single_value, values_text in (
        ("ovr-logistic", args.ovr_logistic_c, args.ovr_logistic_c_values),
        ("ovr-svc", args.ovr_svc_c, args.ovr_svc_c_values),
        ("hybrid-logistic", args.hybrid_logistic_c, args.hybrid_logistic_c_values),
        ("hybrid-svc", args.hybrid_svc_c, args.hybrid_svc_c_values),
    ):
        if single_value is not None and single_value <= 0:
            raise SystemExit(f"--{label}-c must be greater than 0")
        values = [single_value] if single_value is not None else _parse_float_csv(values_text)
        if any(value <= 0 for value in values):
            raise SystemExit(f"All --{label}-c-values entries must be greater than 0")
    for label, single_mode, values_text in (
        ("ovr-logistic", args.ovr_class_weight_mode, args.ovr_class_weight_modes),
        ("ovr-svc", args.ovr_svc_class_weight_mode, args.ovr_svc_class_weight_modes),
        ("ovr-sgd", args.ovr_sgd_class_weight_mode, args.ovr_sgd_class_weight_modes),
        (
            "hybrid-logistic",
            args.hybrid_logistic_class_weight_mode,
            args.hybrid_logistic_class_weight_modes,
        ),
        ("hybrid-svc", args.hybrid_svc_class_weight_mode, args.hybrid_svc_class_weight_modes),
        ("hybrid-sgd", args.hybrid_sgd_class_weight_mode, args.hybrid_sgd_class_weight_modes),
    ):
        modes = [single_mode] if single_mode is not None else _parse_csv_list(values_text)
        if any(mode not in {"none", "balanced"} for mode in modes):
            raise SystemExit(f"{label.upper()} class-weight modes must be drawn from: none, balanced")
    for label, single_value, values_text in (
        ("ovr-logistic", args.ovr_logistic_char_ngram_max, args.ovr_logistic_char_ngram_max_values),
        ("ovr-svc", args.ovr_svc_char_ngram_max, args.ovr_svc_char_ngram_max_values),
        ("ovr-sgd", args.ovr_sgd_char_ngram_max, args.ovr_sgd_char_ngram_max_values),
        ("hybrid-logistic", args.hybrid_logistic_char_ngram_max, args.hybrid_logistic_char_ngram_max_values),
        ("hybrid-svc", args.hybrid_svc_char_ngram_max, args.hybrid_svc_char_ngram_max_values),
        ("hybrid-sgd", args.hybrid_sgd_char_ngram_max, args.hybrid_sgd_char_ngram_max_values),
    ):
        values = [single_value] if single_value is not None else _parse_int_csv(values_text)
        if any(value not in {0} and value < 3 for value in values):
            raise SystemExit(f"All --{label}-char-ngram-max-values entries must be 0 or at least 3")
    for label, single_loss, values_text in (
        ("ovr-sgd", args.ovr_sgd_loss, args.ovr_sgd_losses),
        ("hybrid-sgd", args.hybrid_sgd_loss, args.hybrid_sgd_losses),
    ):
        losses = [single_loss] if single_loss is not None else _parse_csv_list(values_text)
        if any(loss not in SGD_LOSS_CHOICES for loss in losses):
            raise SystemExit(f"{label.upper()} losses must be drawn from: {', '.join(SGD_LOSS_CHOICES)}")
    for label, single_value, values_text in (
        ("ovr-sgd", args.ovr_sgd_alpha, args.ovr_sgd_alpha_values),
        ("hybrid-sgd", args.hybrid_sgd_alpha, args.hybrid_sgd_alpha_values),
    ):
        if single_value is not None and single_value <= 0:
            raise SystemExit(f"--{label}-alpha must be greater than 0")
        values = [single_value] if single_value is not None else _parse_float_csv(values_text)
        if any(value <= 0 for value in values):
            raise SystemExit(f"All --{label}-alpha-values entries must be greater than 0")
    if args.optimise_metric == "weighted_primary":
        _validate_weighted_primary_weights()


def _clean_values(mode: str) -> list[bool]:
    if mode == "off":
        return [False]
    if mode == "on":
        return [True]
    return [False, True]


def _resolve_inputs(
    args: argparse.Namespace,
    selected_approaches: Sequence[str],
) -> tuple[tuple[str, ...], list[int], list[GridParams]]:
    sample_files = tuple(args.sample_files) if args.sample_files else DEFAULT_SAMPLE_FILES
    random_states = [args.random_state] if args.random_state is not None else _parse_int_csv(args.random_states)
    field_sets = _parse_field_sets(args.fields)
    if any(len(field_set) == 0 for field_set in field_sets):
        raise SystemExit("Field sets must include at least one field")
    grid_params = _grid(args, field_sets=field_sets, selected_approaches=selected_approaches)
    if not grid_params:
        raise SystemExit("No parameter combinations generated")
    return sample_files, random_states, grid_params


def _print_comparison_metrics(row: pd.Series) -> None:
    print("\nComparison metrics")
    parts = [
        f"{approach}={int(round(float(row[f'{approach}_rows_mean'])))}"
        for approach in APPROACHES
        if f"{approach}_rows_mean" in row.index
    ]
    print(" - rows: " + " | ".join(parts))
    for metric in DISPLAY_METRICS:
        parts = [
            f"{approach}={_fmt(float(row[f'{approach}_{metric}_mean']))}"
            for approach in APPROACHES
            if f"{approach}_{metric}_mean" in row.index
        ]
        print(f" - {metric}: " + " | ".join(parts))


def _print_single_approach_metrics(row: pd.Series, approach: str) -> None:
    print("\nComparison metrics")
    if DEFAULT_ENABLE_REGEX:
        print(
            f" - rows: regex={int(round(float(row['regex_rows_mean'])))} | "
            f"{approach}={int(round(float(row[f'{approach}_rows_mean'])))}"
        )
    else:
        print(f" - rows: {approach}={int(round(float(row[f'{approach}_rows_mean'])))}")
    for metric in DISPLAY_METRICS:
        if DEFAULT_ENABLE_REGEX:
            print(
                f" - {metric}: "
                f"regex={_fmt(float(row[f'regex_{metric}_mean']))} | "
                f"{approach}={_fmt(float(row[f'{approach}_{metric}_mean']))}"
            )
        else:
            print(f" - {metric}: {approach}={_fmt(float(row[f'{approach}_{metric}_mean']))}")


def _print_top_rows(
    summary_df: pd.DataFrame,
    show_top: int,
    approach: str,
    hybrid_conf_precision: int,
    dedupe_cols: Sequence[str] | None = None,
) -> None:
    rows_df = summary_df
    if dedupe_cols:
        rows_df = rows_df.drop_duplicates(subset=list(dedupe_cols), keep="first").reset_index(drop=True)

    for rank, (_, row) in enumerate(rows_df.head(show_top).iterrows(), start=1):
        base = (
            f" - [{rank}] fields={row['fields_key']}, "
            f"clean_text={'on' if bool(row['clean_text']) else 'off'}, "
            f"model_family={row['model_family']}, "
            f"threshold={row['threshold']:.3f}, "
            f"ngram_max={int(row['ngram_max'])}, "
            f"char_ngram_max={int(row['char_ngram_max'])}, "
            f"class_weight_mode={row['class_weight_mode']}, "
            f"top_k_fallback={int(row['top_k_fallback'])}"
        )
        if row["model_family"] == "sgd":
            base += f", sgd_loss={row['sgd_loss']}, sgd_alpha={float(row['sgd_alpha']):g}"
        else:
            base += f", model_c={float(row['model_c']):g}"
        if approach.startswith("hybrid"):
            base += (
                ", hybrid_label_confidence_threshold="
                f"{_fmt_hybrid_conf(float(row['hybrid_label_confidence_threshold']), hybrid_conf_precision)}"
            )
        print(base)


def _print_best_params(
    best: pd.Series,
    args: argparse.Namespace,
    hybrid_conf_precision: int,
    include_hybrid: bool,
    show_selected_by: str | None = None,
) -> None:
    if show_selected_by is not None:
        print(f" - selected by: {show_selected_by}")
    print(f" - fields: {best['fields_key']}")
    print(f" - clean_text: {'on' if bool(best['clean_text']) else 'off'}")
    print(f" - model_family: {best['model_family']}")
    print(f" - threshold: {best['threshold']:.3f}")
    print(f" - ngram_max: {int(best['ngram_max'])}")
    print(f" - char_ngram_max: {int(best['char_ngram_max'])}")
    if best["model_family"] == "sgd":
        print(f" - sgd_loss: {best['sgd_loss']}")
        print(f" - sgd_alpha: {float(best['sgd_alpha']):g}")
    else:
        print(f" - model_c: {float(best['model_c']):g}")
    print(f" - class_weight_mode: {best['class_weight_mode']}")
    print(f" - top_k_fallback: {int(best['top_k_fallback'])}")
    if include_hybrid:
        print(f" - hybrid_rule: {args.hybrid_rule}")
        print(
            " - hybrid_label_confidence_threshold: "
            f"{_fmt_hybrid_conf(float(best['hybrid_label_confidence_threshold']), hybrid_conf_precision)}"
        )


def _print_run_header(
    sample_files: Sequence[str],
    labelled_rows: int,
    random_states: Sequence[int],
    grid_count: int,
    optimise_metric: str,
    clean_mode: str,
) -> None:
    print(f"Loading labelled data from: {', '.join(sample_files)}")
    print(f"Loaded labelled rows: {labelled_rows:,}")
    print(f"Random state(s): {list(random_states)}")
    print(f"Grid combinations: {grid_count}")
    print(f"Total grid evaluations: {grid_count * len(random_states)}")
    if optimise_metric == "weighted_primary":
        print(
            "Weighted objective in use: "
            f"{WEIGHTED_PRIMARY_F1_MICRO:.3f}*F1_micro + "
            f"{WEIGHTED_PRIMARY_F1_MACRO:.3f}*F1_macro + "
            f"{WEIGHTED_PRIMARY_RECALL_MICRO:.3f}*Recall_micro + "
            f"{WEIGHTED_PRIMARY_PRECISION_MICRO:.3f}*Precision_micro"
        )
    print(f"Clean text mode: {clean_mode}")
    print("")


def _get_cached_ovr_artifacts(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: GridParams,
    args: argparse.Namespace,
    ovr_cache: dict[tuple, tuple[pd.DataFrame, pd.Series]],
) -> tuple[pd.DataFrame, pd.Series]:
    cache_key = (
        params.model_family,
        params.ngram_max,
        params.char_ngram_max,
        params.clean_text,
        params.model_c,
        params.sgd_loss,
        params.sgd_alpha,
        params.class_weight_mode,
        tuple(params.fields),
        int(args.n_jobs),
    )
    if cache_key not in ovr_cache:
        # Only the feature/training knobs require a fresh OvR fit. Thresholds and
        # hybrid gating are cheap when using cached probabilities
        seed_eval_df, seed_prob_df = _evaluate_ukcat_ovr_rows_with_scores(
            train_df=train_df,
            test_df=test_df,
            fields=params.fields,
            threshold=0.5,
            top_k_fallback=0,
            n_jobs=args.n_jobs,
            ngram_max=params.ngram_max,
            char_ngram_max=params.char_ngram_max,
            clean_text=params.clean_text,
            model_family=params.model_family,
            model_c=params.model_c,
            class_weight_mode=params.class_weight_mode,
            sgd_loss=params.sgd_loss or None,
            sgd_alpha=None if params.sgd_alpha < 0 else params.sgd_alpha,
        )
        true_codes_by_org = seed_eval_df.set_index("org_id")["true_codes"].copy()
        true_codes_by_org.index = true_codes_by_org.index.astype(str)
        ovr_cache[cache_key] = (seed_prob_df, true_codes_by_org)
    return ovr_cache[cache_key]


def _run_grid_for_states(
    labelled: pd.DataFrame,
    random_states: Sequence[int],
    grid_params: Sequence[GridParams],
    args: argparse.Namespace,
) -> pd.DataFrame:
    per_state_rows = []
    hybrid_conf_precision = _resolve_hybrid_conf_precision(args)
    for state in random_states:
        print(f"[state={state}] creating shared split")
        train_df, test_df = train_test_split(
            labelled,
            test_size=args.test_size,
            random_state=state,
            shuffle=True,
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        # Regex is the fixed baseline for the split. Every OvR and hybrid run is
        # compared against these same regex predictions on the same test rows
        regex_eval_df = _evaluate_ukcat_regex_rows(test_df=test_df, include_groups=args.include_groups)
        regex_metrics = _score_ukcat_multilabel(regex_eval_df)

        # Conservative cache: reuse only artefacts affected by model training.
        ovr_cache: dict[tuple, tuple[pd.DataFrame, pd.Series]] = {}
        param_iterator = enumerate(grid_params, start=1)
        if not getattr(args, "verbose", False):
            param_iterator = tqdm(
                param_iterator,
                total=len(grid_params),
                desc=f"state={state}",
                leave=False,
            )
        for idx, gp in param_iterator:
            if getattr(args, "verbose", False):
                base = (
                    f"  [{idx}/{len(grid_params)}] fields={','.join(gp.fields)}, "
                    f"clean_text={'on' if gp.clean_text else 'off'}, threshold={gp.threshold:.3f}, "
                    f"model_family={gp.model_family}, "
                    f"ngram_max={gp.ngram_max}, char_ngram_max={gp.char_ngram_max}, "
                    f"class_weight_mode={gp.class_weight_mode}, top_k_fallback={gp.top_k_fallback}"
                )
                if gp.model_family == "sgd":
                    base += f", sgd_loss={gp.sgd_loss}, sgd_alpha={gp.sgd_alpha:g}"
                else:
                    base += f", model_c={gp.model_c:g}"
                if gp.grid_approach.startswith("hybrid"):
                    base += (
                        f", hybrid_label_confidence_threshold="
                        f"{_fmt_hybrid_conf(gp.hybrid_label_confidence_threshold, hybrid_conf_precision)}"
                    )
                print(base)
            ovr_probability_df, true_codes_by_org = _get_cached_ovr_artifacts(
                train_df=train_df,
                test_df=test_df,
                params=gp,
                args=args,
                ovr_cache=ovr_cache,
            )
            result = _score_pair(
                regex_eval_df=regex_eval_df,
                regex_metrics=regex_metrics,
                ovr_probability_df=ovr_probability_df,
                true_codes_by_org=true_codes_by_org,
                params=gp,
                hybrid_rule=args.hybrid_rule,
            )
            result["random_state"] = state
            per_state_rows.append(result)
    return pd.DataFrame(per_state_rows)


def _candidate_column(metric: str, approach: str) -> str:
    return f"{approach}_{metric}_mean"


def _apply_guardrails(summary_df: pd.DataFrame, approach: str) -> pd.DataFrame:
    filtered_df = summary_df
    for metric, operator, threshold in GUARDRAIL_SPECS:
        column = _candidate_column(metric, approach)
        if operator == ">=":
            filtered_df = filtered_df[filtered_df[column] >= threshold].copy()
        elif operator == "<":
            filtered_df = filtered_df[filtered_df[column] < threshold].copy()
        else:
            raise SystemExit(f"Unsupported guardrail operator: {operator}")
    return filtered_df


def _guardrail_message() -> str:
    parts = [
        f"candidate_{metric} {operator} {threshold}"
        for metric, operator, threshold in GUARDRAIL_SPECS
    ]
    return "No parameter combinations left after applying guardrails (" + ", ".join(parts) + ")."


def _select_ranked_summary(summary_df: pd.DataFrame, optimise_metric: str, approach: str) -> tuple[pd.DataFrame, str]:
    summary_df = summary_df[summary_df["grid_approach"] == approach].copy()
    if summary_df.empty:
        raise SystemExit(f"No {approach.upper()} parameter combinations were generated.")
    if optimise_metric == "weighted_primary":
        # Weighted objective uses top-of-file coefficients.
        weighted_col = _candidate_column("weighted_primary", approach)
        summary_df.loc[:, weighted_col] = (
            WEIGHTED_PRIMARY_F1_MICRO * summary_df[_candidate_column("f1_micro", approach)]
            + WEIGHTED_PRIMARY_F1_MACRO * summary_df[_candidate_column("f1_macro", approach)]
            + WEIGHTED_PRIMARY_RECALL_MICRO * summary_df[_candidate_column("recall_micro", approach)]
            + WEIGHTED_PRIMARY_PRECISION_MICRO * summary_df[_candidate_column("precision_micro", approach)]
        )

    sort_by = _candidate_column(optimise_metric, approach)
    if sort_by not in summary_df.columns:
        raise SystemExit(f"Ranking column not found: {sort_by}")

    # Apply guardrails before ranking so selected candidates meet minimum quality thresholds.
    summary_df = _apply_guardrails(summary_df=summary_df, approach=approach)
    if summary_df.empty:
        raise SystemExit(_guardrail_message())

    ascending = optimise_metric == "hamming_loss"
    return summary_df.sort_values(sort_by, ascending=ascending).reset_index(drop=True), sort_by


def _print_results(
    selected_best: dict[str, pd.Series],
    selected_summaries: dict[str, pd.DataFrame],
    selected_sort_bys: dict[str, str],
    args: argparse.Namespace,
    random_states: Sequence[int],
) -> None:
    hybrid_conf_precision = _resolve_hybrid_conf_precision(args)
    merged_best = (
        selected_best["ovr_logistic"].copy()
        if "ovr_logistic" in selected_best
        else next(iter(selected_best.values())).copy()
    )
    for approach, best in selected_best.items():
        if approach == "ovr_logistic":
            continue
        for metric in DISPLAY_METRICS + ("rows",):
            merged_best[f"{approach}_{metric}_mean"] = best[f"{approach}_{metric}_mean"]
    _print_comparison_metrics(merged_best)
    print(
        f"\nNote: Best model combinations were selected independently "
        f"(each ranked by `{args.optimise_metric}`) and averaged across {len(random_states)} random state(s)."
    )
    for approach in ("ovr_logistic", "ovr_svc", "ovr_sgd", "hybrid_logistic", "hybrid_svc", "hybrid_sgd"):
        if approach not in selected_best:
            continue
        print(f"Best {approach.upper()} parameter combination")
        _print_best_params(
            best=selected_best[approach],
            args=args,
            hybrid_conf_precision=hybrid_conf_precision,
            include_hybrid=approach.startswith("hybrid"),
            show_selected_by=selected_sort_bys[approach],
        )

    if args.show_top == 0:
        return

    for approach in ("ovr_logistic", "ovr_svc", "ovr_sgd", "hybrid_logistic", "hybrid_svc", "hybrid_sgd"):
        if approach not in selected_summaries:
            continue
        print(f"\nTop {args.show_top} {approach.upper()} parameter combinations")
        print(f" (ranked by {selected_sort_bys[approach]})")
        _print_top_rows(
            summary_df=selected_summaries[approach],
            show_top=args.show_top,
            approach=approach,
            hybrid_conf_precision=hybrid_conf_precision,
            dedupe_cols=(
                "clean_text",
                "model_family",
                "threshold",
                "ngram_max",
                "char_ngram_max",
                "model_c",
                "sgd_loss",
                "sgd_alpha",
                "class_weight_mode",
                "top_k_fallback",
            ) if not approach.startswith("hybrid") else None,
        )


def _print_single_results(
    best: pd.Series,
    summary_df: pd.DataFrame,
    approach: str,
    sort_by: str,
    args: argparse.Namespace,
    random_states: Sequence[int],
) -> None:
    hybrid_conf_precision = _resolve_hybrid_conf_precision(args)
    _print_single_approach_metrics(best, approach)
    print(
        f"\nNote: The best parameter combination was selected for "
        f"{approach.upper()} "
        f"(selected by `{sort_by}`) and averaged across {len(random_states)} random states."
    )
    print("Best parameter combination")
    _print_best_params(
        best=best,
        args=args,
        hybrid_conf_precision=hybrid_conf_precision,
        include_hybrid=approach.startswith("hybrid"),
    )

    if args.show_top == 0:
        return

    print(f"\nTop {args.show_top} parameter combinations")
    print(f" (ranked by {sort_by})")
    _print_top_rows(
        summary_df=summary_df,
        show_top=args.show_top,
        approach=approach,
        hybrid_conf_precision=hybrid_conf_precision,
    )


def _save_optional_outputs(args: argparse.Namespace, per_state_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    if args.save_per_state:
        out_path = Path(args.save_per_state)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        per_state_df.to_csv(out_path, index=False)
        print(f"\nSaved per-state results to {out_path}")

    if args.save_summary:
        out_path = Path(args.save_summary)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_path, index=False)
        print(f"Saved summary results to {out_path}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    selected_approaches = _resolve_selected_approaches_from_knobs()
    args.optimise_metric = _resolve_optimise_metric_from_knobs()
    _validate_args(args)
    sample_files, random_states, grid_params = _resolve_inputs(args, selected_approaches=selected_approaches)
    labelled = _load_labelled_ukcat(sample_files)
    _print_run_header(
        sample_files=sample_files,
        labelled_rows=len(labelled),
        random_states=random_states,
        grid_count=len(grid_params),
        optimise_metric=args.optimise_metric,
        clean_mode=args.clean_text_mode,
    )

    per_state_df = _run_grid_for_states(labelled=labelled, random_states=random_states, grid_params=grid_params, args=args)
    summary_df = _aggregate_results(per_state_df)
    if len(selected_approaches) > 1:
        selected_summaries: dict[str, pd.DataFrame] = {}
        selected_sort_bys: dict[str, str] = {}
        selected_best: dict[str, pd.Series] = {}
        for approach in selected_approaches:
            summary, sort_by = _select_ranked_summary(
                summary_df=summary_df,
                optimise_metric=args.optimise_metric,
                approach=approach,
            )
            selected_summaries[approach] = summary
            selected_sort_bys[approach] = sort_by
            selected_best[approach] = summary.iloc[0]
        _print_results(
            selected_best=selected_best,
            selected_summaries=selected_summaries,
            selected_sort_bys=selected_sort_bys,
            args=args,
            random_states=random_states,
        )
    else:
        approach = selected_approaches[0]
        selected_summary_df, sort_by = _select_ranked_summary(
            summary_df=summary_df,
            optimise_metric=args.optimise_metric,
            approach=approach,
        )
        best = selected_summary_df.iloc[0]
        _print_single_results(
            best=best,
            summary_df=selected_summary_df,
            approach=approach,
            sort_by=sort_by,
            args=args,
            random_states=random_states,
        )
    _save_optional_outputs(args, per_state_df, summary_df)


if __name__ == "__main__":
    main()
