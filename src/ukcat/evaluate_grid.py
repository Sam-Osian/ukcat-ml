from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

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
DEFAULT_THRESHOLDS = (0.04, 0.05, 0.06, 0.10, 0.2, 0.25, 0.3)
DEFAULT_NGRAM_MAX_VALUES = (1, 2, 3)
DEFAULT_TOP_K_FALLBACK_VALUES = (0, 1)
DEFAULT_CLEAN_TEXT_MODE = "on" # can be set to 'on', 'off' or 'compare'
DEFAULT_FIELD_SETS = (
    ("name", "activities"),
    #("name", "activities", "objects"), # Commented out as objects consistently harms performance
)
DEFAULT_HYBRID_LABEL_CONFIDENCE_THRESHOLDS = (0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.001)

DEFAULT_SELECT_BEST_OVR = True
DEFAULT_SELECT_BEST_HYBRID = True

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
WEIGHTED_PRIMARY_F1_MICRO = 0.40
WEIGHTED_PRIMARY_F1_MACRO = 0.40
WEIGHTED_PRIMARY_RECALL_MICRO = 0.20

# Optional uardrails (applied to the candidate approach being ranked: OVR and/or Hybrid)
# Setting to 0 essentially turns that guardrail 'off'
GUARDRAIL_MIN_PRECISION_MICRO = 0
GUARDRAIL_MIN_F1_MICRO = 0.40
GUARDRAIL_MIN_SUBSET_ACCURACY = 0.10
GUARDRAIL_MAX_HAMMING_LOSS_EXCLUSIVE = 0.0160
GUARDRAIL_MIN_JACCARD_SAMPLES = 0.0

DISPLAY_METRICS = OPTIMISATION_METRICS
APPROACHES = ("regex", "ovr", "hybrid")
BASE_METRICS = ("rows",) + OPTIMISATION_METRICS
GRID_GROUP_COLUMNS = (
    "fields_key",
    "clean_text",
    "threshold",
    "ngram_max",
    "top_k_fallback",
    "hybrid_label_confidence_threshold",
)
AGGREGATED_METRIC_COLUMNS = tuple(
    f"{approach}_{metric}"
    for approach in APPROACHES
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
    fields: tuple[str, ...]
    clean_text: bool
    threshold: float
    ngram_max: int
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
        "--thresholds",
        default=",".join(_parse_csv_list(DEFAULT_THRESHOLDS)),
        help="Comma-separated OvR thresholds to test.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Single threshold override (use instead of --thresholds).",
    )
    parser.add_argument(
        "--ngram-max-values",
        default=",".join(_parse_csv_list(DEFAULT_NGRAM_MAX_VALUES)),
        help="Comma-separated maximum n-gram sizes (uses 1..N).",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=None,
        help="Single n-gram max override (use instead of --ngram-max-values).",
    )
    parser.add_argument(
        "--top-k-fallback-values",
        default=",".join(_parse_csv_list(DEFAULT_TOP_K_FALLBACK_VALUES)),
        help="Comma-separated fallback values (0 disables fallback).",
    )
    parser.add_argument(
        "--top-k-fallback",
        type=int,
        default=None,
        help="Single top-k fallback override (use instead of --top-k-fallback-values).",
    )
    parser.add_argument(
        "--hybrid-label-confidence-threshold-values",
        default=",".join(_parse_csv_list(DEFAULT_HYBRID_LABEL_CONFIDENCE_THRESHOLDS)),
        help="Comma-separated label confidence thresholds for label_conf_gated_regex.",
    )
    parser.add_argument(
        "--hybrid-label-confidence-threshold",
        type=float,
        default=None,
        help="Single hybrid label confidence threshold override.",
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
        "--quiet",
        action="store_true",
        help="Suppress per-parameter progress lines during grid evaluation.",
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
        "ovr": DEFAULT_SELECT_BEST_OVR,
        "hybrid": DEFAULT_SELECT_BEST_HYBRID,
    }
    selected = [name for name, enabled in selected_knobs.items() if enabled]
    if not selected:
        raise SystemExit(
            "At least one model selection knob must be True: "
            "DEFAULT_SELECT_BEST_OVR and/or DEFAULT_SELECT_BEST_HYBRID."
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
    raw_values = getattr(args, "hybrid_label_confidence_threshold_values", "")
    if isinstance(raw_values, str) and raw_values.strip():
        precision = max(_decimal_places(part) for part in raw_values.split(",") if part.strip())
    single_value = getattr(args, "hybrid_label_confidence_threshold", None)
    if single_value is not None:
        precision = max(precision, _decimal_places(str(single_value)))
    return max(4, precision)


def _fmt_hybrid_conf(value: float, precision: int) -> str:
    return f"{value:.{precision}f}"


def _weighted_primary(f1_micro: float, f1_macro: float, recall_micro: float) -> float:
    return (
        WEIGHTED_PRIMARY_F1_MICRO * f1_micro
        + WEIGHTED_PRIMARY_F1_MACRO * f1_macro
        + WEIGHTED_PRIMARY_RECALL_MICRO * recall_micro
    )


def _validate_weighted_primary_weights() -> None:
    total = WEIGHTED_PRIMARY_F1_MICRO + WEIGHTED_PRIMARY_F1_MACRO + WEIGHTED_PRIMARY_RECALL_MICRO
    if min(WEIGHTED_PRIMARY_F1_MICRO, WEIGHTED_PRIMARY_F1_MACRO, WEIGHTED_PRIMARY_RECALL_MICRO) < 0:
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
    hybrid_eval_df = combine_hybrid_predictions(
        ovr_eval_df=ovr_eval_df,
        regex_eval_df=regex_eval_df,
        ovr_probability_df=ovr_probability_df,
        rule=hybrid_rule,
        label_confidence_threshold=params.hybrid_label_confidence_threshold,
    )
    pure_ovr_metrics = _score_ukcat_multilabel(ovr_eval_df)
    hybrid_metrics = _score_ukcat_multilabel(hybrid_eval_df)

    row = {
        "fields_key": ",".join(params.fields),
        "clean_text": params.clean_text,
        "threshold": params.threshold,
        "ngram_max": params.ngram_max,
        "top_k_fallback": params.top_k_fallback,
        "hybrid_label_confidence_threshold": params.hybrid_label_confidence_threshold,
    }
    for approach, metrics in (
        ("regex", regex_metrics),
        ("ovr", pure_ovr_metrics),
        ("hybrid", hybrid_metrics),
    ):
        for key, value in metrics.items():
            row[f"{approach}_{key}"] = value
    row["regex_weighted_primary"] = _weighted_primary(
        f1_micro=float(row["regex_f1_micro"]),
        f1_macro=float(row["regex_f1_macro"]),
        recall_micro=float(row["regex_recall_micro"]),
    )
    row["ovr_weighted_primary"] = _weighted_primary(
        f1_micro=float(row["ovr_f1_micro"]),
        f1_macro=float(row["ovr_f1_macro"]),
        recall_micro=float(row["ovr_recall_micro"]),
    )
    row["hybrid_weighted_primary"] = _weighted_primary(
        f1_micro=float(row["hybrid_f1_micro"]),
        f1_macro=float(row["hybrid_f1_macro"]),
        recall_micro=float(row["hybrid_recall_micro"]),
    )

    return row


def _grid(params: argparse.Namespace, field_sets: Sequence[FieldSet]) -> list[GridParams]:
    clean_values = _clean_values(params.clean_text_mode)
    thresholds = [params.threshold] if params.threshold is not None else _parse_float_csv(params.thresholds)
    ngram_max_values = [params.ngram_max] if params.ngram_max is not None else _parse_int_csv(params.ngram_max_values)
    top_k_values = (
        [params.top_k_fallback]
        if params.top_k_fallback is not None
        else _parse_int_csv(params.top_k_fallback_values)
    )
    hybrid_label_conf_values = (
        [params.hybrid_label_confidence_threshold]
        if params.hybrid_label_confidence_threshold is not None
        else _parse_float_csv(params.hybrid_label_confidence_threshold_values)
    )
    return [
        GridParams(
            fields=field_set,
            clean_text=clean_text,
            threshold=t,
            ngram_max=n,
            top_k_fallback=k,
            hybrid_label_confidence_threshold=h,
        )
        for field_set, clean_text, t, n, k, h in product(
            field_sets,
            clean_values,
            thresholds,
            ngram_max_values,
            top_k_values,
            hybrid_label_conf_values,
        )
    ]


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
    if args.optimise_metric == "weighted_primary":
        _validate_weighted_primary_weights()


def _clean_values(mode: str) -> list[bool]:
    if mode == "off":
        return [False]
    if mode == "on":
        return [True]
    return [False, True]


def _resolve_inputs(args: argparse.Namespace) -> tuple[tuple[str, ...], list[int], list[GridParams]]:
    sample_files = tuple(args.sample_files) if args.sample_files else DEFAULT_SAMPLE_FILES
    random_states = [args.random_state] if args.random_state is not None else _parse_int_csv(args.random_states)
    field_sets = _parse_field_sets(args.fields)
    if any(len(field_set) == 0 for field_set in field_sets):
        raise SystemExit("Field sets must include at least one field")
    grid_params = _grid(args, field_sets=field_sets)
    if not grid_params:
        raise SystemExit("No parameter combinations generated")
    return sample_files, random_states, grid_params


def _print_comparison_metrics(row: pd.Series) -> None:
    print("\nComparison metrics")
    print(
        f" - rows: regex={int(round(float(row['regex_rows_mean'])))} | "
        f"ovr={int(round(float(row['ovr_rows_mean'])))} | "
        f"hybrid={int(round(float(row['hybrid_rows_mean'])))}"
    )
    for metric in DISPLAY_METRICS:
        print(
            f" - {metric}: "
            f"regex={_fmt(float(row[f'regex_{metric}_mean']))} | "
            f"ovr={_fmt(float(row[f'ovr_{metric}_mean']))} | "
            f"hybrid={_fmt(float(row[f'hybrid_{metric}_mean']))}"
        )


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
            f"threshold={row['threshold']:.3f}, "
            f"ngram_max={int(row['ngram_max'])}, "
            f"top_k_fallback={int(row['top_k_fallback'])}"
        )
        if approach == "hybrid":
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
    print(f" - threshold: {best['threshold']:.3f}")
    print(f" - ngram_max: {int(best['ngram_max'])}")
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
            f"{WEIGHTED_PRIMARY_RECALL_MICRO:.3f}*Recall_micro"
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
    cache_key = (params.ngram_max, params.clean_text, tuple(params.fields), int(args.n_jobs))
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
            clean_text=params.clean_text,
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
        for idx, gp in enumerate(grid_params, start=1):
            if not getattr(args, "quiet", False):
                print(
                    f"  [{idx}/{len(grid_params)}] fields={','.join(gp.fields)}, "
                    f"clean_text={'on' if gp.clean_text else 'off'}, threshold={gp.threshold:.3f}, "
                    f"ngram_max={gp.ngram_max}, top_k_fallback={gp.top_k_fallback}"
                    f", hybrid_label_confidence_threshold="
                    f"{_fmt_hybrid_conf(gp.hybrid_label_confidence_threshold, hybrid_conf_precision)}"
                )
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
    if optimise_metric == "weighted_primary":
        # Weighted objective uses top-of-file coefficients.
        weighted_col = _candidate_column("weighted_primary", approach)
        summary_df = summary_df.copy()
        summary_df.loc[:, weighted_col] = (
            WEIGHTED_PRIMARY_F1_MICRO * summary_df[_candidate_column("f1_micro", approach)]
            + WEIGHTED_PRIMARY_F1_MACRO * summary_df[_candidate_column("f1_macro", approach)]
            + WEIGHTED_PRIMARY_RECALL_MICRO * summary_df[_candidate_column("recall_micro", approach)]
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
    ovr_best: pd.Series,
    ovr_summary_df: pd.DataFrame,
    ovr_sort_by: str,
    hybrid_best: pd.Series,
    hybrid_summary_df: pd.DataFrame,
    hybrid_sort_by: str,
    args: argparse.Namespace,
    random_states: Sequence[int],
) -> None:
    hybrid_conf_precision = _resolve_hybrid_conf_precision(args)
    merged_best = ovr_best.copy()
    for metric in DISPLAY_METRICS + ("rows",):
        merged_best[f"hybrid_{metric}_mean"] = hybrid_best[f"hybrid_{metric}_mean"]
    _print_comparison_metrics(merged_best)
    print(
        f"\nNote: Best OVR and best Hybrid combinations were selected independently "
        f"(each ranked by `{args.optimise_metric}`) and averaged across {len(random_states)} random state(s)."
    )
    print("Best OVR parameter combination")
    _print_best_params(
        best=ovr_best,
        args=args,
        hybrid_conf_precision=hybrid_conf_precision,
        include_hybrid=False,
        show_selected_by=ovr_sort_by,
    )

    print("Best Hybrid parameter combination")
    _print_best_params(
        best=hybrid_best,
        args=args,
        hybrid_conf_precision=hybrid_conf_precision,
        include_hybrid=True,
        show_selected_by=hybrid_sort_by,
    )

    if args.show_top == 0:
        return

    print(f"\nTop {args.show_top} OVR parameter combinations")
    print(f" (ranked by {ovr_sort_by})")
    _print_top_rows(
        summary_df=ovr_summary_df,
        show_top=args.show_top,
        approach="ovr",
        hybrid_conf_precision=hybrid_conf_precision,
        dedupe_cols=("clean_text", "threshold", "ngram_max", "top_k_fallback"),
    )
    print(f"\nTop {args.show_top} Hybrid parameter combinations")
    print(f" (ranked by {hybrid_sort_by})")
    _print_top_rows(
        summary_df=hybrid_summary_df,
        show_top=args.show_top,
        approach="hybrid",
        hybrid_conf_precision=hybrid_conf_precision,
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
    _print_comparison_metrics(best)
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
        include_hybrid=(approach == "hybrid"),
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
    sample_files, random_states, grid_params = _resolve_inputs(args)
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
    if len(selected_approaches) == 2:
        ovr_summary_df, ovr_sort_by = _select_ranked_summary(
            summary_df=summary_df,
            optimise_metric=args.optimise_metric,
            approach="ovr",
        )
        hybrid_summary_df, hybrid_sort_by = _select_ranked_summary(
            summary_df=summary_df,
            optimise_metric=args.optimise_metric,
            approach="hybrid",
        )

        ovr_best = ovr_summary_df.iloc[0]
        hybrid_best = hybrid_summary_df.iloc[0]
        _print_results(
            ovr_best=ovr_best,
            ovr_summary_df=ovr_summary_df,
            ovr_sort_by=ovr_sort_by,
            hybrid_best=hybrid_best,
            hybrid_summary_df=hybrid_summary_df,
            hybrid_sort_by=hybrid_sort_by,
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
