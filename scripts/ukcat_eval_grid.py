from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List, Sequence, Union

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
    DEFAULT_HYBRID_LABEL_CONFIDENCE_THRESHOLD,
    HYBRID_RULE_LABEL_CONF_GATED_REGEX,
    combine_hybrid_predictions,
)

# Set the below to either lists of integers to iterate; or just lone itegers if you don't want to iterate
DEFAULT_SAMPLE_FILES = ("data/sample.csv", "data/top2000.csv")
DEFAULT_RANDOM_STATES = (67, 2026, 123, 42)
DEFAULT_THRESHOLDS = (0.10, 0.12, 0.15)
DEFAULT_NGRAM_MAX_VALUES = (1, 2, 3)
DEFAULT_TOP_K_FALLBACK_VALUES = 1
DEFAULT_FIELDS = ("name", "activities", "objects")
DEFAULT_HYBRID_LABEL_CONFIDENCE_THRESHOLDS = (0.01, 0.015, 0.020, 0.025)

OPTIMISATION_METRICS = (
    "precision_micro",
    "recall_micro",
    "f1_micro",
    "f1_macro",
    "subset_accuracy",
    "jaccard_samples",
    "hamming_loss",
)
MODEL_APPROACH_CHOICES = ("ovr", "hybrid")
DEFAULT_MIN_OVR_PRECISION_MICRO = 0.45
DEFAULT_MIN_DELTA_F1_MICRO = 0.0

DISPLAY_METRICS = (
    "precision_micro",
    "recall_micro",
    "f1_micro",
    "f1_macro",
    "subset_accuracy",
    "jaccard_samples",
    "hamming_loss",
)
GRID_GROUP_COLUMNS = ("threshold", "ngram_max", "top_k_fallback", "hybrid_label_confidence_threshold")
AGGREGATED_METRIC_COLUMNS = (
    "regex_rows",
    "ovr_rows",
    "hybrid_rows",
    "regex_precision_micro",
    "regex_recall_micro",
    "regex_f1_micro",
    "regex_f1_macro",
    "regex_subset_accuracy",
    "regex_jaccard_samples",
    "regex_hamming_loss",
    "ovr_precision_micro",
    "ovr_recall_micro",
    "ovr_f1_micro",
    "ovr_f1_macro",
    "ovr_subset_accuracy",
    "ovr_jaccard_samples",
    "ovr_hamming_loss",
    "hybrid_precision_micro",
    "hybrid_recall_micro",
    "hybrid_f1_micro",
    "hybrid_f1_macro",
    "hybrid_subset_accuracy",
    "hybrid_jaccard_samples",
    "hybrid_hamming_loss",
)


@dataclass(frozen=True)
class GridParams:
    threshold: float
    ngram_max: int
    top_k_fallback: int
    hybrid_label_confidence_threshold: float


def _parse_csv_list(value: Union[str, int, float]) -> List[str]:
    if isinstance(value, (tuple, list)):
        return [str(v) for v in value]
    if isinstance(value, (int, float)):
        return [str(value)]
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_int_csv(value: str) -> List[int]:
    return [int(v) for v in _parse_csv_list(value)]


def _parse_float_csv(value: str) -> List[float]:
    return [float(v) for v in _parse_csv_list(value)]


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
    parser.add_argument("--n-jobs", type=int, default=8, help="OvR parallel jobs.")
    parser.add_argument(
        "--fields",
        default=",".join(DEFAULT_FIELDS),
        help="Comma-separated text fields used by OvR.",
    )
    parser.add_argument(
        "--clean-text",
        action="store_true",
        help="Apply the existing NLP text cleaning before vectorisation.",
    )
    parser.add_argument(
        "--include-groups",
        action="store_true",
        help="Include parent/group codes in regex predictions.",
    )
    parser.add_argument(
        "--candidate-approach",
        choices=MODEL_APPROACH_CHOICES,
        default="ovr",
        help="Candidate approach to optimise against regex (ovr or hybrid).",
    )
    parser.add_argument(
        "--hybrid-rule",
        choices=HYBRID_RULE_CHOICES,
        default=HYBRID_RULE_LABEL_CONF_GATED_REGEX,
        help="Hybrid decision rule used when --candidate-approach=hybrid.",
    )
    parser.add_argument(
        "--optimise-metric",
        choices=OPTIMISATION_METRICS,
        default="f1_micro",
        help="Candidate metric used to rank parameter combinations (best combination selection).",
    )
    parser.add_argument(
        "--show-top-5",
        action="store_true",
        help="Print the top 5 parameter combinations below the average results.",
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


def _aggregate_results(per_state_df: pd.DataFrame) -> pd.DataFrame:
    mean_df = per_state_df.groupby(list(GRID_GROUP_COLUMNS))[list(AGGREGATED_METRIC_COLUMNS)].mean().reset_index()
    std_df = per_state_df.groupby(list(GRID_GROUP_COLUMNS))[list(AGGREGATED_METRIC_COLUMNS)].std().reset_index()

    # Suffix columns so the output table is explicit about what is averaged.
    mean_df = mean_df.rename(columns={c: f"{c}_mean" for c in AGGREGATED_METRIC_COLUMNS})
    std_df = std_df.rename(columns={c: f"{c}_std" for c in AGGREGATED_METRIC_COLUMNS})

    return mean_df.merge(std_df, on=list(GRID_GROUP_COLUMNS), how="left")


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _score_pair(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    regex_eval_df: pd.DataFrame,
    params: GridParams,
    fields: Sequence[str],
    n_jobs: int,
    clean_text: bool,
    candidate_approach: str,
    hybrid_rule: str,
) -> dict:
    ovr_eval_df, ovr_probability_df = _evaluate_ukcat_ovr_rows_with_scores(
        train_df=train_df,
        test_df=test_df,
        fields=fields,
        threshold=params.threshold,
        top_k_fallback=params.top_k_fallback,
        n_jobs=n_jobs,
        ngram_max=params.ngram_max,
        clean_text=clean_text,
    )
    hybrid_eval_df = combine_hybrid_predictions(
        ovr_eval_df=ovr_eval_df,
        regex_eval_df=regex_eval_df,
        ovr_probability_df=ovr_probability_df,
        rule=hybrid_rule,
        label_confidence_threshold=params.hybrid_label_confidence_threshold,
    )
    regex_metrics = _score_ukcat_multilabel(regex_eval_df)
    pure_ovr_metrics = _score_ukcat_multilabel(ovr_eval_df)
    hybrid_metrics = _score_ukcat_multilabel(hybrid_eval_df)

    row = {
        "threshold": params.threshold,
        "ngram_max": params.ngram_max,
        "top_k_fallback": params.top_k_fallback,
        "hybrid_label_confidence_threshold": params.hybrid_label_confidence_threshold,
    }
    for key, value in regex_metrics.items():
        row[f"regex_{key}"] = value
    for key, value in pure_ovr_metrics.items():
        row[f"ovr_{key}"] = value
    for key, value in hybrid_metrics.items():
        row[f"hybrid_{key}"] = value

    return row


def _grid(params: argparse.Namespace) -> List[GridParams]:
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
            threshold=t,
            ngram_max=n,
            top_k_fallback=k,
            hybrid_label_confidence_threshold=h,
        )
        for t, n, k, h in product(thresholds, ngram_max_values, top_k_values, hybrid_label_conf_values)
    ]


def _validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.test_size < 1:
        raise SystemExit("--test-size must be between 0 and 1")
    if args.n_jobs < 1:
        raise SystemExit("--n-jobs must be at least 1")


def _resolve_inputs(args: argparse.Namespace) -> tuple[tuple[str, ...], list[int], tuple[str, ...], list[GridParams]]:
    sample_files = tuple(args.sample_files) if args.sample_files else DEFAULT_SAMPLE_FILES
    random_states = [args.random_state] if args.random_state is not None else _parse_int_csv(args.random_states)
    fields = tuple(_parse_csv_list(args.fields))
    grid_params = _grid(args)
    if not grid_params:
        raise SystemExit("No parameter combinations generated")
    return sample_files, random_states, fields, grid_params


def _print_run_header(sample_files: Sequence[str], labelled_rows: int, random_states: Sequence[int], grid_count: int) -> None:
    print(f"Loading labelled data from: {', '.join(sample_files)}")
    print(f"Loaded labelled rows: {labelled_rows:,}")
    print(f"Random states: {list(random_states)}")
    print(f"Grid combinations: {grid_count}")
    print(f"Total OvR runs: {grid_count * len(random_states)}")
    print("")


def _run_grid_for_states(
    labelled: pd.DataFrame,
    random_states: Sequence[int],
    grid_params: Sequence[GridParams],
    args: argparse.Namespace,
    fields: Sequence[str],
) -> pd.DataFrame:
    per_state_rows = []
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

        # Regex is independent of OvR tuning parameters, so compute it once per split
        regex_eval_df = _evaluate_ukcat_regex_rows(test_df=test_df, include_groups=args.include_groups)

        for idx, gp in enumerate(grid_params, start=1):
            print(
                f"  [{idx}/{len(grid_params)}] threshold={gp.threshold:.3f}, "
                f"ngram_max={gp.ngram_max}, top_k_fallback={gp.top_k_fallback}"
                f", hybrid_label_confidence_threshold={gp.hybrid_label_confidence_threshold:.4f}"
            )
            result = _score_pair(
                train_df=train_df,
                test_df=test_df,
                regex_eval_df=regex_eval_df,
                params=gp,
                fields=fields,
                n_jobs=args.n_jobs,
                clean_text=args.clean_text,
                candidate_approach=args.candidate_approach,
                hybrid_rule=args.hybrid_rule,
            )
            result["random_state"] = state
            per_state_rows.append(result)
    return pd.DataFrame(per_state_rows)


def _candidate_column(metric: str, args: argparse.Namespace) -> str:
    return f"{args.candidate_approach}_{metric}_mean"


def _select_ranked_summary(summary_df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    sort_by = _candidate_column(args.optimise_metric, args)
    if sort_by not in summary_df.columns:
        raise SystemExit(f"Ranking column not found: {sort_by}")

    candidate_precision_col = _candidate_column("precision_micro", args)
    candidate_f1_col = _candidate_column("f1_micro", args)

    # Using some built-in guardrails keep recall-first optimisation from selecting obviously poor trade-offs
    summary_df = summary_df[summary_df[candidate_precision_col] >= DEFAULT_MIN_OVR_PRECISION_MICRO].copy()
    summary_df = summary_df[(summary_df[candidate_f1_col] - summary_df["regex_f1_micro_mean"]) >= DEFAULT_MIN_DELTA_F1_MICRO].copy()
    if summary_df.empty:
        raise SystemExit(
            "No parameter combinations left after applying built-in guardrails "
            f"(candidate_precision_micro >= {DEFAULT_MIN_OVR_PRECISION_MICRO}, "
            f"candidate_delta_f1_micro >= {DEFAULT_MIN_DELTA_F1_MICRO})."
        )

    ascending = args.optimise_metric == "hamming_loss"
    return summary_df.sort_values(sort_by, ascending=ascending).reset_index(drop=True), sort_by


def _print_results(best: pd.Series, summary_df: pd.DataFrame, args: argparse.Namespace, random_states: Sequence[int], sort_by: str) -> None:
    print("\nComparison metrics")
    print(
        f" - rows: regex={int(round(float(best['regex_rows_mean'])))} | "
        f"ovr={int(round(float(best['ovr_rows_mean'])))} | "
        f"hybrid={int(round(float(best['hybrid_rows_mean'])))}"
    )
    for metric in DISPLAY_METRICS:
        print(
            f" - {metric}: "
            f"regex={_fmt(float(best[f'regex_{metric}_mean']))} | "
            f"ovr={_fmt(float(best[f'ovr_{metric}_mean']))} | "
            f"hybrid={_fmt(float(best[f'hybrid_{metric}_mean']))}"
        )
    print(
        f"\nNote: The best parameter combination was selected for "
        f"{args.candidate_approach.upper()} "
        f"(selected by `{sort_by}`) and averaged across {len(random_states)} random states."
    )
    print("Best parameter combination")
    print(f" - threshold: {best['threshold']:.3f}")
    print(f" - ngram_max: {int(best['ngram_max'])}")
    print(f" - top_k_fallback: {int(best['top_k_fallback'])}")
    if args.candidate_approach == "hybrid":
        print(f" - hybrid_rule: {args.hybrid_rule}")
        print(f" - hybrid_label_confidence_threshold: {best['hybrid_label_confidence_threshold']:.4f}")

    if not args.show_top_5:
        return

    print("\nTop 5 parameter combinations")
    print(f" (ranked by {sort_by})")
    for rank, (_, row) in enumerate(summary_df.head(5).iterrows(), start=1):
        print(
            f" - [{rank}] threshold={row['threshold']:.3f}, "
            f"ngram_max={int(row['ngram_max'])}, "
            f"top_k_fallback={int(row['top_k_fallback'])}, "
            f"hybrid_label_confidence_threshold={row['hybrid_label_confidence_threshold']:.4f}"
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


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args)
    sample_files, random_states, fields, grid_params = _resolve_inputs(args)
    labelled = _load_labelled_ukcat(sample_files)
    _print_run_header(sample_files, len(labelled), random_states, len(grid_params))

    per_state_df = _run_grid_for_states(labelled=labelled, random_states=random_states, grid_params=grid_params, args=args, fields=fields)
    summary_df = _aggregate_results(per_state_df)
    summary_df, sort_by = _select_ranked_summary(summary_df, args)

    best = summary_df.iloc[0]
    _print_results(best, summary_df, args, random_states, sort_by)
    _save_optional_outputs(args, per_state_df, summary_df)


if __name__ == "__main__":
    main()
