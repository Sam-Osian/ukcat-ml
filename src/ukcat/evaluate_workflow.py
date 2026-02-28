from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import pandas as pd
from sklearn.model_selection import train_test_split

from ukcat.evaluate import (
    _evaluate_ukcat_ovr_rows_with_scores,
    _evaluate_ukcat_regex_rows,
    _load_labelled_ukcat,
    _score_ukcat_multilabel,
)
from ukcat import evaluate_grid as grid
from ukcat.ml_ukcat_hybrid import combine_hybrid_predictions
from ukcat.settings import (
    SAMPLE_FILE,
    TOP2000_FILE,
    UKCAT_BEST_DEV_CONFIG,
    UKCAT_DEV_FILE,
    UKCAT_FINAL_TEST_FILE,
    UKCAT_HOLDOUT_SPLIT_FILE,
)

DEFAULT_SOURCE_FILES = (SAMPLE_FILE, TOP2000_FILE)
DEFAULT_DEV_FILE = UKCAT_DEV_FILE
DEFAULT_TEST_FILE = UKCAT_FINAL_TEST_FILE
DEFAULT_META_FILE = UKCAT_HOLDOUT_SPLIT_FILE
DEFAULT_CFG_FILE = UKCAT_BEST_DEV_CONFIG
DEFAULT_SPLIT_RANDOM_STATE = 2026
DEFAULT_TEST_SIZE = 0.2
DEFAULT_N_JOBS = 4
FINAL_METRIC_ORDER = ("rows",) + grid.DISPLAY_METRICS

# Workflow model notes:
# - The saved OVR config describes how charity text is converted into one shared
#   text model, then split into one binary classifier per UK-CAT code.
# - The saved hybrid config points to that trained OVR model and adds only the
#   combination rule needed to merge OvR evidence with regex outputs.


def _ensure_parent(path_str: str) -> Path:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _check_exists(path_str: str, label: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise click.ClickException(f"{label} not found: {path}. Run `ukcat evaluate make-split` first.")
    return path


def _format_metric(value: float, metric_name: str) -> str:
    if metric_name == "rows":
        return str(int(value))
    return f"{value:.4f}"


def _weighted_primary_from_config(metrics: dict[str, float], config: dict[str, Any]) -> float:
    weights = config.get("weighted_primary", {})
    micro_w = float(weights.get("f1_micro", grid.WEIGHTED_PRIMARY_F1_MICRO))
    macro_w = float(weights.get("f1_macro", grid.WEIGHTED_PRIMARY_F1_MACRO))
    recall_w = float(weights.get("recall_micro", grid.WEIGHTED_PRIMARY_RECALL_MICRO))
    return (
        micro_w * float(metrics["f1_micro"])
        + macro_w * float(metrics["f1_macro"])
        + recall_w * float(metrics["recall_micro"])
    )


def _metrics_with_weighted_primary(eval_df: pd.DataFrame, config: dict[str, Any]) -> dict[str, float]:
    metrics = _score_ukcat_multilabel(eval_df)
    metrics["weighted_primary"] = _weighted_primary_from_config(metrics, config)
    return metrics


def _make_split(random_state: int, test_size: float) -> None:
    if not 0 < test_size < 1:
        raise click.ClickException("--final-test-size must be between 0 and 1")

    labelled = _load_labelled_ukcat(DEFAULT_SOURCE_FILES)
    dev_df, test_df = train_test_split(
        labelled,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    dev_df = dev_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    overlap = set(dev_df["org_id"].astype(str)).intersection(test_df["org_id"].astype(str))
    if overlap:
        raise click.ClickException("Split failed: dev and final test contain overlapping org_id values.")

    dev_path = _ensure_parent(DEFAULT_DEV_FILE)
    test_path = _ensure_parent(DEFAULT_TEST_FILE)
    meta_path = _ensure_parent(DEFAULT_META_FILE)

    dev_df.to_csv(dev_path, index=False)
    test_df.to_csv(test_path, index=False)
    meta = {
        "source_files": list(DEFAULT_SOURCE_FILES),
        "random_state": random_state,
        "final_test_size": test_size,
        "labelled_rows": int(len(labelled)),
        "dev_rows": int(len(dev_df)),
        "final_test_rows": int(len(test_df)),
        "dev_file": DEFAULT_DEV_FILE,
        "final_test_file": DEFAULT_TEST_FILE,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    click.echo(f"Loaded labelled rows: {len(labelled):,}")
    click.echo(f"Dev rows: {len(dev_df):,}")
    click.echo(f"Final test rows: {len(test_df):,}")
    click.echo(f"Saved dev split to {dev_path}")
    click.echo(f"Saved final test split to {test_path}")
    click.echo(f"Saved split metadata to {meta_path}")


def _row_to_config(row: pd.Series, approach: str, args, sort_by: str) -> dict[str, Any]:
    config = {
        "approach": approach,
        "selected_by": sort_by,
        "fields": [part for part in str(row["fields_key"]).split(",") if part],
        "threshold": float(row["threshold"]),
        "ngram_max": int(row["ngram_max"]),
        "top_k_fallback": int(row["top_k_fallback"]),
        "clean_text": bool(row["clean_text"]),
        "include_groups": bool(args.include_groups),
    }
    if approach == "hybrid":
        config["hybrid_rule"] = args.hybrid_rule
        config["hybrid_conf"] = float(row["hybrid_label_confidence_threshold"])
    return config


def _save_best_config(payload: dict[str, Any]) -> None:
    out_path = _ensure_parent(DEFAULT_CFG_FILE)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    click.echo(f"\nSaved best dev config to {out_path}")


def _load_best_config() -> dict[str, Any]:
    path = _check_exists(DEFAULT_CFG_FILE, "Best dev config")
    return json.loads(path.read_text(encoding="utf-8"))


def _run_dev_grid(
    show_top: int,
    n_jobs: int,
    clean_mode: str,
    random_state: int | None = None,
) -> dict[str, Any]:
    _check_exists(DEFAULT_DEV_FILE, "Dev split")
    parser = grid._build_parser()
    args = parser.parse_args([])
    args.sample_files = [DEFAULT_DEV_FILE]
    args.show_top = show_top
    args.n_jobs = n_jobs
    args.clean_text_mode = clean_mode
    args.random_state = random_state

    selected = grid._resolve_selected_approaches_from_knobs()
    args.optimise_metric = grid._resolve_optimise_metric_from_knobs()
    grid._validate_args(args)

    sample_files, random_states, grid_params = grid._resolve_inputs(args)
    labelled = _load_labelled_ukcat(sample_files)
    grid._print_run_header(
        sample_files=sample_files,
        labelled_rows=len(labelled),
        random_states=random_states,
        grid_count=len(grid_params),
        optimise_metric=args.optimise_metric,
        clean_mode=args.clean_text_mode,
    )

    state_df = grid._run_grid_for_states(
        labelled=labelled,
        random_states=random_states,
        grid_params=grid_params,
        args=args,
    )
    summary_df = grid._aggregate_results(state_df)

    payload: dict[str, Any] = {
        "dev_file": DEFAULT_DEV_FILE,
        "final_test_file": DEFAULT_TEST_FILE,
        "best_config_file": DEFAULT_CFG_FILE,
        "optimise_metric": args.optimise_metric,
        "selected_approaches": selected,
        "weighted_primary": {
            "f1_micro": grid.WEIGHTED_PRIMARY_F1_MICRO,
            "f1_macro": grid.WEIGHTED_PRIMARY_F1_MACRO,
            "recall_micro": grid.WEIGHTED_PRIMARY_RECALL_MICRO,
        },
        "random_states": list(random_states),
        "ovr": None,
        "hybrid": None,
    }

    if len(selected) == 2:
        ovr_summary_df, ovr_sort_by = grid._select_ranked_summary(
            summary_df=summary_df,
            optimise_metric=args.optimise_metric,
            approach="ovr",
        )
        hybrid_summary_df, hybrid_sort_by = grid._select_ranked_summary(
            summary_df=summary_df,
            optimise_metric=args.optimise_metric,
            approach="hybrid",
        )
        ovr_best = ovr_summary_df.iloc[0]
        hybrid_best = hybrid_summary_df.iloc[0]
        grid._print_results(
            ovr_best=ovr_best,
            ovr_summary_df=ovr_summary_df,
            ovr_sort_by=ovr_sort_by,
            hybrid_best=hybrid_best,
            hybrid_summary_df=hybrid_summary_df,
            hybrid_sort_by=hybrid_sort_by,
            args=args,
            random_states=random_states,
        )
        payload["ovr"] = _row_to_config(ovr_best, "ovr", args, ovr_sort_by)
        payload["hybrid"] = _row_to_config(hybrid_best, "hybrid", args, hybrid_sort_by)
    else:
        approach = selected[0]
        best_df, sort_by = grid._select_ranked_summary(
            summary_df=summary_df,
            optimise_metric=args.optimise_metric,
            approach=approach,
        )
        best = best_df.iloc[0]
        grid._print_single_results(
            best=best,
            summary_df=best_df,
            approach=approach,
            sort_by=sort_by,
            args=args,
            random_states=random_states,
        )
        payload[approach] = _row_to_config(best, approach, args, sort_by)

    _save_best_config(payload)
    return payload


def _load_split(path_str: str) -> pd.DataFrame:
    path = _check_exists(path_str, "Split file")
    df = pd.read_csv(path)
    if "org_id" not in df.columns or "UKCAT" not in df.columns:
        raise click.ClickException(f"Expected UK-CAT labelled data columns in {path}")
    return df.reset_index(drop=True)


def _build_ovr_artifacts(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict[str, Any],
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Final holdout evaluation re-fits the selected OvR setup on the dev split,
    # then scores the untouched test split once.
    #
    # Under the hood, the OvR fit:
    # - concatenates the chosen text fields for each charity row
    # - vectorises that text into one shared feature matrix
    # - fits one binary classifier per UK-CAT code on that matrix
    # The returned probability frame is the charity-by-code score table from that
    # fit, and the hybrid combiner reuses it directly.
    return _evaluate_ukcat_ovr_rows_with_scores(
        train_df=train_df,
        test_df=test_df,
        fields=tuple(config["fields"]),
        threshold=float(config["threshold"]),
        top_k_fallback=int(config["top_k_fallback"]),
        n_jobs=n_jobs,
        ngram_max=int(config["ngram_max"]),
        clean_text=bool(config["clean_text"]),
    )


def _print_final_metrics(summary_df: pd.DataFrame) -> None:
    click.echo("Final holdout comparison metrics")
    for metric_name in FINAL_METRIC_ORDER:
        parts = [
            f"{row['approach']}={_format_metric(float(row[metric_name]), metric_name)}"
            for _, row in summary_df.iterrows()
        ]
        click.echo(f" - {metric_name}: " + " | ".join(parts))


def _run_final_holdout(n_jobs: int) -> pd.DataFrame:
    config = _load_best_config()
    dev_df = _load_split(DEFAULT_DEV_FILE)
    test_df = _load_split(DEFAULT_TEST_FILE)

    overlap = set(dev_df["org_id"].astype(str)).intersection(test_df["org_id"].astype(str))
    if overlap:
        raise click.ClickException("Dev and final test splits overlap on org_id. Final evaluation would leak.")

    ovr_config = config.get("ovr")
    hybrid_config = config.get("hybrid")
    if ovr_config is None and hybrid_config is None:
        raise click.ClickException("Best dev config does not contain OVR or Hybrid selections. Run `ukcat evaluate dev-grid` first.")

    include_groups = bool((hybrid_config or ovr_config)["include_groups"])
    click.echo(f"Loaded dev rows: {len(dev_df):,}")
    click.echo(f"Loaded final test rows: {len(test_df):,}")

    regex_eval_df = _evaluate_ukcat_regex_rows(
        test_df=test_df,
        include_groups=include_groups,
    )

    # The final report always includes regex so the selected ML variants are
    # judged against the existing production-style baseline on exactly the same
    # holdout rows.
    summary_rows: list[dict[str, Any]] = [
        {"approach": "regex", **_metrics_with_weighted_primary(regex_eval_df, config)}
    ]

    if ovr_config is not None:
        click.echo(
            "Best OVR config: "
            f"fields={','.join(ovr_config['fields'])}, threshold={ovr_config['threshold']:.3f}, "
            f"ngram_max={ovr_config['ngram_max']}, top_k_fallback={ovr_config['top_k_fallback']}"
        )
        ovr_eval_df, _ = _build_ovr_artifacts(
            train_df=dev_df,
            test_df=test_df,
            config=ovr_config,
            n_jobs=n_jobs,
        )
        summary_rows.append({"approach": "ovr", **_metrics_with_weighted_primary(ovr_eval_df, config)})

    if hybrid_config is not None:
        click.echo(
            "Best Hybrid config: "
            f"fields={','.join(hybrid_config['fields'])}, threshold={hybrid_config['threshold']:.3f}, "
            f"ngram_max={hybrid_config['ngram_max']}, top_k_fallback={hybrid_config['top_k_fallback']}, "
            f"hybrid_rule={hybrid_config['hybrid_rule']}, "
            f"hybrid_conf={hybrid_config['hybrid_conf']}"
        )
        hybrid_eval_ovr, hybrid_prob = _build_ovr_artifacts(
            train_df=dev_df,
            test_df=test_df,
            config=hybrid_config,
            n_jobs=n_jobs,
        )
        # Hybrid is a decision layer over regex + OvR. 
        # - regex contributes deterministic label hits from explicit rules
        # - OvR contributes per-label probabilities learned from labelled text
        # - the hybrid rule decides when the statistical signal should reinforce,
        #   suppress, or otherwise alter the regex output
        # There is no extra hybrid training stage beyond the OVR fit above.
        hybrid_eval_df = combine_hybrid_predictions(
            ovr_eval_df=hybrid_eval_ovr,
            regex_eval_df=regex_eval_df,
            ovr_probability_df=hybrid_prob,
            rule=str(hybrid_config["hybrid_rule"]),
            label_confidence_threshold=float(hybrid_config["hybrid_conf"]),
        )
        summary_rows.append({"approach": "hybrid", **_metrics_with_weighted_primary(hybrid_eval_df, config)})

    summary_df = pd.DataFrame(summary_rows)
    _print_final_metrics(summary_df)
    return summary_df


@click.command("make-split")
@click.option("--random-state", default=DEFAULT_SPLIT_RANDOM_STATE, type=int, show_default=True)
@click.option("--final-test-size", default=DEFAULT_TEST_SIZE, type=float, show_default=True)
def evaluate_make_split(random_state: int, final_test_size: float) -> None:
    """Create the fixed dev/final-test split."""
    _make_split(random_state=random_state, test_size=final_test_size)


@click.command("dev-grid")
@click.option("--show-top", default=0, type=int, show_default=True)
@click.option(
    "--clean-text-mode",
    default=grid.DEFAULT_CLEAN_TEXT_MODE,
    type=click.Choice(["off", "on", "compare"]),
    show_default=True,
)
@click.option("--random-state", default=None, type=int)
@click.option("--n-jobs", default=DEFAULT_N_JOBS, type=int, show_default=True)
def evaluate_dev_grid(show_top: int, clean_text_mode: str, random_state: int | None, n_jobs: int) -> None:
    """Run the dev-only grid search on the fixed dev split."""
    _run_dev_grid(show_top=show_top, n_jobs=n_jobs, clean_mode=clean_text_mode, random_state=random_state)


@click.command("final-holdout")
@click.option("--n-jobs", default=DEFAULT_N_JOBS, type=int, show_default=True)
def evaluate_final_holdout(n_jobs: int) -> None:
    """Run the locked final holdout evaluation using the saved best dev config."""
    _run_final_holdout(n_jobs=n_jobs)


@click.command("run-full-workflow")
@click.option("--show-top", default=0, type=int, show_default=True)
@click.option(
    "--clean-text-mode",
    default=grid.DEFAULT_CLEAN_TEXT_MODE,
    type=click.Choice(["off", "on", "compare"]),
    show_default=True,
)
@click.option("--random-state", default=None, type=int)
@click.option("--n-jobs", default=DEFAULT_N_JOBS, type=int, show_default=True)
def evaluate_run_full_workflow(show_top: int, clean_text_mode: str, random_state: int | None, n_jobs: int) -> None:
    """Run dev-grid and then locked final-holdout using the saved best config."""
    _check_exists(DEFAULT_DEV_FILE, "Dev split")
    _check_exists(DEFAULT_TEST_FILE, "Final test split")
    _run_dev_grid(
        show_top=show_top,
        n_jobs=n_jobs,
        clean_mode=clean_text_mode,
        random_state=random_state,
    )
    click.echo("")
    _run_final_holdout(n_jobs=n_jobs)
