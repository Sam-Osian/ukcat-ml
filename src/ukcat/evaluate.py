import warnings
from typing import Iterable, List, Optional, Sequence

import click
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from ukcat.ml_icnptso import get_text_corpus
from ukcat.ml_ukcat_ovr import _build_ukcat_ovr_pipeline, _predict_codes
from ukcat.settings import (
    ML_DEFAULT_FIELDS,
    ML_RANDOM_STATE,
    ML_TEST_TRAIN_SIZE,
    SAMPLE_FILE,
    TOP2000_FILE,
    UKCAT_FILE,
)

DEFAULT_SAMPLE_FILES = [SAMPLE_FILE, TOP2000_FILE]
DEFAULT_UKCAT_FIELDS = ["name", "activities"]


def _split_codes(value: object) -> List[str]:
    if not isinstance(value, str):
        return []
    return [code.strip() for code in value.split(";") if code.strip()]


def _join_codes(values: Iterable[str]) -> str:
    return ";".join(sorted(set(values)))


def _train_icnptso_pipeline(x_train: Sequence[str], y_train: Sequence[str]) -> Pipeline:
    model = Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", LogisticRegression(n_jobs=5, C=1e5, max_iter=1000)),
        ]
    )
    model.fit(x_train, y_train)
    return model


@click.command()
@click.option(
    "--sample-files",
    "-s",
    multiple=True,
    default=DEFAULT_SAMPLE_FILES,
    help="CSV files used as gold labels for evaluation",
)
@click.option(
    "--random-state",
    default=2026,
    type=int,
    show_default=True,
    help="Random seed used for the train/test split",
)
@click.option(
    "--test-size",
    default=ML_TEST_TRAIN_SIZE,
    type=float,
    show_default=True,
    help="Fraction of rows used for the test split",
)
@click.option(
    "--save-location",
    default=None,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True),
    help="Optional CSV output with row-level evaluation details",
)
def evaluate_icnptso(
    sample_files: Sequence[str],
    random_state: int,
    test_size: float,
    save_location: Optional[str],
) -> pd.DataFrame:
    """Evaluate ICNPTSO ML predictions against labelled sample files."""
    id_field = "org_id"
    category_field = "ICNPTSO"
    fields_to_use = list(ML_DEFAULT_FIELDS)

    df = pd.concat([pd.read_csv(f) for f in sample_files], ignore_index=True)
    df = df[df[category_field].notna()].copy()
    df = df.drop_duplicates(subset=[id_field], keep="first")

    click.echo(f"Loaded labelled data [{len(df):,} rows]")

    if not 0 < test_size < 1:
        raise click.ClickException("--test-size must be between 0 and 1")

    click.echo("Using holdout train/test evaluation mode")
    x_all = get_text_corpus(df, fields=fields_to_use, do_cleaning=True)
    y_all = df[category_field].astype(str).reset_index(drop=True)
    id_all = df[id_field].reset_index(drop=True)

    stratify = y_all if y_all.nunique() > 1 and y_all.value_counts().min() >= 2 else None
    if stratify is None:
        click.echo(" - unstratified split (insufficient class counts for stratification)")
    else:
        click.echo(" - stratified split")

    x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(
        x_all,
        y_all,
        id_all,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    click.echo(f" - training rows: {len(y_train):,}")
    click.echo(f" - test rows: {len(y_test):,}")

    model = _train_icnptso_pipeline(x_train, y_train)
    y_pred_proba = model.predict_proba(x_test)
    y_pred_proba_df = pd.DataFrame(y_pred_proba, columns=model.classes_)
    y_pred = y_pred_proba_df.idxmax(axis=1).astype(str)
    y_pred_probability = y_pred_proba_df.max(axis=1).round(4)

    results = pd.DataFrame(
        {
            id_field: pd.Series(id_test).values,
            "y_true": pd.Series(y_test).values,
            "y_pred": y_pred.values,
            "prediction_probability": y_pred_probability.values,
            "prediction_source": "ml_model_holdout",
        }
    )

    accuracy = accuracy_score(results["y_true"], results["y_pred"])
    precision_macro = precision_score(results["y_true"], results["y_pred"], average="macro", zero_division=0)
    recall_macro = recall_score(results["y_true"], results["y_pred"], average="macro", zero_division=0)
    f1_macro = f1_score(results["y_true"], results["y_pred"], average="macro", zero_division=0)
    f1_weighted = f1_score(results["y_true"], results["y_pred"], average="weighted", zero_division=0)

    click.echo("Evaluation metrics")
    click.echo(f" - rows: {len(results):,}")
    click.echo(f" - accuracy: {accuracy:.4f}")
    click.echo(f" - precision_macro: {precision_macro:.4f}")
    click.echo(f" - recall_macro: {recall_macro:.4f}")
    click.echo(f" - f1_macro: {f1_macro:.4f}")
    click.echo(f" - f1_weighted: {f1_weighted:.4f}")

    if save_location:
        results.to_csv(save_location, index=False)
        click.echo(f"Saved row-level results to [{save_location}]")

    return results


def _load_labelled_ukcat(sample_files: Sequence[str]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f) for f in sample_files], ignore_index=True)
    labelled = df[df["UKCAT"].notna()].copy()
    # Keep one labelled row per organisation so all approaches score the same units.
    return labelled.drop_duplicates(subset=["org_id"], keep="first").reset_index(drop=True)


def _evaluate_ukcat_regex_rows(test_df: pd.DataFrame, include_groups: bool) -> pd.DataFrame:
    id_field = "org_id"
    corpus = pd.Series(
        index=test_df[id_field].values,
        data=get_text_corpus(test_df, fields=DEFAULT_UKCAT_FIELDS, do_cleaning=False),
    )

    ukcat = pd.read_csv(UKCAT_FILE, index_col="Code")
    results_list = []
    for code, row in ukcat.iterrows():
        if not isinstance(row["Regular expression"], str) or row["Regular expression"] == r"\b()\b":
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="This pattern is interpreted as a regular expression, and has match groups.*",
                category=UserWarning,
            )
            criteria = corpus.str.contains(row["Regular expression"], case=False, regex=True)
            if (
                isinstance(row["Exclude regular expression"], str)
                and row["Exclude regular expression"] != r"\b()\b"
            ):
                criteria = criteria & ~corpus.str.contains(
                    row["Exclude regular expression"], case=False, regex=True
                )
        matched_ids = corpus.index[criteria]
        if len(matched_ids) > 0:
            results_list.append(pd.Series(data=code, index=matched_ids, name="ukcat_code"))

    if results_list:
        predicted = pd.concat(results_list)
    else:
        predicted = pd.Series(dtype=object, name="ukcat_code")

    if include_groups and not predicted.empty:
        predicted = pd.concat(
            [
                predicted,
                predicted.str[0:2],
                predicted[predicted.str[2].astype(int) > 1].apply(lambda x: x[0:3] + "00"),
            ]
        )

    pred_map = predicted.groupby(level=0).apply(lambda x: sorted(set(x.astype(str)))).rename("predicted_codes")
    true_map = (
        test_df.set_index(id_field)["UKCAT"]
        .astype(str)
        .apply(_split_codes)
        .apply(lambda x: sorted(set(x)))
        .rename("true_codes")
    )

    eval_df = true_map.to_frame().join(pred_map, how="left").reset_index()
    eval_df.loc[:, "predicted_codes"] = eval_df["predicted_codes"].apply(lambda x: x if isinstance(x, list) else [])
    eval_df.loc[:, "prediction_source"] = "regex_rules"
    return eval_df[[id_field, "true_codes", "predicted_codes", "prediction_source"]]


def _evaluate_ukcat_ovr_rows(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fields: Sequence[str],
    threshold: float,
    n_jobs: int,
    ngram_max: int,
    clean_text: bool,
) -> pd.DataFrame:
    # Fit on the shared training split and score only the shared test split
    x_train = get_text_corpus(train_df, fields=list(fields), do_cleaning=clean_text)
    x_test = get_text_corpus(test_df, fields=list(fields), do_cleaning=clean_text)
    y_train_codes = train_df["UKCAT"].astype(str).apply(_split_codes)
    y_test_codes = test_df["UKCAT"].astype(str).apply(_split_codes)

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train_codes)

    model = _build_ukcat_ovr_pipeline(n_jobs=n_jobs, ngram_max=ngram_max)
    model.fit(x_train, y_train)

    pred_codes, _ = _predict_codes(model, mlb, x_test, threshold=threshold)

    return pd.DataFrame(
        {
            "org_id": test_df["org_id"].astype(str).values,
            "true_codes": [sorted(set(codes)) for codes in y_test_codes],
            "predicted_codes": pred_codes,
            "prediction_source": "ml_model_holdout_ovr",
        }
    )


def _score_ukcat_multilabel(eval_df: pd.DataFrame) -> dict:
    mlb = MultiLabelBinarizer()
    # Fit the scorer on the union so missing predictions/labels do not drop columns
    all_labels = list(eval_df["true_codes"]) + list(eval_df["predicted_codes"])
    mlb.fit(all_labels)

    y_true = mlb.transform(eval_df["true_codes"])
    y_pred = mlb.transform(eval_df["predicted_codes"])

    return {
        "rows": float(len(eval_df)),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "subset_accuracy": accuracy_score(y_true, y_pred),
        "jaccard_samples": jaccard_score(y_true, y_pred, average="samples", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
    }


def _print_ukcat_metrics(metrics: dict) -> None:
    click.echo("Evaluation metrics")
    click.echo(f" - rows: {int(metrics['rows']):,}")
    click.echo(f" - precision_micro: {metrics['precision_micro']:.4f}")
    click.echo(f" - recall_micro: {metrics['recall_micro']:.4f}")
    click.echo(f" - f1_micro: {metrics['f1_micro']:.4f}")
    click.echo(f" - f1_macro: {metrics['f1_macro']:.4f}")
    click.echo(f" - subset_accuracy: {metrics['subset_accuracy']:.4f}")
    click.echo(f" - jaccard_samples: {metrics['jaccard_samples']:.4f}")
    click.echo(f" - hamming_loss: {metrics['hamming_loss']:.4f}")


def _serialize_ukcat_eval_rows(eval_df: pd.DataFrame) -> pd.DataFrame:
    output = eval_df.copy()
    output.loc[:, "true_codes"] = output["true_codes"].apply(_join_codes)
    output.loc[:, "predicted_codes"] = output["predicted_codes"].apply(_join_codes)
    return output


def _format_compare_metric(value: float, metric_name: str) -> str:
    if metric_name == "rows":
        return str(int(value))
    return f"{value:.4f}"


@click.command()
@click.option(
    "--sample-files",
    "-s",
    multiple=True,
    default=DEFAULT_SAMPLE_FILES,
    help="CSV files used as gold labels for evaluation",
)
@click.option(
    "--include-groups/--no-include-groups",
    default=False,
    help="Include parent/group codes in predictions before scoring",
)
@click.option(
    "--compare/--no-compare",
    default=False,
    show_default=True,
    help="Compare all registered UKCAT approaches on a shared holdout split",
)
@click.option(
    "--approach",
    "approaches",
    type=click.Choice(["regex", "ovr"], case_sensitive=False),
    multiple=True,
    default=(),
    help="Approach(es) to run in compare mode (defaults to all registered approaches)",
)
@click.option(
    "--random-state",
    default=2026,
    type=int,
    show_default=True,
    help="Random seed used for compare-mode holdout split",
)
@click.option(
    "--test-size",
    default=ML_TEST_TRAIN_SIZE,
    type=float,
    show_default=True,
    help="Fraction of rows used for compare-mode test split",
)
@click.option(
    "--threshold",
    default=0.5,
    type=float,
    show_default=True,
    help="Probability threshold used by the OvR approach in compare mode",
)
@click.option(
    "--n-jobs",
    default=1,
    type=int,
    show_default=True,
    help="Number of parallel jobs used by the OvR approach in compare mode",
)
@click.option(
    "--ngram-max",
    default=2,
    type=int,
    show_default=True,
    help="Maximum n-gram size for the OvR vectoriser in compare mode (uses 1..N)",
)
@click.option(
    "--fields",
    "-f",
    multiple=True,
    default=DEFAULT_UKCAT_FIELDS,
    help="Fields used by ML approaches to create the text corpus in compare mode",
)
@click.option(
    "--clean-text/--no-clean-text",
    default=False,
    show_default=True,
    help="Apply NLP cleaning for the OvR approach in compare mode",
)
@click.option(
    "--save-location",
    default=None,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True),
    help="Row-level CSV for regex mode, or summary CSV in compare mode",
)
@click.option(
    "--save-predictions-prefix",
    default=None,
    type=str,
    help="Optional prefix for per-approach row-level CSV outputs in compare mode",
)
def evaluate_ukcat(
    sample_files: Sequence[str],
    include_groups: bool,
    compare: bool,
    approaches: Sequence[str],
    random_state: int,
    test_size: float,
    threshold: float,
    n_jobs: int,
    ngram_max: int,
    fields: Sequence[str],
    clean_text: bool,
    save_location: Optional[str],
    save_predictions_prefix: Optional[str],
) -> pd.DataFrame:
    """Evaluate UKCAT regex tagging, or compare UKCAT approaches on a shared split."""
    if not compare:
        labelled = _load_labelled_ukcat(sample_files)
        click.echo(f"Loaded labelled data [{len(labelled):,} rows]")

        eval_df = _evaluate_ukcat_regex_rows(labelled, include_groups=include_groups)
        metrics = _score_ukcat_multilabel(eval_df)
        _print_ukcat_metrics(metrics)

        output = _serialize_ukcat_eval_rows(eval_df)
        if save_location:
            output.to_csv(save_location, index=False)
            click.echo(f"Saved row-level results to [{save_location}]")
        return output

    if not 0 < test_size < 1:
        raise click.ClickException("--test-size must be between 0 and 1")
    if not 0 <= threshold <= 1:
        raise click.ClickException("--threshold must be between 0 and 1")
    if n_jobs < 1:
        raise click.ClickException("--n-jobs must be at least 1")
    if ngram_max < 1:
        raise click.ClickException("--ngram-max must be at least 1")

    labelled = _load_labelled_ukcat(sample_files)
    click.echo(f"Loaded labelled data [{len(labelled):,} rows]")
    click.echo("Using shared holdout split for all approaches")

    train_df, test_df = train_test_split(
        labelled,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    click.echo(f" - training rows: {len(train_df):,}")
    click.echo(f" - test rows: {len(test_df):,}")

    evaluators = {
        "regex": lambda: _evaluate_ukcat_regex_rows(test_df=test_df, include_groups=include_groups),
        "ovr": lambda: _evaluate_ukcat_ovr_rows(
            train_df=train_df,
            test_df=test_df,
            fields=fields,
            threshold=threshold,
            n_jobs=n_jobs,
            ngram_max=ngram_max,
            clean_text=clean_text,
        ),
    }
    # Default compare mode runs every registered approach so future models slot in cleanly..
    approach_list = [a.lower() for a in approaches] if approaches else list(evaluators.keys())
    click.echo(f" - approaches: {', '.join(approach_list)}")

    metric_rows = []
    for approach in approach_list:
        click.echo(f"Scoring approach: {approach}")
        eval_df = evaluators[approach]()
        metric_rows.append({"approach": approach, **_score_ukcat_multilabel(eval_df)})

        if save_predictions_prefix:
            output_df = _serialize_ukcat_eval_rows(eval_df)
            output_path = f"{save_predictions_prefix}_{approach}.csv"
            output_df.to_csv(output_path, index=False)
            click.echo(f" - saved row-level predictions: {output_path}")

    summary = pd.DataFrame(metric_rows)
    metric_order = [
        "rows",
        "precision_micro",
        "recall_micro",
        "f1_micro",
        "f1_macro",
        "subset_accuracy",
        "jaccard_samples",
        "hamming_loss",
    ]
    click.echo("Comparison metrics")
    for metric_name in metric_order:
        parts = [
            f"{row['approach']}={_format_compare_metric(float(row[metric_name]), metric_name)}"
            for _, row in summary.iterrows()
        ]
        click.echo(f" - {metric_name}: " + " | ".join(parts))

    if save_location:
        summary.to_csv(save_location, index=False)
        click.echo(f"Saved summary metrics to [{save_location}]")

    return summary
