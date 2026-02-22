import pickle
from typing import Iterable, List, Optional, Sequence, Tuple

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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from ukcat.ml_icnptso import get_text_corpus
from ukcat.settings import (
    ML_RANDOM_STATE,
    ML_TEST_TRAIN_SIZE,
    SAMPLE_FILE,
    TOP2000_FILE,
    UKCAT_ML_OVR_MODEL,
)

DEFAULT_SAMPLE_FILES = [SAMPLE_FILE, TOP2000_FILE]
DEFAULT_FIELDS = ["name", "activities"]


def _split_codes(value: object) -> List[str]:
    if not isinstance(value, str):
        return []
    return [code.strip() for code in value.split(";") if code.strip()]


def _join_codes(values: Iterable[str]) -> str:
    return ";".join(sorted(set(values)))


def _load_labelled_ukcat(sample_files: Sequence[str]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f) for f in sample_files], ignore_index=True)
    df = df[df["UKCAT"].notna()].copy()
    df = df.drop_duplicates(subset=["org_id"], keep="first")
    return df


def _build_ukcat_ovr_pipeline(n_jobs: int, ngram_max: int = 2) -> Pipeline:
    if ngram_max < 1:
        raise ValueError("ngram_max must be at least 1")
    # Parallelise at the OvR level and avoid nested parallelism in LogisticRegression.
    return Pipeline(
        [
            ("vect", CountVectorizer(ngram_range=(1, ngram_max), min_df=3)),
            ("tfidf", TfidfTransformer()),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(C=1e5, max_iter=1000, n_jobs=1, class_weight="balanced"),
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )


def _predict_codes(
    model: Pipeline,
    mlb: MultiLabelBinarizer,
    x_test: Sequence[str],
    threshold: float,
) -> Tuple[List[List[str]], pd.DataFrame]:
    probabilities = model.predict_proba(x_test)
    prob_df = pd.DataFrame(probabilities, columns=mlb.classes_)
    # Apply one shared cut-off across labels; this is easy to tune from the CLI.
    pred_binary = (prob_df.values >= threshold).astype(int)
    pred_codes = mlb.inverse_transform(pred_binary)
    pred_codes = [sorted(set(map(str, codes))) for codes in pred_codes]
    return pred_codes, prob_df


@click.command()
@click.option(
    "--sample-files",
    "-s",
    multiple=True,
    default=DEFAULT_SAMPLE_FILES,
    help="CSV files used as labelled training data",
)
@click.option(
    "--fields",
    "-f",
    multiple=True,
    default=DEFAULT_FIELDS,
    help="Fields from which to create a text corpus",
)
@click.option(
    "--save-location",
    default=UKCAT_ML_OVR_MODEL,
    help="Where the OvR UKCAT model will be saved as a pickle file",
)
@click.option(
    "--n-jobs",
    default=1,
    type=int,
    show_default=True,
    help="Number of parallel jobs used by One-vs-Rest training",
)
@click.option(
    "--clean-text/--no-clean-text",
    default=False,
    show_default=True,
    help="Apply NLP cleaning before vectorization",
)
@click.option(
    "--ngram-max",
    default=2,
    type=int,
    show_default=True,
    help="Maximum n-gram size for the OvR vectoriser (uses 1..N)",
)
def create_ukcat_ovr_model(
    sample_files: Sequence[str],
    fields: Sequence[str],
    save_location: str,
    n_jobs: int,
    clean_text: bool,
    ngram_max: int,
):
    """Train a multilabel One-vs-Rest UKCAT text classifier."""
    if n_jobs < 1:
        raise click.ClickException("--n-jobs must be at least 1")
    if ngram_max < 1:
        raise click.ClickException("--ngram-max must be at least 1")

    df = _load_labelled_ukcat(sample_files)
    click.echo(f"Loaded labelled UKCAT data [{len(df):,} rows]")

    x_all = get_text_corpus(df, fields=list(fields), do_cleaning=clean_text)
    y_codes = df["UKCAT"].astype(str).apply(_split_codes)

    mlb = MultiLabelBinarizer()
    y_all = mlb.fit_transform(y_codes)

    click.echo(f" - labels: {len(mlb.classes_):,}")
    click.echo(f" - n_jobs (OvR): {n_jobs}")
    click.echo(f" - n-grams: 1..{ngram_max} (min_df=3)")

    model = _build_ukcat_ovr_pipeline(n_jobs=n_jobs, ngram_max=ngram_max)
    click.echo("Fitting OvR model")
    model.fit(x_all, y_all)

    artifact = {
        "model": model,
        "mlb": mlb,
        "fields": list(fields),
        "clean_text": clean_text,
        "ngram_max": ngram_max,
        "min_df": 3,
        "model_type": "ukcat_ovr",
    }

    if save_location:
        click.echo(f"Saving model to [{save_location}]")
        with open(save_location, "wb") as model_file:
            pickle.dump(artifact, model_file)

    return artifact


@click.command()
@click.option(
    "--sample-files",
    "-s",
    multiple=True,
    default=DEFAULT_SAMPLE_FILES,
    help="CSV files used as gold labels for evaluation",
)
@click.option(
    "--fields",
    "-f",
    multiple=True,
    default=DEFAULT_FIELDS,
    help="Fields from which to create a text corpus",
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
    "--threshold",
    default=0.5,
    type=float,
    show_default=True,
    help="Probability threshold applied per label",
)
@click.option(
    "--n-jobs",
    default=1,
    type=int,
    show_default=True,
    help="Number of parallel jobs used by One-vs-Rest training",
)
@click.option(
    "--clean-text/--no-clean-text",
    default=False,
    show_default=True,
    help="Apply NLP cleaning before vectorization",
)
@click.option(
    "--ngram-max",
    default=2,
    type=int,
    show_default=True,
    help="Maximum n-gram size for the OvR vectoriser (uses 1..N)",
)
@click.option(
    "--save-location",
    default=None,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True),
    help="Optional CSV output with row-level evaluation details",
)
def evaluate_ukcat_ovr(
    sample_files: Sequence[str],
    fields: Sequence[str],
    random_state: int,
    test_size: float,
    threshold: float,
    n_jobs: int,
    clean_text: bool,
    ngram_max: int,
    save_location: Optional[str],
) -> pd.DataFrame:
    """Evaluate a holdout OvR UKCAT model with regex-matching multilabel metrics."""
    if n_jobs < 1:
        raise click.ClickException("--n-jobs must be at least 1")
    if not 0 < test_size < 1:
        raise click.ClickException("--test-size must be between 0 and 1")
    if not 0 <= threshold <= 1:
        raise click.ClickException("--threshold must be between 0 and 1")
    if ngram_max < 1:
        raise click.ClickException("--ngram-max must be at least 1")

    df = _load_labelled_ukcat(sample_files)
    click.echo(f"Loaded labelled data [{len(df):,} rows]")
    click.echo("Using holdout train/test evaluation mode")

    x_all = pd.Series(get_text_corpus(df, fields=list(fields), do_cleaning=clean_text))
    id_all = df["org_id"].astype(str).reset_index(drop=True)
    y_codes_all = df["UKCAT"].astype(str).apply(_split_codes).reset_index(drop=True)

    x_train, x_test, y_train_codes, y_test_codes, id_train, id_test = train_test_split(
        x_all,
        y_codes_all,
        id_all,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    click.echo(f" - training rows: {len(x_train):,}")
    click.echo(f" - test rows: {len(x_test):,}")
    click.echo(f" - threshold: {threshold:.2f}")
    click.echo(f" - n_jobs (OvR): {n_jobs}")
    click.echo(f" - n-grams: 1..{ngram_max} (min_df=3)")

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train_codes)
    model = _build_ukcat_ovr_pipeline(n_jobs=n_jobs, ngram_max=ngram_max)
    model.fit(x_train, y_train)

    y_pred_codes, _ = _predict_codes(model, mlb, x_test, threshold=threshold)

    eval_df = pd.DataFrame(
        {
            "org_id": pd.Series(id_test).values,
            "true_codes": [sorted(set(codes)) for codes in y_test_codes],
            "predicted_codes": y_pred_codes,
        }
    )

    score_mlb = MultiLabelBinarizer()
    score_mlb.fit(list(eval_df["true_codes"]) + list(eval_df["predicted_codes"]))
    y_true = score_mlb.transform(eval_df["true_codes"])
    y_pred = score_mlb.transform(eval_df["predicted_codes"])

    precision_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average="micro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    subset_accuracy = accuracy_score(y_true, y_pred)
    jaccard_samples = jaccard_score(y_true, y_pred, average="samples", zero_division=0)
    hamming = hamming_loss(y_true, y_pred)

    click.echo("Evaluation metrics")
    click.echo(f" - rows: {len(eval_df):,}")
    click.echo(f" - precision_micro: {precision_micro:.4f}")
    click.echo(f" - recall_micro: {recall_micro:.4f}")
    click.echo(f" - f1_micro: {f1_micro:.4f}")
    click.echo(f" - f1_macro: {f1_macro:.4f}")
    click.echo(f" - subset_accuracy: {subset_accuracy:.4f}")
    click.echo(f" - jaccard_samples: {jaccard_samples:.4f}")
    click.echo(f" - hamming_loss: {hamming:.4f}")

    output = eval_df.copy()
    output.loc[:, "true_codes"] = output["true_codes"].apply(_join_codes)
    output.loc[:, "predicted_codes"] = output["predicted_codes"].apply(_join_codes)
    output.loc[:, "prediction_source"] = "ml_model_holdout_ovr"

    if save_location:
        output.to_csv(save_location, index=False)
        click.echo(f"Saved row-level results to [{save_location}]")

    return output
