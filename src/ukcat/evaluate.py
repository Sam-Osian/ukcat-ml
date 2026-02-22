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
from ukcat.settings import (
    ML_DEFAULT_FIELDS,
    ML_RANDOM_STATE,
    ML_TEST_TRAIN_SIZE,
    SAMPLE_FILE,
    TOP2000_FILE,
    UKCAT_FILE,
)

DEFAULT_SAMPLE_FILES = [SAMPLE_FILE, TOP2000_FILE]


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
    default=ML_RANDOM_STATE,
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
    "--save-location",
    default=None,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True),
    help="Optional CSV output with row-level evaluation details",
)
def evaluate_ukcat(
    sample_files: Sequence[str],
    include_groups: bool,
    save_location: Optional[str],
) -> pd.DataFrame:
    """Evaluate UKCAT regex tagging against labelled sample files."""
    id_field = "org_id"
    category_field = "UKCAT"
    fields_to_use = ["name", "activities"]

    df = pd.concat([pd.read_csv(f) for f in sample_files], ignore_index=True)
    labelled = df[df[category_field].notna()].copy()

    click.echo(f"Loaded labelled data [{len(labelled):,} rows]")

    charities = labelled.drop_duplicates(subset=[id_field], keep="first").copy()
    charities = charities.set_index(id_field)
    corpus = pd.Series(
        index=charities.index,
        data=get_text_corpus(charities, fields=fields_to_use, do_cleaning=False),
    )

    ukcat = pd.read_csv(UKCAT_FILE, index_col="Code")
    results_list = []
    for code, row in ukcat.iterrows():
        if not isinstance(row["Regular expression"], str) or row["Regular expression"] == r"\b()\b":
            continue
        with warnings.catch_warnings():
            # Many project regexes use capture groups; pandas warns on this in str.contains,
            # but matching behaviour is still correct for boolean filtering
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
        results_list.append(pd.Series(data=code, index=charities[criteria].index, name="ukcat_code"))

    if results_list:
        predicted = pd.concat(results_list)
    else:
        predicted = pd.Series(dtype=object, name="ukcat_code")

    if include_groups:
        predicted = pd.concat(
            [
                predicted,
                predicted.str[0:2],
                predicted[predicted.str[2].astype(int) > 1].apply(lambda x: x[0:3] + "00"),
            ]
        )

    pred_map = predicted.groupby(level=0).apply(lambda x: sorted(set(x.astype(str)))).rename("predicted_codes")
    true_map = (
        labelled.groupby(id_field)[category_field]
        .first()
        .apply(_split_codes)
        .apply(lambda x: sorted(set(x)))
        .rename("true_codes")
    )

    eval_df = true_map.to_frame().join(pred_map, how="left")
    eval_df.loc[:, "predicted_codes"] = eval_df["predicted_codes"].apply(lambda x: x if isinstance(x, list) else [])

    mlb = MultiLabelBinarizer()
    all_labels = list(eval_df["true_codes"]) + list(eval_df["predicted_codes"])
    mlb.fit(all_labels)

    y_true = mlb.transform(eval_df["true_codes"])
    y_pred = mlb.transform(eval_df["predicted_codes"])

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

    output = eval_df.reset_index()
    output.loc[:, "true_codes"] = output["true_codes"].apply(_join_codes)
    output.loc[:, "predicted_codes"] = output["predicted_codes"].apply(_join_codes)

    if save_location:
        output.to_csv(save_location, index=False)
        click.echo(f"Saved row-level results to [{save_location}]")

    return output
