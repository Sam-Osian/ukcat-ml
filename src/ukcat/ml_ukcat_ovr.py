import json
import pickle
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import click
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
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
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

from ukcat.ml_icnptso import get_text_corpus
from ukcat.settings import (
    ML_RANDOM_STATE,
    ML_TEST_TRAIN_SIZE,
    SAMPLE_FILE,
    TOP2000_FILE,
    UKCAT_BEST_DEV_CONFIG,
    UKCAT_ML_OVR_MODEL,
)

DEFAULT_SAMPLE_FILES = [SAMPLE_FILE, TOP2000_FILE]
DEFAULT_FIELDS = ["name", "activities"]
MODEL_FAMILY_LOGISTIC = "logistic"
MODEL_FAMILY_LINEAR_SVC = "linear_svc"
MODEL_FAMILY_SGD = "sgd"
SGD_LOSS_CHOICES = ["log_loss", "modified_huber"]
MODEL_FAMILY_CHOICES = [MODEL_FAMILY_LOGISTIC, MODEL_FAMILY_LINEAR_SVC, MODEL_FAMILY_SGD]


def _resolve_class_weight(class_weight_mode: str) -> str | None:
    if class_weight_mode == "none":
        return None
    if class_weight_mode == "balanced":
        return "balanced"
    raise ValueError(f"Unsupported class_weight_mode: {class_weight_mode}")


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


def _build_ukcat_ovr_pipeline(
    n_jobs: int,
    ngram_max: int,
    char_ngram_max: int,
    model_family: str,
    model_c: float,
    class_weight_mode: str,
    sgd_loss: str | None = None,
    sgd_alpha: float | None = None,
) -> Pipeline:
    if ngram_max < 1:
        raise ValueError("ngram_max must be at least 1")
    if char_ngram_max not in {0} and char_ngram_max < 3:
        raise ValueError("char_ngram_max must be 0 or at least 3")
    if model_family != MODEL_FAMILY_SGD and model_c <= 0:
        raise ValueError("model_c must be greater than 0")
    if model_family == MODEL_FAMILY_LOGISTIC:
        base_estimator = LogisticRegression(
            C=model_c,
            max_iter=1000,
            n_jobs=1,
            class_weight=_resolve_class_weight(class_weight_mode),
        )
    elif model_family == MODEL_FAMILY_LINEAR_SVC:
        base_estimator = LinearSVC(
            C=model_c,
            class_weight=_resolve_class_weight(class_weight_mode),
            max_iter=5000,
        )
    elif model_family == MODEL_FAMILY_SGD:
        if sgd_loss not in SGD_LOSS_CHOICES:
            raise ValueError(f"Unsupported sgd_loss: {sgd_loss}")
        if sgd_alpha is None or sgd_alpha <= 0:
            raise ValueError("sgd_alpha must be greater than 0")
        base_estimator = SGDClassifier(
            loss=sgd_loss,
            alpha=sgd_alpha,
            class_weight=_resolve_class_weight(class_weight_mode),
            max_iter=5000,
            tol=1e-3,
            random_state=ML_RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unsupported model_family: {model_family}")
    # Parallelise at the OvR level and avoid nested parallelism in LogisticRegression.
    word_features = Pipeline(
        [
            ("vect", CountVectorizer(ngram_range=(1, ngram_max), min_df=3)),
            ("tfidf", TfidfTransformer()),
        ]
    )
    if char_ngram_max > 0:
        features = FeatureUnion(
            [
                ("word", word_features),
                (
                    "char",
                    Pipeline(
                        [
                            (
                                "vect",
                                CountVectorizer(
                                    analyzer="char_wb",
                                    ngram_range=(3, char_ngram_max),
                                    min_df=3,
                                ),
                            ),
                            ("tfidf", TfidfTransformer()),
                        ]
                    ),
                ),
            ]
        )
    else:
        features = word_features
    return Pipeline(
        [
            ("features", features),
            ("clf", OneVsRestClassifier(base_estimator, n_jobs=n_jobs)),
        ]
    )


def _load_best_dev_config(best_config_path: str) -> dict[str, Any]:
    path = Path(best_config_path)
    if not path.exists():
        raise click.ClickException(
            f"Best dev config not found at [{path}]. Run `ukcat evaluate dev-grid` first."
        )
    with open(path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def _get_selected_model_config(best_config_path: str, approach: str) -> dict[str, Any]:
    config = _load_best_dev_config(best_config_path)
    model_config = config.get(approach)
    if model_config is None:
        raise click.ClickException(
            f"Best dev config [{best_config_path}] does not contain a selected [{approach}] config."
        )
    return model_config


def _train_ukcat_ovr_artifact(
    sample_files: Sequence[str],
    fields: Sequence[str],
    save_location: str,
    n_jobs: int,
    clean_text: bool,
    ngram_max: int,
    char_ngram_max: int,
    model_family: str,
    model_c: float,
    class_weight_mode: str,
    sgd_loss: str | None,
    sgd_alpha: float | None,
    threshold: float,
    top_k_fallback: int,
    model_type: str = "ukcat_ovr",
    extra_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if n_jobs < 1:
        raise click.ClickException("--n-jobs must be at least 1")
    if ngram_max < 1:
        raise click.ClickException("--ngram-max must be at least 1")
    if char_ngram_max not in {0} and char_ngram_max < 3:
        raise click.ClickException("--char-ngram-max must be 0 or at least 3")
    if model_family not in MODEL_FAMILY_CHOICES:
        raise click.ClickException(
            f"--model-family must be one of: {', '.join(MODEL_FAMILY_CHOICES)}"
        )
    if model_family != MODEL_FAMILY_SGD and model_c <= 0:
        raise click.ClickException("--model-c must be greater than 0")
    if class_weight_mode not in {"none", "balanced"}:
        raise click.ClickException("--class-weight-mode must be one of: none, balanced")
    if model_family == MODEL_FAMILY_SGD:
        if sgd_loss not in SGD_LOSS_CHOICES:
            raise click.ClickException(f"--sgd-loss must be one of: {', '.join(SGD_LOSS_CHOICES)}")
        if sgd_alpha is None or sgd_alpha <= 0:
            raise click.ClickException("--sgd-alpha must be greater than 0 for SGD models")

    df = _load_labelled_ukcat(sample_files)
    click.echo(f"Loaded labelled UKCAT data [{len(df):,} rows]")

    x_all = get_text_corpus(df, fields=list(fields), do_cleaning=clean_text)
    y_codes = df["UKCAT"].astype(str).apply(_split_codes)

    mlb = MultiLabelBinarizer()
    y_all = mlb.fit_transform(y_codes)

    click.echo(f" - labels: {len(mlb.classes_):,}")
    click.echo(f" - n_jobs (OvR): {n_jobs}")
    click.echo(f" - n-grams: 1..{ngram_max} (min_df=3)")
    click.echo(
        " - char n-grams: "
        + (f"char_wb 3..{char_ngram_max} (min_df=3)" if char_ngram_max > 0 else "off")
    )
    click.echo(f" - model_family: {model_family}")
    if model_family == MODEL_FAMILY_SGD:
        click.echo(f" - sgd_loss: {sgd_loss}")
        click.echo(f" - sgd_alpha: {sgd_alpha:g}")
    else:
        click.echo(f" - model_c: {model_c:g}")
    click.echo(f" - class_weight_mode: {class_weight_mode}")
    click.echo(f" - threshold: {threshold:.3f}")
    click.echo(f" - top_k_fallback: {top_k_fallback}")

    model = _build_ukcat_ovr_pipeline(
        n_jobs=n_jobs,
        ngram_max=ngram_max,
        char_ngram_max=char_ngram_max,
        model_family=model_family,
        model_c=model_c,
        class_weight_mode=class_weight_mode,
        sgd_loss=sgd_loss,
        sgd_alpha=sgd_alpha,
    )
    click.echo("Fitting OvR model")
    model.fit(x_all, y_all)

    artifact = {
        "model": model,
        "mlb": mlb,
        "fields": list(fields),
        "clean_text": clean_text,
        "ngram_max": ngram_max,
        "char_ngram_max": char_ngram_max,
        "model_family": model_family,
        "model_c": model_c,
        "class_weight_mode": class_weight_mode,
        "sgd_loss": sgd_loss,
        "sgd_alpha": sgd_alpha,
        "min_df": 3,
        "threshold": threshold,
        "top_k_fallback": top_k_fallback,
        "model_type": model_type,
    }
    if extra_metadata:
        artifact.update(extra_metadata)

    if save_location:
        click.echo(f"Saving model to [{save_location}]")
        with open(save_location, "wb") as model_file:
            pickle.dump(artifact, model_file)

    return artifact


def _predict_codes(
    model: Pipeline,
    mlb: MultiLabelBinarizer,
    x_test: Sequence[str],
    threshold: float,
    top_k_fallback: int = 0,
) -> Tuple[List[List[str]], pd.DataFrame]:
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x_test)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(x_test)
    else:
        raise ValueError("Model must provide predict_proba or decision_function")
    prob_df = pd.DataFrame(scores, columns=mlb.classes_)
    # Apply one shared cut-off across labels; this is easy to tune from the CLI.
    pred_binary = (prob_df.values >= threshold).astype(int)
    if top_k_fallback > 0 and pred_binary.shape[1] > 0:
        # If a row gets no labels at the threshold, back-fill with the top-k scores.
        k = min(top_k_fallback, pred_binary.shape[1])
        for i in range(pred_binary.shape[0]):
            if pred_binary[i].sum() > 0:
                continue
            top_idx = prob_df.iloc[i].to_numpy().argsort()[-k:]
            pred_binary[i, top_idx] = 1
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
    "--best-config",
    default=UKCAT_BEST_DEV_CONFIG,
    show_default=True,
    help="Best dev config JSON created by `ukcat evaluate dev-grid`.",
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
def create_ukcat_ovr_model(
    sample_files: Sequence[str],
    best_config: str,
    save_location: str,
    n_jobs: int,
):
    """Train the final selected OvR UKCAT model on all labelled data."""
    model_config = _get_selected_model_config(best_config, "ovr_logistic")
    return _train_ukcat_ovr_artifact(
        sample_files=sample_files,
        fields=tuple(model_config["fields"]),
        save_location=save_location,
        n_jobs=n_jobs,
        clean_text=bool(model_config["clean_text"]),
        ngram_max=int(model_config["ngram_max"]),
        char_ngram_max=int(model_config.get("char_ngram_max", 0)),
        model_family=str(model_config["model_family"]),
        model_c=float(model_config.get("model_c", -1.0)),
        class_weight_mode=str(model_config["class_weight_mode"]),
        sgd_loss=model_config.get("sgd_loss"),
        sgd_alpha=(
            float(model_config["sgd_alpha"])
            if model_config.get("sgd_alpha") is not None
            else None
        ),
        threshold=float(model_config["threshold"]),
        top_k_fallback=int(model_config["top_k_fallback"]),
        model_type="ukcat_ovr",
        extra_metadata={
            "selected_by": model_config["selected_by"],
            "best_config_path": best_config,
        },
    )


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
    "--top-k-fallback",
    default=0,
    type=int,
    show_default=True,
    help="If no labels pass threshold, assign the top-k labels by probability",
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
    "--char-ngram-max",
    default=0,
    type=int,
    show_default=True,
    help="Maximum char_wb n-gram size for an optional character feature branch (0 disables)",
)
@click.option(
    "--model-family",
    required=True,
    type=click.Choice(MODEL_FAMILY_CHOICES),
    help="Base estimator family used by One-vs-Rest",
)
@click.option(
    "--model-c",
    required=False,
    type=float,
    help="Inverse regularisation strength used by logistic or linear_svc",
)
@click.option(
    "--class-weight-mode",
    required=True,
    type=click.Choice(["none", "balanced"]),
    help="Class weighting strategy used by the selected model family",
)
@click.option(
    "--sgd-loss",
    required=False,
    type=click.Choice(SGD_LOSS_CHOICES),
    help="Loss used by SGDClassifier",
)
@click.option(
    "--sgd-alpha",
    required=False,
    type=float,
    help="Regularisation strength used by SGDClassifier",
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
    top_k_fallback: int,
    n_jobs: int,
    clean_text: bool,
    ngram_max: int,
    char_ngram_max: int,
    model_family: str,
    model_c: Optional[float],
    class_weight_mode: str,
    sgd_loss: Optional[str],
    sgd_alpha: Optional[float],
    save_location: Optional[str],
) -> pd.DataFrame:
    """Evaluate a holdout OvR UKCAT model with regex-matching multilabel metrics."""
    if n_jobs < 1:
        raise click.ClickException("--n-jobs must be at least 1")
    if not 0 < test_size < 1:
        raise click.ClickException("--test-size must be between 0 and 1")
    if top_k_fallback < 0:
        raise click.ClickException("--top-k-fallback must be 0 or greater")
    if ngram_max < 1:
        raise click.ClickException("--ngram-max must be at least 1")
    if char_ngram_max not in {0} and char_ngram_max < 3:
        raise click.ClickException("--char-ngram-max must be 0 or at least 3")
    if model_family == MODEL_FAMILY_LOGISTIC and not 0 <= threshold <= 1:
        raise click.ClickException("--threshold must be between 0 and 1 for logistic models")
    if model_family != MODEL_FAMILY_SGD:
        if model_c is None or model_c <= 0:
            raise click.ClickException("--model-c must be greater than 0 for logistic or linear_svc models")
    if model_family == MODEL_FAMILY_SGD:
        if sgd_loss not in SGD_LOSS_CHOICES:
            raise click.ClickException(f"--sgd-loss must be one of: {', '.join(SGD_LOSS_CHOICES)}")
        if sgd_alpha is None or sgd_alpha <= 0:
            raise click.ClickException("--sgd-alpha must be greater than 0 for SGD models")

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
    click.echo(f" - top-k fallback: {top_k_fallback}")
    click.echo(f" - n_jobs (OvR): {n_jobs}")
    click.echo(f" - n-grams: 1..{ngram_max} (min_df=3)")
    click.echo(
        " - char n-grams: "
        + (f"char_wb 3..{char_ngram_max} (min_df=3)" if char_ngram_max > 0 else "off")
    )
    click.echo(f" - model_family: {model_family}")
    if model_family == MODEL_FAMILY_SGD:
        click.echo(f" - sgd_loss: {sgd_loss}")
        click.echo(f" - sgd_alpha: {sgd_alpha:g}")
    else:
        click.echo(f" - model_c: {float(model_c):g}")
    click.echo(f" - class_weight_mode: {class_weight_mode}")

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train_codes)
    model = _build_ukcat_ovr_pipeline(
        n_jobs=n_jobs,
        ngram_max=ngram_max,
        char_ngram_max=char_ngram_max,
        model_family=model_family,
        model_c=float(model_c) if model_c is not None else -1.0,
        class_weight_mode=class_weight_mode,
        sgd_loss=sgd_loss,
        sgd_alpha=sgd_alpha,
    )
    model.fit(x_train, y_train)

    y_pred_codes, _ = _predict_codes(model, mlb, x_test, threshold=threshold, top_k_fallback=top_k_fallback)

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
