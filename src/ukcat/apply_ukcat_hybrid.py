import json
import pickle
from pathlib import Path
from typing import Optional, Sequence

import click
import pandas as pd

from ukcat.apply_icnptso import MANUAL_FILES
from ukcat.apply_ukcat import (
    _add_code_names,
    _apply_manual_overrides,
    _apply_regex_codes,
    _expand_group_codes,
    _results_series_to_frame,
)
from ukcat.ml_icnptso import get_text_corpus
from ukcat.ml_ukcat_hybrid import combine_hybrid_predictions
from ukcat.ml_ukcat_ovr import _predict_codes
from ukcat.settings import CHARITY_CSV, UKCAT_FILE, UKCAT_ML_HYBRID_CONFIG


@click.command()
@click.option(
    "--charity-csv",
    default=CHARITY_CSV,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--hybrid-config",
    default=UKCAT_ML_HYBRID_CONFIG,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--ukcat-csv",
    default=UKCAT_FILE,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--id-field", default="org_id", type=str)
@click.option("--name-field", default="name", type=str)
@click.option(
    "--save-location",
    default=None,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True),
)
@click.option(
    "--sample",
    default=0,
    type=int,
    help="Only do a sample of the charities (for testing purposes)",
)
@click.option(
    "--add-names/--no-add-names",
    default=False,
    help="Add the charity and category names to the data",
)
@click.option(
    "--manual-files",
    "-m",
    multiple=True,
    default=MANUAL_FILES,
    type=str,
    help="Overwrite the values for charities found in the labelled sample files",
)
def apply_ukcat_hybrid(
    charity_csv: str,
    hybrid_config: str,
    ukcat_csv: str,
    id_field: str,
    name_field: str,
    save_location: Optional[str],
    sample: int,
    add_names: bool,
    manual_files: Sequence[str],
) -> pd.DataFrame:
    """Apply the trained UK-CAT hybrid model to a charity CSV."""
    if not save_location:
        save_location = charity_csv.replace(".csv", "-ukcat-hybrid.csv")

    charities = pd.read_csv(charity_csv, index_col=id_field)
    if sample > 0:
        charities = charities.sample(sample)

    cfg = json.loads(Path(hybrid_config).read_text(encoding="utf-8"))
    with open(cfg["ovr_model_path"], "rb") as model_file:
        ovr_art = pickle.load(model_file)

    regex_codes = _apply_regex_codes(
        charities=charities,
        ukcat_csv=ukcat_csv,
        id_field=id_field,
        fields_to_use=ovr_art["fields"],
    )
    if bool(cfg.get("include_groups", False)):
        regex_codes = _expand_group_codes(regex_codes)
    regex_df = _results_series_to_frame(regex_codes, id_field=id_field)
    regex_df = regex_df.groupby(id_field)["ukcat_code"].apply(list).reset_index()
    regex_df["predicted_codes"] = regex_df["ukcat_code"].apply(lambda codes: sorted(set(map(str, codes))))
    regex_df = regex_df.drop(columns=["ukcat_code"])
    regex_df["prediction_source"] = "regex_rules"

    corpus = get_text_corpus(
        charities,
        fields=list(ovr_art["fields"]),
        do_cleaning=bool(ovr_art.get("clean_text", False)),
    )
    ovr_codes, prob_df = _predict_codes(
        ovr_art["model"],
        ovr_art["mlb"],
        corpus,
        threshold=float(ovr_art.get("threshold", 0.5)),
        top_k_fallback=int(ovr_art.get("top_k_fallback", 0)),
    )
    ovr_df = pd.DataFrame(
        {
            "org_id": charities.index.astype(str),
            "true_codes": [[] for _ in range(len(charities))],
            "predicted_codes": ovr_codes,
            "prediction_source": "ml_model_ovr",
        }
    )
    prob_df.index = charities.index.astype(str)
    regex_map = regex_df.set_index("org_id")["predicted_codes"]
    regex_pred = [
        value if isinstance(value, list) else []
        for value in regex_map.reindex(charities.index.astype(str)).tolist()
    ]
    # Regex emits sparse row-per-code output; rebuild a full per-row list so the
    # hybrid combiner can align regex and OvR decisions by charity.
    regex_df = pd.DataFrame(
        {
            "org_id": charities.index.astype(str),
            "predicted_codes": regex_pred,
            "prediction_source": "regex_rules",
        }
    )

    hybrid_df = combine_hybrid_predictions(
        ovr_eval_df=ovr_df,
        regex_eval_df=regex_df,
        ovr_probability_df=prob_df,
        rule=str(cfg["hybrid_rule"]),
        label_confidence_threshold=float(cfg["hybrid_conf"]),
    )

    # Keep the same row-per-code output contract as regex and OVR apply.
    codes = pd.Series(
        data=[code for codes in hybrid_df["predicted_codes"] for code in codes],
        index=[org_id for org_id, codes in zip(charities.index, hybrid_df["predicted_codes"]) for _ in codes],
        name="ukcat_code",
        dtype=object,
    )
    codes = _apply_manual_overrides(codes, manual_files)

    out_df = _results_series_to_frame(codes, id_field=id_field)
    if add_names:
        out_df = _add_code_names(
            results=out_df,
            charities=charities,
            ukcat_csv=ukcat_csv,
            id_field=id_field,
            name_field=name_field,
        )

    out_df = out_df.drop_duplicates()
    if save_location:
        out_df.to_csv(save_location, index=False)
    return out_df
