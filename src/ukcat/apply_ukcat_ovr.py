import pickle
from typing import Optional, Sequence

import click
import pandas as pd

from ukcat.apply_icnptso import MANUAL_FILES
from ukcat.apply_ukcat import (
    _add_code_names,
    _apply_manual_overrides,
    _expand_group_codes,
    _results_series_to_frame,
)
from ukcat.ml_icnptso import get_text_corpus
from ukcat.ml_ukcat_ovr import _predict_codes
from ukcat.settings import CHARITY_CSV, UKCAT_FILE, UKCAT_ML_OVR_MODEL


@click.command()
@click.option(
    "--charity-csv",
    default=CHARITY_CSV,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--ovr-model",
    default=UKCAT_ML_OVR_MODEL,
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
    "--include-groups/--no-include-groups",
    default=False,
    help="Add codes for the intermediate groups to the results",
)
@click.option(
    "--manual-files",
    "-m",
    multiple=True,
    default=MANUAL_FILES,
    type=str,
    help="Overwrite the values for charities found in the labelled sample files",
)
def apply_ukcat_ovr(
    charity_csv: str,
    ovr_model: str,
    ukcat_csv: str,
    id_field: str,
    name_field: str,
    save_location: Optional[str],
    sample: int,
    add_names: bool,
    include_groups: bool,
    manual_files: Sequence[str],
) -> pd.DataFrame:
    """Apply the trained UK-CAT OVR model to a charity CSV."""
    if not save_location:
        save_location = charity_csv.replace(".csv", "-ukcat-ovr.csv")

    charities = pd.read_csv(charity_csv, index_col=id_field)
    if sample > 0:
        charities = charities.sample(sample)

    with open(ovr_model, "rb") as model_file:
        art = pickle.load(model_file)

    corpus = get_text_corpus(
        charities,
        fields=list(art["fields"]),
        do_cleaning=bool(art.get("clean_text", False)),
    )
    pred_codes, _ = _predict_codes(
        art["model"],
        art["mlb"],
        corpus,
        threshold=float(art.get("threshold", 0.5)),
        top_k_fallback=int(art.get("top_k_fallback", 0)),
    )

    # The apply output stays in the repo's row-per-code shape so downstream
    # consumers can treat regex, OVR, and hybrid outputs the same way.
    codes = pd.Series(
        data=[code for codes in pred_codes for code in codes],
        index=[org_id for org_id, codes in zip(charities.index, pred_codes) for _ in codes],
        name="ukcat_code",
        dtype=object,
    )
    codes = _apply_manual_overrides(codes, manual_files)
    if include_groups:
        codes = _expand_group_codes(codes)

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
