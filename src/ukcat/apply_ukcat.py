from typing import Optional, Sequence

import click
import pandas as pd
from tqdm import tqdm

from ukcat.apply_icnptso import MANUAL_FILES
from ukcat.ml_icnptso import get_text_corpus
from ukcat.settings import CHARITY_CSV, UKCAT_FILE


def _apply_regex_codes(
    charities: pd.DataFrame,
    ukcat_csv: str,
    id_field: str,
    fields_to_use: Sequence[str],
) -> pd.Series:
    corpus = pd.Series(
        index=charities.index,
        data=get_text_corpus(charities, fields=list(fields_to_use), do_cleaning=False),
    )
    ukcat = pd.read_csv(ukcat_csv, index_col="Code")

    results_list = []
    for index, row in tqdm(ukcat.iterrows(), total=len(ukcat)):
        if not isinstance(row["Regular expression"], str) or row["Regular expression"] == r"\b()\b":
            continue
        criteria = corpus.str.contains(row["Regular expression"], case=False, regex=True)
        if isinstance(row["Exclude regular expression"], str) and row["Exclude regular expression"] != r"\b()\b":
            criteria = criteria & ~corpus.str.contains(row["Exclude regular expression"], case=False, regex=True)

        results_list.append(
            pd.Series(
                data=index,
                index=charities[criteria].index,
                name="ukcat_code",
            )
        )

    if not results_list:
        return pd.Series(name="ukcat_code", dtype=object)
    return pd.concat(results_list)


def _apply_manual_overrides(results: pd.Series, manual_files: Sequence[str]) -> pd.Series:
    if not manual_files:
        return results
    manual_data = (
        pd.concat([pd.read_csv(f) for f in manual_files])
        .groupby("org_id")
        .first()["UKCAT"]
        .rename("manual_icnptso_code")
    )
    manual_data = manual_data[manual_data.notnull()].apply(lambda x: x.split(";")).explode()
    results = results.drop(results.index.isin(manual_data.index), errors="ignore")
    return pd.concat([results, manual_data.rename("ukcat_code")])


def _expand_group_codes(results: pd.Series) -> pd.Series:
    if results.empty:
        return results
    return pd.concat(
        [
            results,
            results.str[0:2],
            results[results.str[2].astype(int) > 1].apply(lambda x: x[0:3] + "00"),
        ]
    )


def _results_series_to_frame(results: pd.Series, id_field: str) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=[id_field, "ukcat_code"])
    return (
        results.to_frame()
        .reset_index()
        .rename(columns={"index": id_field})
        .sort_values([id_field, "ukcat_code"])
        .drop_duplicates()
    )


def _add_code_names(
    results: pd.DataFrame,
    charities: pd.DataFrame,
    ukcat_csv: str,
    id_field: str,
    name_field: str,
) -> pd.DataFrame:
    ukcat = pd.read_csv(ukcat_csv, index_col="Code")
    results = results.join(charities[name_field], on=id_field)
    return results.join(ukcat["tag"].rename("ukcat_name"), on="ukcat_code")


@click.command()
@click.option(
    "--charity-csv",
    default=CHARITY_CSV,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--ukcat-csv",
    default=UKCAT_FILE,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--id-field", default="org_id", type=str)
@click.option("--name-field", default="name", type=str)
@click.option("--fields-to-use", "-f", multiple=True, default=["name", "activities"], type=str)
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
    help="Overwrite the values for the charities in the sample with the manually found ICNPTSO from these files",
)
def apply_ukcat(
    charity_csv: str,
    ukcat_csv: str,
    id_field: str,
    name_field: str,
    fields_to_use: Sequence[str],
    save_location: Optional[str],
    sample: int,
    add_names: bool,
    include_groups: bool,
    manual_files: Sequence[str],
) -> pd.DataFrame:
    if not save_location:
        save_location = charity_csv.replace(".csv", "-ukcat-regex.csv")

    # open the charity csv file
    charities = pd.read_csv(charity_csv, index_col=id_field)
    if sample > 0:
        charities = charities.sample(sample)

    results = _apply_regex_codes(
        charities=charities,
        ukcat_csv=ukcat_csv,
        id_field=id_field,
        fields_to_use=fields_to_use,
    )
    results = _apply_manual_overrides(results, manual_files)

    # add 2-digit versions of the codes & mid-level codes
    if include_groups:
        results = _expand_group_codes(results)

    # convert data to dataframe
    results = _results_series_to_frame(results, id_field=id_field)

    # add in name and code names
    if add_names:
        results = _add_code_names(
            results=results,
            charities=charities,
            ukcat_csv=ukcat_csv,
            id_field=id_field,
            name_field=name_field,
        )

    results = results.drop_duplicates()

    # save the results
    if save_location:
        results.to_csv(save_location, index=False)

    return results
