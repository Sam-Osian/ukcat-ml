import json
from pathlib import Path
from typing import Sequence

import click

from ukcat.ml_ukcat_ovr import _get_selected_model_config, _train_ukcat_ovr_artifact
from ukcat.settings import (
    SAMPLE_FILE,
    TOP2000_FILE,
    UKCAT_BEST_DEV_CONFIG,
    UKCAT_ML_HYBRID_CONFIG,
    UKCAT_ML_HYBRID_OVR_MODEL,
)

DEFAULT_SAMPLE_FILES = [SAMPLE_FILE, TOP2000_FILE]


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
    "--ovr-save-location",
    default=UKCAT_ML_HYBRID_OVR_MODEL,
    show_default=True,
    help="Where the hybrid's OVR model artifact will be saved.",
)
@click.option(
    "--save-location",
    default=UKCAT_ML_HYBRID_CONFIG,
    show_default=True,
    help="Where the hybrid config JSON will be saved.",
)
@click.option(
    "--n-jobs",
    default=1,
    type=int,
    show_default=True,
    help="Number of parallel jobs used by One-vs-Rest training",
)
def create_ukcat_hybrid_model(
    sample_files: Sequence[str],
    best_config: str,
    ovr_save_location: str,
    save_location: str,
    n_jobs: int,
):
    """Train the final selected hybrid UKCAT model on all labelled data."""
    cfg = _get_selected_model_config(best_config, "hybrid")
    _train_ukcat_ovr_artifact(
        sample_files=sample_files,
        fields=tuple(cfg["fields"]),
        save_location=ovr_save_location,
        n_jobs=n_jobs,
        clean_text=bool(cfg["clean_text"]),
        ngram_max=int(cfg["ngram_max"]),
        threshold=float(cfg["threshold"]),
        top_k_fallback=int(cfg["top_k_fallback"]),
        model_type="ukcat_hybrid_ovr",
        extra_metadata={
            "selected_by": cfg["selected_by"],
            "best_config_path": best_config,
        },
    )

    art = {
        "model_type": "ukcat_hybrid",
        "best_config_path": best_config,
        "ovr_model_path": ovr_save_location,
        "fields": list(cfg["fields"]),
        "threshold": float(cfg["threshold"]),
        "top_k_fallback": int(cfg["top_k_fallback"]),
        "clean_text": bool(cfg["clean_text"]),
        "ngram_max": int(cfg["ngram_max"]),
        "include_groups": bool(cfg["include_groups"]),
        "hybrid_rule": cfg["hybrid_rule"],
        "hybrid_conf": float(cfg["hybrid_conf"]),
        "selected_by": cfg["selected_by"],
    }

    out_path = Path(save_location)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    click.echo(f"Saving hybrid config to [{out_path}]")
    out_path.write_text(json.dumps(art, indent=2) + "\n", encoding="utf-8")
    return art
