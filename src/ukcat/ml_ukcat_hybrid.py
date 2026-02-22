from typing import Dict, Iterable, List

import pandas as pd

HYBRID_RULE_OVR_REGEX_FALLBACK = "ovr_regex_fallback"
HYBRID_RULE_HIGH_CONF_UNION = "high_conf_union"
HYBRID_RULE_LABEL_CONF_GATED_REGEX = "label_conf_gated_regex"

HYBRID_RULE_CHOICES = [
    HYBRID_RULE_OVR_REGEX_FALLBACK,
    HYBRID_RULE_HIGH_CONF_UNION,
    HYBRID_RULE_LABEL_CONF_GATED_REGEX,
]

# Sensible defaults for the first hybrid experiments.
DEFAULT_HYBRID_FALLBACK_MAX_OVR_LABELS = 1
DEFAULT_HYBRID_ROW_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_HYBRID_LABEL_CONFIDENCE_THRESHOLD = 0.03


def _normalise_codes(value: object) -> List[str]:
    if isinstance(value, list):
        return sorted(set(str(v) for v in value))
    return []


def _union_codes(a: Iterable[str], b: Iterable[str]) -> List[str]:
    return sorted(set(map(str, a)).union(set(map(str, b))))


def combine_hybrid_predictions(
    ovr_eval_df: pd.DataFrame,
    regex_eval_df: pd.DataFrame,
    ovr_probability_df: pd.DataFrame,
    rule: str,
    fallback_max_ovr_labels: int = DEFAULT_HYBRID_FALLBACK_MAX_OVR_LABELS,
    row_confidence_threshold: float = DEFAULT_HYBRID_ROW_CONFIDENCE_THRESHOLD,
    label_confidence_threshold: float = DEFAULT_HYBRID_LABEL_CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    """
    Combine OVR and regex predictions using a hybrid decision rule.

    Expected inputs:
    - ovr_eval_df / regex_eval_df columns: org_id, true_codes, predicted_codes
    - ovr_probability_df index: org_id, columns: UKCAT codes, values: probabilities
    """
    if rule not in HYBRID_RULE_CHOICES:
        raise ValueError(f"Unsupported hybrid rule: {rule}")

    ovr_map = (
        ovr_eval_df.set_index("org_id")[["true_codes", "predicted_codes"]]
        .rename(columns={"predicted_codes": "ovr_predicted_codes"})
        .copy()
    )
    regex_map = (
        regex_eval_df.set_index("org_id")[["predicted_codes"]]
        .rename(columns={"predicted_codes": "regex_predicted_codes"})
        .copy()
    )

    combined = ovr_map.join(regex_map, how="left")
    combined.loc[:, "ovr_predicted_codes"] = combined["ovr_predicted_codes"].apply(_normalise_codes)
    combined.loc[:, "regex_predicted_codes"] = combined["regex_predicted_codes"].apply(_normalise_codes)

    final_codes: List[List[str]] = []
    for org_id, row in combined.iterrows():
        ovr_codes = row["ovr_predicted_codes"]
        regex_codes = row["regex_predicted_codes"]
        if org_id in ovr_probability_df.index:
            prob_row = ovr_probability_df.loc[org_id]
            if isinstance(prob_row, pd.DataFrame):
                prob_row = prob_row.iloc[0]
        else:
            prob_row = pd.Series(dtype=float)

        if rule == HYBRID_RULE_OVR_REGEX_FALLBACK:
            if len(ovr_codes) <= fallback_max_ovr_labels:
                final_codes.append(_union_codes(ovr_codes, regex_codes))
            else:
                final_codes.append(ovr_codes)
            continue

        if rule == HYBRID_RULE_HIGH_CONF_UNION:
            row_max = float(prob_row.max()) if len(prob_row.index) > 0 else 0.0
            if row_max >= row_confidence_threshold:
                final_codes.append(_union_codes(ovr_codes, regex_codes))
            else:
                final_codes.append(ovr_codes)
            continue

        # HYBRID_RULE_LABEL_CONF_GATED_REGEX:
        gated_regex_codes = []
        for code in regex_codes:
            score = float(prob_row.get(code, 0.0))
            if score >= label_confidence_threshold:
                gated_regex_codes.append(code)
        final_codes.append(_union_codes(ovr_codes, gated_regex_codes))

    output = pd.DataFrame(
        {
            "org_id": combined.index.astype(str),
            "true_codes": combined["true_codes"].tolist(),
            "predicted_codes": final_codes,
            "prediction_source": f"hybrid_{rule}",
        }
    )
    return output.reset_index(drop=True)
