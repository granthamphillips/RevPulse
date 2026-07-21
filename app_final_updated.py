"""
RevPulse Streamlit predictor for the fitted Elastic Net model.

Required files in the same directory:
    elastic_net_streamlit_bundle.pkl
    airroi_comparables.csv

The bundle must contain the fitted model, imputer, scaler, training feature
schema, category metadata, and amenity group definitions. Use the companion
export cell generated for the modeling notebook.
"""

from __future__ import annotations

import base64
import pickle
import re
from html import escape
from pathlib import Path
from typing import Any

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn
import streamlit as st
from streamlit_folium import st_folium


APP_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = APP_DIR / "elastic_net_streamlit_bundle.pkl"
COMPARABLES_PATH = APP_DIR / "airroi_comparables.csv"
LOGO_PATH = APP_DIR / "revpulse_logo.png"

COLOR_POSITIVE = "#2f9e62"
COLOR_NEGATIVE = "#b83232"
COLOR_NEUTRAL = "#9ca3af"

CITY_REVPAR_MEDIANS = {
    "carolina_beach": 100.8,
    "gatlinburg": 97.4,
    "wilmington": 73.6,
    "charlotte": 71.7,
    "myrtle_beach": 65.6,
    "raleigh": 65.2,
    "durham": 61.6,
    "asheville": 59.0,
    "pigeon_forge": 54.8,
    "williamsburg": 33.4,
}


st.set_page_config(
    page_title="RevPulse - Short-Term Rental Market Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html,
        body,
        [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .block-container {
            padding: 2rem 2.5rem;
            max-width: 1240px;
        }

        .brand-header {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            margin: 0 auto 1.35rem auto;
        }

        .brand-logo {
            display: block;
            width: min(570px, 92vw);
            max-width: 100%;
            height: auto;
            filter: drop-shadow(0 2px 8px rgba(0, 0, 0, .14));
        }

        .brand-tagline {
            margin-top: .22rem;
            font-size: .98rem;
            font-weight: 500;
            color: inherit;
            opacity: .72;
            letter-spacing: .05px;
        }

        div[data-testid="stExpander"] details {
            border: 1px solid rgba(127, 127, 127, .30);
            border-radius: 10px;
            background: rgba(127, 127, 127, .045);
        }

        div[data-testid="stExpander"] summary {
            font-weight: 650;
        }

        .amenity-group-title {
            font-size: .82rem;
            font-weight: 700;
            margin-bottom: .5rem;
        }

        .section-label {
            font-size: .69rem;
            font-weight: 600;
            letter-spacing: 1.1px;
            text-transform: uppercase;
            color: inherit;
            opacity: .64;
            margin: 1.3rem 0 .55rem 0;
        }

        .hero-value {
            font-size: 3.25rem;
            font-weight: 800;
            color: #2f9e62;
            line-height: 1;
            letter-spacing: -1.5px;
        }

        .hero-label {
            font-size: .8rem;
            font-weight: 500;
            color: inherit;
            opacity: .67;
            text-transform: uppercase;
            letter-spacing: .75px;
            margin-top: .3rem;
        }

        .hero-context {
            font-size: .95rem;
            color: inherit;
            opacity: .70;
            margin-top: .6rem;
        }

        .revenue-card {
            box-sizing: border-box;
            padding: 1.15rem 1.2rem;
            background:
                linear-gradient(
                    135deg,
                    rgba(47, 158, 98, .11),
                    rgba(127, 127, 127, .08)
                );
            color: inherit;
            border: 1px solid rgba(47, 158, 98, .72);
            border-radius: 12px;
            margin-top: 1rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, .12);
        }

        .revenue-eyebrow {
            font-size: .68rem;
            font-weight: 700;
            letter-spacing: 1.05px;
            text-transform: uppercase;
            color: #2f9e62;
            margin-bottom: .35rem;
        }

        .revenue-main {
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.05;
            letter-spacing: -.5px;
            color: inherit;
        }

        .revenue-sub {
            font-size: .75rem;
            color: inherit;
            opacity: .72;
            margin-top: .25rem;
        }

        .revenue-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: .65rem;
            margin-top: .95rem;
        }

        .revenue-stat {
            box-sizing: border-box;
            min-width: 0;
            padding: .7rem .75rem;
            background: rgba(127, 127, 127, .09);
            border: 1px solid rgba(127, 127, 127, .28);
            border-radius: 8px;
            color: inherit;
        }

        .revenue-stat-value {
            font-size: 1.02rem;
            font-weight: 700;
            color: inherit;
        }

        .revenue-stat-label {
            font-size: .65rem;
            color: inherit;
            opacity: .66;
            text-transform: uppercase;
            letter-spacing: .65px;
            margin-top: .12rem;
            overflow-wrap: anywhere;
        }

        .revenue-context {
            font-size: .72rem;
            color: inherit;
            opacity: .67;
            margin-top: .8rem;
            overflow-wrap: anywhere;
        }

        .input-note {
            font-size: .75rem;
            color: inherit;
            opacity: .68;
            margin-top: .15rem;
        }

        .rule {
            border: none;
            border-top: 1px solid rgba(127, 127, 127, .28);
            margin: 1.35rem 0;
        }

        div[data-testid="stCheckbox"] label {
            font-size: .82rem !important;
            color: inherit !important;
        }

        div[data-testid="stSelectbox"] label,
        div[data-testid="stNumberInput"] label {
            font-size: .84rem !important;
            font-weight: 500 !important;
            color: inherit !important;
        }

        .opportunity-grid {
            display: grid;
            grid-template-columns: repeat(
                auto-fit,
                minmax(260px, 1fr)
            );
            gap: 1.1rem;
            align-items: stretch;
            margin: .8rem 0 2.2rem 0;
        }

        .opportunity-card {
            box-sizing: border-box;
            min-width: 0;
            min-height: 285px;
            height: 100%;
            padding: 1.05rem 1.1rem;
            display: flex;
            flex-direction: column;
            background: rgba(127, 127, 127, .10);
            color: inherit;
            border: 1px solid rgba(127, 127, 127, .30);
            border-radius: 11px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, .10);
            overflow: hidden;
        }

        .opportunity-rank {
            font-size: .65rem;
            font-weight: 700;
            letter-spacing: .9px;
            text-transform: uppercase;
            color: inherit;
            opacity: .62;
            margin-bottom: .4rem;
        }

        .opportunity-title {
            font-size: 1rem;
            font-weight: 700;
            color: inherit;
            line-height: 1.28;
            min-height: 2.56em;
            overflow-wrap: anywhere;
        }

        .opportunity-impact {
            font-size: 1rem;
            font-weight: 700;
            color: #2f9e62;
            margin-top: .75rem;
            overflow-wrap: anywhere;
        }

        .opportunity-annual {
            font-size: .73rem;
            color: inherit;
            opacity: .72;
            margin-top: .15rem;
            overflow-wrap: anywhere;
        }

        .opportunity-detail {
            font-size: .76rem;
            line-height: 1.48;
            color: inherit;
            opacity: .68;
            margin-top: auto;
            padding-top: .9rem;
            overflow-wrap: anywhere;
        }

        .footer {
            font-size: .72rem;
            color: inherit;
            opacity: .66;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(127, 127, 127, .28);
        }

        @media (max-width: 760px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .revenue-grid {
                grid-template-columns: 1fr;
            }

            .opportunity-grid {
                grid-template-columns: 1fr;
            }

            .opportunity-card {
                min-height: 0;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_bundle(path: Path) -> dict[str, Any]:
    if not path.exists():
        st.error(
            "elastic_net_streamlit_bundle.pkl was not found. "
            "Run the supplied export cell in the modeling notebook, then place "
            "the downloaded bundle beside app.py."
        )
        st.stop()

    try:
        with path.open("rb") as handle:
            bundle = pickle.load(handle)
    except Exception as exc:
        st.error(f"Could not load the model bundle: {exc}")
        st.stop()

    required = {
        "model",
        "imputer",
        "scaler",
        "feature_names",
        "amenity_groups",
        "standalone_amenities",
        "property_reference",
        "property_options",
        "city_reference",
        "city_options",
        "city_destination_map",
        "cancellation_options",
        "numeric_summary",
    }
    missing = required - set(bundle)
    if missing:
        st.error(f"The model bundle is missing: {sorted(missing)}")
        st.stop()

    return bundle

@st.cache_data
def load_comparables(path: Path) -> pd.DataFrame:
    """Load valid historical listings with complete core comparison fields."""
    if not path.exists():
        return pd.DataFrame()

    try:
        comparables = pd.read_csv(
            path,
            low_memory=False,
        )
    except Exception:
        return pd.DataFrame()

    important_columns = [
        "city",
        "latitude",
        "longitude",
        "ttm_adjusted_revpar",
        "listing_type",
        "room_type",
        "guests",
        "bedrooms",
        "beds",
        "baths",
        "photos_count",
        "min_nights",
    ]

    if not set(important_columns).issubset(
        comparables.columns
    ):
        return pd.DataFrame()

    numeric_columns = [
        "latitude",
        "longitude",
        "ttm_adjusted_revpar",
        "guests",
        "bedrooms",
        "beds",
        "baths",
        "photos_count",
        "min_nights",
    ]

    for column in numeric_columns:
        comparables[column] = pd.to_numeric(
            comparables[column],
            errors="coerce",
        )

    comparables["city"] = (
        comparables["city"]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace("", pd.NA)
    )

    for column in [
        "listing_type",
        "room_type",
    ]:
        comparables[column] = (
            comparables[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )

    # A displayed comparable must have every core structural field used to
    # judge whether it genuinely resembles the proposed listing. Remaining
    # secondary model inputs may still be imputed by the fitted pipeline.
    comparables = comparables.dropna(
        subset=important_columns
    )

    comparables = comparables.loc[
        comparables["latitude"].between(
            -90,
            90,
        )
        & comparables["longitude"].between(
            -180,
            180,
        )
        & (
            comparables[
                "ttm_adjusted_revpar"
            ]
            > 0
        )
        & (comparables["guests"] > 0)
        & (comparables["bedrooms"] >= 0)
        & (comparables["beds"] > 0)
        & (comparables["baths"] > 0)
        & (comparables["photos_count"] > 0)
        & (comparables["min_nights"] >= 1)
    ].copy()

    if "listing_id" in comparables.columns:
        comparables = comparables.drop_duplicates(
            subset="listing_id",
            keep="first",
        )

    return comparables.reset_index(
        drop=True
    )


def comparable_property_group(
    room_type: Any,
    listing_type: Any,
) -> str:
    """Map source listing labels to the four app property categories."""
    room = (
        ""
        if pd.isna(room_type)
        else str(room_type).strip().lower()
    )

    listing = (
        ""
        if pd.isna(listing_type)
        else str(listing_type).strip().lower()
    )

    hosted_terms = (
        "hotel",
        "hostel",
        "resort",
        "bed and breakfast",
        "aparthotel",
        "boutique",
    )

    specialty_terms = (
        "tiny home",
        "yurt",
        "tent",
        "camper",
        "rv",
        "boat",
        "houseboat",
        "treehouse",
        "tree house",
        "farm stay",
        "barn",
        "cave",
        "dome",
        "castle",
        "hut",
        "earth home",
        "shipping container",
        "bus",
    )

    if (
        room == "hotel_room"
        or any(
            term in listing
            for term in hosted_terms
        )
    ):
        return "hotel_or_hosted_lodging"

    if any(
        term in listing
        for term in specialty_terms
    ):
        return "other_specialty"

    if room == "private_room":
        return "private_room"

    if room == "entire_home":
        return "entire_place"

    return "other_specialty"


def parse_clock_minutes(
    value: Any,
) -> float:
    """Convert source clock labels such as 4:00 PM to minutes after midnight."""
    if pd.isna(value):
        return np.nan

    text = (
        str(value)
        .replace("\u202f", " ")
        .replace("\xa0", " ")
        .strip()
        .upper()
    )

    match = re.search(
        r"(\d{1,2}):(\d{2})\s*([AP]M)",
        text,
    )

    if match is None:
        return np.nan

    hour = int(match.group(1))
    minute = int(match.group(2))
    period = match.group(3)

    if hour == 12:
        hour = 0

    if period == "PM":
        hour += 12

    return float(
        hour * 60 + minute
    )


def source_numeric(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in df.columns:
        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


def source_binary(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in df.columns:
        return pd.Series(
            0.0,
            index=df.index,
            dtype=float,
        )

    values = df[column]

    if pd.api.types.is_bool_dtype(
        values
    ):
        return values.astype(float)

    normalized = (
        values
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        {
            "true",
            "1",
            "yes",
            "y",
        }
    ).astype(float)


def build_comparable_feature_matrix(
    bundle: dict[str, Any],
    comparables: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Reconstruct the app's 40 model inputs for each historical listing.

    This makes the comparable search use the same imputer, scaler, feature
    schema, amenity group definitions, and categorical structure as the
    prediction itself.
    """
    features = list(
        bundle["feature_names"]
    )

    matrix = pd.DataFrame(
        0.0,
        index=comparables.index,
        columns=features,
    )

    numeric_features = [
        "photos_count",
        "guests",
        "bedrooms",
        "beds",
        "baths",
        "min_nights",
        "cleaning_fee",
        "extra_guest_fee",
    ]

    for feature in numeric_features:
        if feature in matrix.columns:
            matrix[feature] = source_numeric(
                comparables,
                feature,
            )

    if "is_destination" in matrix.columns:
        matrix["is_destination"] = (
            comparables["city"]
            .map(
                bundle[
                    "city_destination_map"
                ]
            )
            .fillna(0)
            .astype(float)
        )

    for group_feature, members in bundle[
        "amenity_groups"
    ].items():

        if group_feature not in matrix.columns:
            continue

        group_total = pd.Series(
            0.0,
            index=comparables.index,
        )

        for member in members:
            group_total = (
                group_total
                + source_binary(
                    comparables,
                    f"amenity_{member}",
                )
            )

        matrix[group_feature] = (
            group_total / len(members)
            if members
            else 0.0
        )

    for info in bundle[
        "standalone_amenities"
    ].values():

        feature_name = info["feature"]
        source_name = info["source"]

        if feature_name in matrix.columns:
            matrix[feature_name] = (
                source_binary(
                    comparables,
                    f"amenity_{source_name}",
                )
            )

    if "checkin_time" in comparables.columns:
        checkin_minutes = (
            comparables[
                "checkin_time"
            ].map(
                parse_clock_minutes
            )
        )
    else:
        checkin_minutes = pd.Series(
            np.nan,
            index=comparables.index,
        )

    if "checkout_time" in comparables.columns:
        checkout_minutes = (
            comparables[
                "checkout_time"
            ].map(
                parse_clock_minutes
            )
        )
    else:
        checkout_minutes = pd.Series(
            np.nan,
            index=comparables.index,
        )

    if (
        "Checkin_Time_Unknown"
        in matrix.columns
    ):
        matrix[
            "Checkin_Time_Unknown"
        ] = checkin_minutes.isna().astype(
            float
        )

    if (
        "Checkin_Nonstandard_Time"
        in matrix.columns
    ):
        matrix[
            "Checkin_Nonstandard_Time"
        ] = (
            checkin_minutes.notna()
            & ~checkin_minutes.between(
                15 * 60,
                17 * 60 - 1,
            )
        ).astype(float)

    if (
        "Checkout_Time_Unknown"
        in matrix.columns
    ):
        matrix[
            "Checkout_Time_Unknown"
        ] = checkout_minutes.isna().astype(
            float
        )

    if (
        "Checkout_Nonstandard_Time"
        in matrix.columns
    ):
        matrix[
            "Checkout_Nonstandard_Time"
        ] = (
            checkout_minutes.notna()
            & ~checkout_minutes.between(
                10 * 60,
                12 * 60 - 1,
            )
        ).astype(float)

    room_values = (
        comparables["room_type"]
        if "room_type" in comparables.columns
        else pd.Series(
            "",
            index=comparables.index,
        )
    )

    listing_values = (
        comparables["listing_type"]
        if "listing_type" in comparables.columns
        else pd.Series(
            "",
            index=comparables.index,
        )
    )

    property_groups = pd.Series(
        [
            comparable_property_group(
                room_type,
                listing_type,
            )
            for (
                room_type,
                listing_type,
            ) in zip(
                room_values,
                listing_values,
            )
        ],
        index=comparables.index,
        dtype="object",
    )

    for property_option in bundle[
        "property_options"
    ]:

        if (
            property_option
            == bundle[
                "property_reference"
            ]
        ):
            continue

        feature_name = (
            f"PropertyType__"
            f"{property_option}"
        )

        if feature_name in matrix.columns:
            matrix[feature_name] = (
                property_groups
                .eq(property_option)
                .astype(float)
            )

    city_values = (
        comparables["city"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    for city_option in bundle[
        "city_options"
    ]:

        if (
            city_option
            == bundle[
                "city_reference"
            ]
        ):
            continue

        feature_name = (
            f"City__{city_option}"
        )

        if feature_name in matrix.columns:
            matrix[feature_name] = (
                city_values
                .eq(city_option)
                .astype(float)
            )

    if (
        "instant_book_True"
        in matrix.columns
    ):
        matrix[
            "instant_book_True"
        ] = source_binary(
            comparables,
            "instant_book",
        )

    cancellation_values = (
        comparables[
            "cancellation_policy"
        ]
        .astype(str)
        .str.strip()
        if (
            "cancellation_policy"
            in comparables.columns
        )
        else pd.Series(
            "",
            index=comparables.index,
        )
    )

    for cancellation_option in bundle[
        "cancellation_options"
    ]:

        feature_name = (
            "cancellation_policy_"
            f"{cancellation_option}"
        )

        if feature_name in matrix.columns:
            matrix[feature_name] = (
                cancellation_values
                .eq(cancellation_option)
                .astype(float)
            )

    if (
        "exact_location_True"
        in matrix.columns
    ):
        matrix[
            "exact_location_True"
        ] = source_binary(
            comparables,
            "exact_location",
        )

    if (
        "single_fee_structure_True"
        in matrix.columns
    ):
        matrix[
            "single_fee_structure_True"
        ] = source_binary(
            comparables,
            "single_fee_structure",
        )

    return (
        matrix,
        property_groups,
    )


def select_comparable_listings(
    bundle: dict[str, Any],
    comparables: pd.DataFrame,
    raw_row: pd.DataFrame,
    selected_city: str,
    selected_property: str,
    predicted_revpar: float,
    count: int = 8,
) -> tuple[pd.DataFrame, bool]:
    """
    Rank historical listings by model-feature similarity and RevPAR proximity.

    The score is 65% coefficient-weighted distance in the fitted model's
    standardized feature space and 35% closeness of the historical listing's
    actual adjusted RevPAR to the current prediction.
    """
    if comparables.empty:
        return (
            pd.DataFrame(),
            False,
        )

    (
        historical_features,
        property_groups,
    ) = build_comparable_feature_matrix(
        bundle,
        comparables,
    )

    city_mask = (
        comparables["city"]
        .eq(selected_city)
    )

    same_property_mask = (
        city_mask
        & property_groups.eq(
            selected_property
        )
    )

    used_property_filter = (
        int(
            same_property_mask.sum()
        )
        >= max(
            5,
            count,
        )
    )

    candidate_mask = (
        same_property_mask
        if used_property_filter
        else city_mask
    )

    candidates = comparables.loc[
        candidate_mask
    ].copy()

    candidate_features = (
        historical_features.loc[
            candidate_mask
        ]
    )

    if candidates.empty:
        return (
            pd.DataFrame(),
            used_property_filter,
        )

    candidate_imputed = (
        bundle["imputer"].transform(
            candidate_features
        )
    )

    candidate_imputed_df = pd.DataFrame(
        candidate_imputed,
        columns=bundle[
            "feature_names"
        ],
        index=candidates.index,
    )

    candidate_scaled = (
        bundle["scaler"].transform(
            candidate_imputed_df
        )
    )

    query_imputed = (
        bundle["imputer"].transform(
            raw_row
        )
    )

    query_imputed_df = pd.DataFrame(
        query_imputed,
        columns=bundle[
            "feature_names"
        ],
    )

    query_scaled = (
        bundle["scaler"].transform(
            query_imputed_df
        )
    )[0]

    feature_differences = (
        candidate_scaled
        - query_scaled
    )

    coefficient_weights = np.abs(
        np.asarray(
            bundle["model"].coef_,
            dtype=float,
        )
    )

    coefficient_weights = np.nan_to_num(
        coefficient_weights,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if np.allclose(
        coefficient_weights.sum(),
        0.0,
    ):
        distance_weights = np.ones_like(
            coefficient_weights
        )
    else:
        max_weight = max(
            float(
                coefficient_weights.max()
            ),
            1e-9,
        )

        distance_weights = (
            0.35
            + 1.65
            * (
                coefficient_weights
                / max_weight
            )
        )

    feature_distance = np.sqrt(
        np.average(
            feature_differences ** 2,
            axis=1,
            weights=distance_weights,
        )
    )

    actual_revpar = pd.to_numeric(
        candidates[
            "ttm_adjusted_revpar"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    revpar_gap = np.abs(
        actual_revpar
        - predicted_revpar
    )

    positive_feature_distance = (
        feature_distance[
            feature_distance > 0
        ]
    )

    positive_revpar_gap = (
        revpar_gap[
            revpar_gap > 0
        ]
    )

    feature_scale = (
        float(
            np.nanmedian(
                positive_feature_distance
            )
        )
        if (
            positive_feature_distance.size
        )
        else 1.0
    )

    revpar_scale = (
        float(
            np.nanmedian(
                positive_revpar_gap
            )
        )
        if positive_revpar_gap.size
        else max(
            predicted_revpar,
            1.0,
        )
    )

    combined_distance = (
        0.65
        * (
            feature_distance
            / max(
                feature_scale,
                1e-9,
            )
        )
        + 0.35
        * (
            revpar_gap
            / max(
                revpar_scale,
                1e-9,
            )
        )
    )

    candidates[
        "Property group"
    ] = property_groups.loc[
        candidate_mask
    ].to_numpy()

    candidates[
        "Feature distance"
    ] = feature_distance

    candidates[
        "RevPAR difference"
    ] = (
        actual_revpar
        - predicted_revpar
    )

    candidates[
        "Match index"
    ] = (
        100
        * np.exp(
            -0.45
            * combined_distance
        )
    )

    candidates[
        "_combined_distance"
    ] = combined_distance

    candidates = (
        candidates
        .sort_values(
            "_combined_distance"
        )
        .head(count)
        .copy()
    )

    candidates["Rank"] = np.arange(
        1,
        len(candidates) + 1,
    )

    return (
        candidates,
        used_property_filter,
    )


def compact_number(
    value: Any,
) -> str:
    numeric_value = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(numeric_value):
        return "Unknown"

    return f"{float(numeric_value):g}"


def listing_identifier(
    value: Any,
) -> str:
    numeric_value = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]

    if not pd.isna(numeric_value):
        return str(
            int(
                numeric_value
            )
        )

    text = (
        ""
        if pd.isna(value)
        else str(value).strip()
    )

    return text


def build_comparable_map(
    comparable_rows: pd.DataFrame,
    predicted_revpar: float,
) -> folium.Map:
    """Create the Folium map and listing popups."""
    center = [
        float(
            comparable_rows[
                "latitude"
            ].mean()
        ),
        float(
            comparable_rows[
                "longitude"
            ].mean()
        ),
    ]

    comparable_map = folium.Map(
        location=center,
        zoom_start=11,
        tiles="CartoDB positron",
        control_scale=True,
    )

    for _, listing in comparable_rows.iterrows():

        actual_revpar = float(
            listing[
                "ttm_adjusted_revpar"
            ]
        )

        revpar_difference = (
            actual_revpar
            - predicted_revpar
        )

        relative_difference = (
            abs(
                revpar_difference
            )
            / predicted_revpar
            if predicted_revpar > 0
            else 0.0
        )

        if relative_difference <= 0.05:
            marker_color = (
                COLOR_NEUTRAL
            )
        elif revpar_difference > 0:
            marker_color = (
                COLOR_POSITIVE
            )
        else:
            marker_color = (
                COLOR_NEGATIVE
            )

        identifier = listing_identifier(
            listing.get(
                "listing_id",
                "",
            )
        )

        rank = int(
            listing.get(
                "Rank",
                0,
            )
        )

        comparable_label = (
            f"Comparable property {rank}"
        )

        property_category = pretty_label(
            str(
                listing.get(
                    "Property group",
                    "property",
                )
            )
        )

        cover_photo = listing.get(
            "cover_photo_url",
            "",
        )

        photo_html = ""

        if (
            not pd.isna(
                cover_photo
            )
            and str(
                cover_photo
            ).startswith(
                ("http://", "https://")
            )
        ):
            photo_html = (
                '<img src="'
                f'{escape(str(cover_photo), quote=True)}'
                '" style="'
                'width:100%;'
                'height:120px;'
                'object-fit:cover;'
                'border-radius:7px;'
                'margin-bottom:8px;'
                '">'
            )

        listing_link = (
            "https://www.airbnb.com/"
            f"rooms/{identifier}"
            if identifier
            else ""
        )

        difference_sign = (
            "+"
            if revpar_difference >= 0
            else "-"
        )

        link_html = (
            '<div style="'
            'margin-top:8px;'
            '">'
            '<a href="'
            f'{escape(listing_link, quote=True)}'
            '" target="_blank" '
            'rel="noopener noreferrer">'
            "Open listing"
            "</a>"
            "</div>"
            if listing_link
            else ""
        )

        popup_html = (
            '<div style="'
            'font-family:Arial,sans-serif;'
            'width:250px;'
            'line-height:1.35;'
            '">'
            f'{photo_html}'
            '<div style="'
            'font-size:14px;'
            'font-weight:700;'
            'margin-bottom:3px;'
            '">'
            f'{escape(comparable_label)}'
            '</div>'
            '<div style="'
            'font-size:11px;'
            'color:#64748b;'
            'margin-bottom:8px;'
            '">'
            f'{escape(property_category)}'
            '</div>'
            '<div style="'
            'font-size:12px;'
            '">'
            '<b>Adjusted TTM RevPAR:</b> '
            f'${actual_revpar:,.1f}<br>'
            '<b>Difference from estimate:</b> '
            f'{difference_sign}'
            f'${abs(revpar_difference):,.1f}<br>'
            '<b>Bedrooms:</b> '
            f'{compact_number(listing.get("bedrooms"))}'
            ' &nbsp; '
            '<b>Baths:</b> '
            f'{compact_number(listing.get("baths"))}'
            '<br>'
            '<b>Guests:</b> '
            f'{compact_number(listing.get("guests"))}'
            ' &nbsp; '
            '<b>Match index:</b> '
            f'{float(listing["Match index"]):.0f}/100'
            '</div>'
            f'{link_html}'
            '</div>'
        )

        tooltip_text = (
            f"Comparable {rank} | "
            f"${actual_revpar:,.1f} RevPAR"
        )

        folium.CircleMarker(
            location=[
                float(
                    listing[
                        "latitude"
                    ]
                ),
                float(
                    listing[
                        "longitude"
                    ]
                ),
            ],
            radius=8,
            color=marker_color,
            weight=2,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.82,
            tooltip=tooltip_text,
            popup=folium.Popup(
                popup_html,
                max_width=290,
            ),
        ).add_to(
            comparable_map
        )

    unique_locations = (
        comparable_rows[
            [
                "latitude",
                "longitude",
            ]
        ]
        .drop_duplicates()
    )

    if len(
        unique_locations
    ) > 1:

        comparable_map.fit_bounds(
            [
                [
                    float(
                        comparable_rows[
                            "latitude"
                        ].min()
                    ),
                    float(
                        comparable_rows[
                            "longitude"
                        ].min()
                    ),
                ],
                [
                    float(
                        comparable_rows[
                            "latitude"
                        ].max()
                    ),
                    float(
                        comparable_rows[
                            "longitude"
                        ].max()
                    ),
                ],
            ],
            padding=(
                30,
                30,
            ),
        )

    return comparable_map



def pretty_label(value: str) -> str:
    cleaned = (
        value
        .replace("AmenityGroup_", "")
        .replace("AmenityStandalone_", "")
        .replace("__", "_")
    )

    phrase_key = (
        cleaned
        .strip()
        .lower()
        .replace(" ", "_")
    )

    phrase_special = {
        "leisure_destination": "Outdoor Leisure",
    }

    if phrase_key in phrase_special:
        return phrase_special[phrase_key]

    special = {
        "bbq": "BBQ",
        "tv": "TV",
        "wifi": "Wi-Fi",
        "ev": "EV",
        "revpar": "RevPAR",
    }

    words = cleaned.replace("_", " ").split()
    return " ".join(special.get(word.lower(), word.title()) for word in words)


def categorical_label(value: str) -> str:
    return pretty_label(value)


def number_defaults(summary: dict[str, Any], feature: str, fallback: float) -> tuple[float, float, float]:
    stats = summary.get(feature, {})
    minimum = float(stats.get("min", 0.0))
    maximum = float(stats.get("max", max(fallback, 1.0)))
    median = float(stats.get("median", fallback))

    if not np.isfinite(minimum):
        minimum = 0.0
    if not np.isfinite(maximum):
        maximum = max(fallback, 1.0)
    if not np.isfinite(median):
        median = fallback

    maximum = max(maximum, minimum)
    median = min(max(median, minimum), maximum)
    return minimum, maximum, median


def build_feature_row(
    bundle: dict[str, Any],
    numeric_inputs: dict[str, float],
    amenity_checks: dict[str, bool],
    selected_city: str,
    selected_property: str,
    selected_checkin: str,
    selected_checkout: str,
    selected_cancellation: str,
    instant_book: str,
    exact_location: str,
    single_fee_structure: str,
) -> pd.DataFrame:
    features = list(bundle["feature_names"])
    row = pd.DataFrame(0.0, index=[0], columns=features)

    for feature, value in numeric_inputs.items():
        if feature in row.columns:
            row.at[0, feature] = float(value)

    city_key = selected_city
    destination_map = bundle.get("city_destination_map", {})
    if "is_destination" in row.columns:
        row.at[0, "is_destination"] = float(destination_map.get(city_key, 0))

    for group_feature, members in bundle["amenity_groups"].items():
        if group_feature not in row.columns:
            continue
        valid_members = list(members)
        checked = sum(bool(amenity_checks.get(member, False)) for member in valid_members)
        row.at[0, group_feature] = checked / len(valid_members) if valid_members else 0.0

    for display_name, info in bundle["standalone_amenities"].items():
        feature_name = info["feature"]
        source_name = info["source"]
        if feature_name in row.columns:
            row.at[0, feature_name] = float(bool(amenity_checks.get(source_name, False)))

    if selected_checkin == "Unknown":
        row.at[0, "Checkin_Time_Unknown"] = 1.0
    elif selected_checkin == "Nonstandard: before 3 PM or 5 PM and later":
        row.at[0, "Checkin_Nonstandard_Time"] = 1.0

    if selected_checkout == "Unknown":
        row.at[0, "Checkout_Time_Unknown"] = 1.0
    elif selected_checkout == "Nonstandard: before 10 AM or noon and later":
        row.at[0, "Checkout_Nonstandard_Time"] = 1.0

    property_reference = bundle["property_reference"]
    if selected_property != property_reference:
        feature_name = f"PropertyType__{selected_property}"
        if feature_name in row.columns:
            row.at[0, feature_name] = 1.0

    city_reference = bundle["city_reference"]
    if selected_city != city_reference:
        feature_name = f"City__{selected_city}"
        if feature_name in row.columns:
            row.at[0, feature_name] = 1.0

    if instant_book == "Yes" and "instant_book_True" in row.columns:
        row.at[0, "instant_book_True"] = 1.0

    cancellation_feature = f"cancellation_policy_{selected_cancellation}"
    if cancellation_feature in row.columns:
        row.at[0, cancellation_feature] = 1.0

    if exact_location == "Yes" and "exact_location_True" in row.columns:
        row.at[0, "exact_location_True"] = 1.0

    if single_fee_structure == "Yes" and "single_fee_structure_True" in row.columns:
        row.at[0, "single_fee_structure_True"] = 1.0

    return row


def neutralize_inapplicable_market_features(
    bundle: dict[str, Any],
    raw_row: pd.DataFrame,
    selected_city: str,
) -> pd.DataFrame:
    """
    Mark features that should not affect a listing in the selected market.

    Beach Access is treated as not applicable outside Myrtle Beach,
    Carolina Beach, and Wilmington. The raw input remains zero for display,
    while predict_revpar sets the corresponding standardized value to zero,
    removing both its prediction effect and chart contribution.
    """
    coastal_markets = {
        "myrtle_beach",
        "carolina_beach",
        "wilmington",
    }

    neutral_features: list[str] = []

    if selected_city not in coastal_markets:

        for display_name, info in bundle[
            "standalone_amenities"
        ].items():

            source_name = str(
                info.get(
                    "source",
                    "",
                )
            )

            feature_name = str(
                info.get(
                    "feature",
                    "",
                )
            )

            combined_name = " ".join(
                [
                    str(display_name),
                    source_name,
                    feature_name,
                ]
            ).lower()

            if (
                "beach" in combined_name
                and feature_name
                in raw_row.columns
            ):
                neutral_features.append(
                    feature_name
                )

    raw_row.attrs[
        "neutral_features"
    ] = neutral_features

    return raw_row


def predict_revpar(bundle: dict[str, Any], raw_row: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    imputed = bundle["imputer"].transform(raw_row)
    imputed_df = pd.DataFrame(imputed, columns=bundle["feature_names"])
    scaled = bundle["scaler"].transform(imputed_df)
    scaled_df = pd.DataFrame(scaled, columns=bundle["feature_names"])

    neutral_features = raw_row.attrs.get(
        "neutral_features",
        [],
    )

    for feature_name in neutral_features:
        if feature_name in scaled_df.columns:
            scaled_df.at[
                0,
                feature_name,
            ] = 0.0

    log_prediction = float(bundle["model"].predict(scaled_df)[0])
    revpar = max(0.0, float(np.expm1(log_prediction)))

    coefficients = np.asarray(bundle["model"].coef_, dtype=float)
    contributions = scaled_df.iloc[0].to_numpy(dtype=float) * coefficients

    contribution_df = pd.DataFrame(
        {
            "Feature": bundle["feature_names"],
            "Contribution": contributions,
        }
    )

    return revpar, contribution_df



def find_actionable_opportunities(
    bundle: dict[str, Any],
    raw_row: pd.DataFrame,
    current_revpar: float,
    numeric_inputs: dict[str, float],
    amenity_checks: dict[str, bool],
    selected_city: str,
    selected_checkin: str,
    selected_checkout: str,
    selected_cancellation: str,
    instant_book: str,
    exact_location: str,
    single_fee_structure: str,
) -> list[dict[str, Any]]:
    """
    Test realistic one-at-a-time input changes and rank only those that
    increase the fitted model's predicted adjusted RevPAR.

    These are predictive scenario comparisons, not causal estimates.
    """
    opportunities: list[dict[str, Any]] = []

    def add_candidate(
        title: str,
        detail: str,
        candidate_row: pd.DataFrame,
        category: str,
    ) -> None:
        candidate_revpar, _ = predict_revpar(
            bundle,
            candidate_row,
        )

        uplift = (
            candidate_revpar
            - current_revpar
        )

        uplift_pct = (
            uplift
            / current_revpar
            * 100
            if current_revpar > 0
            else 0.0
        )

        # Ignore changes too small to be useful in the interface.
        if uplift <= 0 or uplift_pct < 0.5:
            return

        opportunities.append(
            {
                "Title": title,
                "Detail": detail,
                "Category": category,
                "Candidate RevPAR": candidate_revpar,
                "Uplift": uplift,
                "Uplift percent": uplift_pct,
                "Annual uplift": uplift * 365,
            }
        )

    # --------------------------------------------------------
    # Amenity groups: test adding one currently missing amenity
    # --------------------------------------------------------
    for group_feature, members in bundle[
        "amenity_groups"
    ].items():

        if group_feature not in raw_row.columns:
            continue

        missing_members = [
            member
            for member in members
            if not amenity_checks.get(
                member,
                False,
            )
        ]

        if not missing_members:
            continue

        candidate_row = raw_row.copy(
            deep=True
        )

        current_fraction = float(
            candidate_row.at[
                0,
                group_feature,
            ]
        )

        step = (
            1.0 / len(members)
            if members
            else 0.0
        )

        candidate_row.at[
            0,
            group_feature,
        ] = min(
            1.0,
            current_fraction + step,
        )

        group_name = pretty_label(
            group_feature
        )

        example_amenities = ", ".join(
            pretty_label(member)
            for member in missing_members[:3]
        )

        add_candidate(
            title=(
                f"Add to the {group_name} "
                f"amenity group"
            ),
            detail=(
                f"Add one currently missing amenity, such as "
                f"{example_amenities}. This one-step group increase "
                f"is ranked against every other tested change."
            ),
            candidate_row=candidate_row,
            category="Amenity group",
        )

    # --------------------------------------------------------
    # Standalone amenities
    # --------------------------------------------------------
    beach_city_keys = {
        "myrtle_beach",
        "carolina_beach",
        "wilmington",
    }

    for display_name, info in bundle[
        "standalone_amenities"
    ].items():

        source_name = info["source"]
        feature_name = info["feature"]

        if (
            feature_name
            not in raw_row.columns
            or amenity_checks.get(
                source_name,
                False,
            )
        ):
            continue

        is_beach_access = (
            "beach" in display_name.lower()
            or "beach" in source_name.lower()
        )

        if (
            is_beach_access
            and selected_city
            not in beach_city_keys
        ):
            continue

        candidate_row = raw_row.copy(
            deep=True
        )

        candidate_row.at[
            0,
            feature_name,
        ] = 1.0

        add_candidate(
            title=(
                f"Add "
                f"{pretty_label(display_name)}"
            ),
            detail=(
                "Scenario estimate for adding "
                "this standalone amenity."
            ),
            candidate_row=candidate_row,
            category="Standalone amenity",
        )

    # --------------------------------------------------------
    # Binary listing settings
    # --------------------------------------------------------
    binary_changes = [
        (
            "instant_book_True",
            instant_book,
            "Enable Instant Book",
            "Disable Instant Book",
            "Booking setting",
        ),
        (
            "exact_location_True",
            exact_location,
            "Show the exact location",
            "Hide the exact location",
            "Location setting",
        ),
        (
            "single_fee_structure_True",
            single_fee_structure,
            "Use a single fee structure",
            "Use separate fee components",
            "Fee setting",
        ),
    ]

    for (
        feature_name,
        current_value,
        enable_title,
        disable_title,
        category,
    ) in binary_changes:

        if feature_name not in raw_row.columns:
            continue

        candidate_row = raw_row.copy(
            deep=True
        )

        enable_change = (
            current_value == "No"
        )

        candidate_row.at[
            0,
            feature_name,
        ] = (
            1.0
            if enable_change
            else 0.0
        )

        add_candidate(
            title=(
                enable_title
                if enable_change
                else disable_title
            ),
            detail=(
                "Estimated effect of changing "
                "only this listing setting."
            ),
            candidate_row=candidate_row,
            category=category,
        )

    # --------------------------------------------------------
    # Check-in and checkout windows
    # --------------------------------------------------------
    checkin_options = [
        "Standard: 3 PM through 4:59 PM",
        "Nonstandard: before 3 PM or 5 PM and later",
        "Unknown",
    ]

    checkout_options = [
        "Standard: 10 AM through 11:59 AM",
        "Nonstandard: before 10 AM or noon and later",
        "Unknown",
    ]

    def set_time_window(
        row: pd.DataFrame,
        prefix: str,
        option: str,
    ) -> None:

        nonstandard_feature = (
            f"{prefix}_Nonstandard_Time"
        )

        unknown_feature = (
            f"{prefix}_Time_Unknown"
        )

        if nonstandard_feature in row.columns:
            row.at[
                0,
                nonstandard_feature,
            ] = 0.0

        if unknown_feature in row.columns:
            row.at[
                0,
                unknown_feature,
            ] = 0.0

        if option == "Unknown":
            if unknown_feature in row.columns:
                row.at[
                    0,
                    unknown_feature,
                ] = 1.0

        elif option.startswith(
            "Nonstandard"
        ):
            if nonstandard_feature in row.columns:
                row.at[
                    0,
                    nonstandard_feature,
                ] = 1.0

    for (
        prefix,
        current_option,
        options,
        label,
    ) in [
        (
            "Checkin",
            selected_checkin,
            checkin_options,
            "check-in",
        ),
        (
            "Checkout",
            selected_checkout,
            checkout_options,
            "checkout",
        ),
    ]:

        alternative_rows = []

        for option in options:

            if option == current_option:
                continue

            candidate_row = raw_row.copy(
                deep=True
            )

            set_time_window(
                candidate_row,
                prefix,
                option,
            )

            candidate_revpar, _ = predict_revpar(
                bundle,
                candidate_row,
            )

            alternative_rows.append(
                (
                    candidate_revpar,
                    option,
                    candidate_row,
                )
            )

        if alternative_rows:

            (
                _,
                best_option,
                best_row,
            ) = max(
                alternative_rows,
                key=lambda item: item[0],
            )

            add_candidate(
                title=(
                    f"Change the {label} "
                    f"window to "
                    f"{best_option.split(':')[0]}"
                ),
                detail=(
                    f"Current: {current_option}. "
                    f"Tested against the other "
                    f"available {label} options."
                ),
                candidate_row=best_row,
                category=(
                    f"{label.title()} window"
                ),
            )

    # --------------------------------------------------------
    # Cancellation policy
    # --------------------------------------------------------
    cancellation_columns = [
        column
        for column in raw_row.columns
        if column.startswith(
            "cancellation_policy_"
        )
    ]

    cancellation_alternatives = []

    for option in bundle[
        "cancellation_options"
    ]:

        if option == selected_cancellation:
            continue

        candidate_row = raw_row.copy(
            deep=True
        )

        for column in cancellation_columns:
            candidate_row.at[
                0,
                column,
            ] = 0.0

        option_feature = (
            f"cancellation_policy_{option}"
        )

        if option_feature in candidate_row.columns:
            candidate_row.at[
                0,
                option_feature,
            ] = 1.0

        candidate_revpar, _ = predict_revpar(
            bundle,
            candidate_row,
        )

        cancellation_alternatives.append(
            (
                candidate_revpar,
                option,
                candidate_row,
            )
        )

    if cancellation_alternatives:

        (
            _,
            best_policy,
            best_policy_row,
        ) = max(
            cancellation_alternatives,
            key=lambda item: item[0],
        )

        add_candidate(
            title=(
                "Consider the "
                f"{pretty_label(best_policy)} "
                "cancellation policy"
            ),
            detail=(
                f"Current policy: "
                f"{pretty_label(selected_cancellation)}. "
                f"This is the strongest alternative "
                f"tested by the model."
            ),
            candidate_row=best_policy_row,
            category="Cancellation policy",
        )

    # --------------------------------------------------------
    # Listing photos
    # --------------------------------------------------------
    if "photos_count" in raw_row.columns:

        current_photos = float(
            numeric_inputs.get(
                "photos_count",
                raw_row.at[
                    0,
                    "photos_count",
                ],
            )
        )

        photo_max = float(
            bundle[
                "numeric_summary"
            ]
            .get(
                "photos_count",
                {},
            )
            .get(
                "max",
                current_photos,
            )
        )

        target_photos = min(
            photo_max,
            current_photos + 5,
        )

        if target_photos > current_photos:

            candidate_row = raw_row.copy(
                deep=True
            )

            candidate_row.at[
                0,
                "photos_count",
            ] = target_photos

            add_candidate(
                title=(
                    f"Increase listing photos "
                    f"to {target_photos:.0f}"
                ),
                detail=(
                    f"Tests adding "
                    f"{target_photos-current_photos:.0f} "
                    f"photos while leaving all "
                    f"other inputs unchanged."
                ),
                candidate_row=candidate_row,
                category="Listing content",
            )

    # --------------------------------------------------------
    # Minimum-stay alternatives
    # --------------------------------------------------------
    if "min_nights" in raw_row.columns:

        current_min_nights = float(
            numeric_inputs.get(
                "min_nights",
                raw_row.at[
                    0,
                    "min_nights",
                ],
            )
        )

        minimum_allowed = float(
            bundle[
                "numeric_summary"
            ]
            .get(
                "min_nights",
                {},
            )
            .get(
                "min",
                1.0,
            )
        )

        maximum_allowed = float(
            bundle[
                "numeric_summary"
            ]
            .get(
                "min_nights",
                {},
            )
            .get(
                "max",
                current_min_nights,
            )
        )

        stay_options = sorted(
            {
                value
                for value in [
                    minimum_allowed,
                    1.0,
                    2.0,
                    3.0,
                    7.0,
                    14.0,
                ]
                if (
                    minimum_allowed
                    <= value
                    <= maximum_allowed
                    and value
                    != current_min_nights
                )
            }
        )

        stay_alternatives = []

        for option in stay_options:

            candidate_row = raw_row.copy(
                deep=True
            )

            candidate_row.at[
                0,
                "min_nights",
            ] = option

            candidate_revpar, _ = predict_revpar(
                bundle,
                candidate_row,
            )

            stay_alternatives.append(
                (
                    candidate_revpar,
                    option,
                    candidate_row,
                )
            )

        if stay_alternatives:

            (
                _,
                best_stay,
                best_stay_row,
            ) = max(
                stay_alternatives,
                key=lambda item: item[0],
            )

            add_candidate(
                title=(
                    f"Set the minimum stay "
                    f"to {best_stay:.0f} nights"
                ),
                detail=(
                    f"Current minimum: "
                    f"{current_min_nights:.0f} nights. "
                    f"This is the strongest tested "
                    f"alternative."
                ),
                candidate_row=best_stay_row,
                category="Minimum stay",
            )

    opportunities.sort(
        key=lambda item: item[
            "Uplift"
        ],
        reverse=True,
    )

    return opportunities[:3]

def aggregate_display_contributions(
    contribution_df: pd.DataFrame,
    selected_city: str,
    selected_property: str,
    selected_checkin: str,
    selected_checkout: str,
    selected_cancellation: str,
    instant_book: str,
    exact_location: str,
    single_fee_structure: str,
    destination_value: int,
) -> pd.DataFrame:
    """
    Combine all standardized dummy columns belonging to one categorical
    variable into one displayed contribution.

    StandardScaler centers dummy variables, so an unselected city dummy can
    still have a nonzero standardized value. Summing the entire dummy family
    gives the correct contribution for the selected category without showing
    unrelated cities.
    """
    df = contribution_df.copy()
    used = pd.Series(False, index=df.index)
    grouped_rows = []

    grouped_prefixes = [
        ("City__", f"City: {pretty_label(selected_city)}"),
        ("PropertyType__", f"Property type: {pretty_label(selected_property)}"),
        (
            "cancellation_policy_",
            f"Cancellation policy: {pretty_label(selected_cancellation)}",
        ),
        ("Checkin_", f"Check-in: {selected_checkin}"),
        ("Checkout_", f"Checkout: {selected_checkout}"),
        ("instant_book_", f"Instant Book: {instant_book}"),
        ("exact_location_", f"Exact location shown: {exact_location}"),
        (
            "single_fee_structure_",
            f"Single fee: {single_fee_structure}",
        ),
    ]

    for prefix, label in grouped_prefixes:
        mask = df["Feature"].str.startswith(prefix)
        if mask.any():
            grouped_rows.append(
                {
                    "Feature": prefix.rstrip("_"),
                    "Contribution": float(df.loc[mask, "Contribution"].sum()),
                    "Label": label,
                }
            )
            used = used | mask

    destination_mask = df["Feature"].eq("is_destination")
    if destination_mask.any():
        grouped_rows.append(
            {
                "Feature": "is_destination",
                "Contribution": float(
                    df.loc[destination_mask, "Contribution"].sum()
                ),
                "Label": (
                    "Market type: Destination"
                    if destination_value
                    else "Market type: Metro"
                ),
            }
        )
        used = used | destination_mask

    remaining = df.loc[~used, ["Feature", "Contribution"]].copy()
    remaining["Label"] = remaining["Feature"].map(pretty_label)

    grouped = pd.DataFrame(grouped_rows)
    display_df = pd.concat(
        [remaining, grouped],
        ignore_index=True,
    )

    display_df = display_df.loc[
        display_df["Contribution"].abs() > 1e-8
    ].copy()

    display_df["Percent impact"] = (
        np.expm1(display_df["Contribution"].abs()) * 100
    )
    display_df["Display impact"] = np.where(
        display_df["Contribution"] >= 0,
        "+",
        "-",
    ) + display_df["Percent impact"].round(1).astype(str) + "%"

    return display_df


bundle = load_bundle(BUNDLE_PATH)

stored_version = str(bundle.get("sklearn_version", "unknown"))
if stored_version != "unknown" and stored_version != sklearn.__version__:
    st.warning(
        f"This model was saved with scikit-learn {stored_version}, "
        f"but the app is running {sklearn.__version__}. "
        "Pin the saved version in requirements.txt."
    )

logo_data_uri = ""
if LOGO_PATH.exists():
    logo_data_uri = (
        "data:image/png;base64,"
        + base64.b64encode(
            LOGO_PATH.read_bytes()
        ).decode("ascii")
    )

if logo_data_uri:
    st.markdown(
        f"""
        <div class="brand-header">
            <img
                class="brand-logo"
                src="{logo_data_uri}"
                alt="RevPulse"
            >
            <div class="brand-tagline">
                Your pulse on the short-term rental market.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="brand-header">
            <div style="font-size:2rem;font-weight:800;">RevPulse</div>
            <div class="brand-tagline">
                Your pulse on the short-term rental market.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Reserve this position so the prediction summary stays at the top
# even though its values are calculated after the input widgets.
summary_placeholder = st.container()

# ============================================================
# INPUTS
# ============================================================

with st.expander(
    "Market and property",
    expanded=True,
):
    market_cols = st.columns(4, gap="medium")

    city_options = list(bundle["city_options"])
    with market_cols[0]:
        selected_city = st.selectbox(
            "City",
            options=city_options,
            format_func=categorical_label,
        )

    property_options = list(bundle["property_options"])
    with market_cols[1]:
        selected_property = st.selectbox(
            "Property type",
            options=property_options,
            format_func=categorical_label,
        )

    with market_cols[2]:
        selected_checkin = st.selectbox(
            "Check-in window",
            options=[
                "Standard: 3 PM through 4:59 PM",
                "Nonstandard: before 3 PM or 5 PM and later",
                "Unknown",
            ],
        )

    with market_cols[3]:
        selected_checkout = st.selectbox(
            "Checkout window",
            options=[
                "Standard: 10 AM through 11:59 AM",
                "Nonstandard: before 10 AM or noon and later",
                "Unknown",
            ],
        )

    destination_value = int(
        bundle["city_destination_map"].get(
            selected_city,
            0,
        )
    )

    destination_label = (
        "Destination market"
        if destination_value
        else "Metro market"
    )

    st.caption(
        f"Model market classification for "
        f"{pretty_label(selected_city)}: {destination_label}"
    )

with st.expander(
    "Listing structure",
    expanded=True,
):
    numeric_summary = bundle["numeric_summary"]

    numeric_specs = [
        ("bedrooms", "Bedrooms", 2.0, 1.0),
        ("beds", "Beds", 2.0, 1.0),
        ("baths", "Bathrooms", 1.0, 0.5),
        ("guests", "Maximum guests", 4.0, 1.0),
        ("photos_count", "Listing photos", 20.0, 1.0),
        ("min_nights", "Minimum nights", 2.0, 1.0),
        ("cleaning_fee", "Cleaning fee ($)", 100.0, 5.0),
        ("extra_guest_fee", "Extra guest fee ($)", 0.0, 5.0),
    ]

    numeric_inputs: dict[str, float] = {}

    for row_start in range(0, len(numeric_specs), 4):
        row_specs = numeric_specs[
            row_start:row_start + 4
        ]

        row_cols = st.columns(
            len(row_specs),
            gap="medium",
        )

        for column, (
            feature,
            label,
            fallback,
            step,
        ) in zip(row_cols, row_specs):
            minimum, maximum, median = number_defaults(
                numeric_summary,
                feature,
                fallback,
            )

            with column:
                numeric_inputs[feature] = st.number_input(
                    label,
                    min_value=float(minimum),
                    max_value=float(maximum),
                    value=float(median),
                    step=float(step),
                    key=f"num_{feature}",
                )

with st.expander(
    "Policies and listing settings",
    expanded=True,
):
    policy_cols = st.columns(4, gap="medium")

    with policy_cols[0]:
        selected_cancellation = st.selectbox(
            "Cancellation policy",
            options=list(
                bundle["cancellation_options"]
            ),
            format_func=categorical_label,
        )

    with policy_cols[1]:
        instant_book = st.selectbox(
            "Instant Book",
            ["No", "Yes"],
        )

    with policy_cols[2]:
        exact_location = st.selectbox(
            "Exact location shown",
            ["No", "Yes"],
        )

    with policy_cols[3]:
        single_fee_structure = st.selectbox(
            "Single fee structure",
            ["No", "Yes"],
        )

amenity_checks: dict[str, bool] = {}

with st.expander(
    "Amenities",
    expanded=False,
):
    st.caption(
        "Amenities are grouped exactly as they were during model training. "
        "Property Readiness amenities are selected by default."
    )

    amenity_columns = st.columns(3, gap="medium")

    for group_index, (
        group_feature,
        members,
    ) in enumerate(
        bundle["amenity_groups"].items()
    ):
        group_title = pretty_label(
            group_feature
        )

        is_property_readiness = (
            group_feature
            == "AmenityGroup_Property_Readiness"
        )

        with amenity_columns[
            group_index % 3
        ]:
            with st.container(border=True):
                st.markdown(
                    f'<div class="amenity-group-title">'
                    f'{escape(group_title)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                control_cols = st.columns(2)

                with control_cols[0]:
                    select_all_clicked = st.button(
                        "Select all",
                        key=f"select_all_{group_feature}",
                        use_container_width=True,
                    )

                with control_cols[1]:
                    clear_all_clicked = st.button(
                        "Clear all",
                        key=f"clear_all_{group_feature}",
                        use_container_width=True,
                    )

                if select_all_clicked:
                    for amenity in members:
                        st.session_state[
                            f"amenity_{group_feature}_{amenity}"
                        ] = True

                if clear_all_clicked:
                    for amenity in members:
                        st.session_state[
                            f"amenity_{group_feature}_{amenity}"
                        ] = False

                checkbox_cols = st.columns(2)

                for index, amenity in enumerate(
                    members
                ):
                    amenity_key = (
                        f"amenity_"
                        f"{group_feature}_"
                        f"{amenity}"
                    )

                    with checkbox_cols[index % 2]:
                        amenity_checks[
                            amenity
                        ] = st.checkbox(
                            pretty_label(amenity),
                            value=is_property_readiness,
                            key=amenity_key,
                        )

    standalone_column_index = (
        len(bundle["amenity_groups"])
        % 3
    )

    beach_city_keys = {
        "myrtle_beach",
        "carolina_beach",
        "wilmington",
    }

    beach_access_available = (
        selected_city in beach_city_keys
    )

    with amenity_columns[
        standalone_column_index
    ]:
        with st.container(border=True):
            st.markdown(
                '<div class="amenity-group-title">'
                'Standalone amenities'
                '</div>',
                unsafe_allow_html=True,
            )

            standalone_cols = st.columns(2)

            for index, (
                display_name,
                info,
            ) in enumerate(
                bundle[
                    "standalone_amenities"
                ].items()
            ):
                source_name = info["source"]

                is_beach_access = (
                    "beach" in display_name.lower()
                    or "beach" in source_name.lower()
                )

                checkbox_disabled = (
                    is_beach_access
                    and not beach_access_available
                )

                checkbox_key = (
                    f"standalone_"
                    f"{source_name}"
                )

                if (
                    checkbox_disabled
                    and st.session_state.get(
                        checkbox_key,
                        False,
                    )
                ):
                    st.session_state[
                        checkbox_key
                    ] = False

                with standalone_cols[index % 2]:
                    amenity_checks[
                        source_name
                    ] = st.checkbox(
                        pretty_label(
                            display_name
                        ),
                        key=checkbox_key,
                        disabled=checkbox_disabled,
                        help=(
                            "Available only for Myrtle Beach, "
                            "Carolina Beach, or Wilmington."
                            if is_beach_access
                            else None
                        ),
                    )

            if not beach_access_available:
                st.caption(
                    "Beach Access can only be selected for "
                    "Myrtle Beach, Carolina Beach, or Wilmington."
                )


# ============================================================
# BUILD MODEL INPUT AND PREDICT
# ============================================================

raw_row = build_feature_row(
    bundle=bundle,
    numeric_inputs=numeric_inputs,
    amenity_checks=amenity_checks,
    selected_city=selected_city,
    selected_property=selected_property,
    selected_checkin=selected_checkin,
    selected_checkout=selected_checkout,
    selected_cancellation=selected_cancellation,
    instant_book=instant_book,
    exact_location=exact_location,
    single_fee_structure=single_fee_structure,
)

raw_row = neutralize_inapplicable_market_features(
    bundle=bundle,
    raw_row=raw_row,
    selected_city=selected_city,
)

revpar, raw_contribution_df = predict_revpar(
    bundle,
    raw_row,
)

contribution_df = aggregate_display_contributions(
    contribution_df=raw_contribution_df,
    selected_city=selected_city,
    selected_property=selected_property,
    selected_checkin=selected_checkin,
    selected_checkout=selected_checkout,
    selected_cancellation=selected_cancellation,
    instant_book=instant_book,
    exact_location=exact_location,
    single_fee_structure=single_fee_structure,
    destination_value=destination_value,
)

dataset_median = float(
    bundle.get(
        "market_median",
        75.0,
    )
)

selected_city_median = float(
    CITY_REVPAR_MEDIANS.get(
        selected_city,
        dataset_median,
    )
)

selected_city_label = pretty_label(
    selected_city
)

difference = (
    revpar
    - selected_city_median
)

difference_pct = (
    difference
    / selected_city_median
    * 100
    if selected_city_median
    else 0.0
)

actionable_opportunities = (
    find_actionable_opportunities(
        bundle=bundle,
        raw_row=raw_row,
        current_revpar=revpar,
        numeric_inputs=numeric_inputs,
        amenity_checks=amenity_checks,
        selected_city=selected_city,
        selected_checkin=selected_checkin,
        selected_checkout=selected_checkout,
        selected_cancellation=selected_cancellation,
        instant_book=instant_book,
        exact_location=exact_location,
        single_fee_structure=single_fee_structure,
    )
)


# ============================================================
# RENDER PREDICTION SUMMARY IN THE RESERVED TOP POSITION
# ============================================================

with summary_placeholder:
    st.markdown(
        '<hr class="rule">',
        unsafe_allow_html=True,
    )

    summary_left, summary_right = st.columns(
        [0.72, 1.28],
        gap="large",
    )

    with summary_left:

        st.markdown(
            '<div class="section-label">Prediction</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="hero-value">'
            f'${revpar:,.0f}'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="hero-label">'
            'Predicted adjusted RevPAR per night'
            '</div>',
            unsafe_allow_html=True,
        )

        direction = (
            "above"
            if difference >= 0
            else "below"
        )

        sign = (
            "+"
            if difference >= 0
            else "-"
        )

        st.markdown(
            f'<div class="hero-context">'
            f'{sign}${abs(difference):,.0f} '
            f'({sign}{abs(difference_pct):.0f}%) '
            f'{direction} the {selected_city_label} median'
            f'</div>',
            unsafe_allow_html=True,
        )


    with summary_right:

        annualized_revenue = (
            revpar * 365
        )

        monthly_revenue = (
            annualized_revenue / 12
        )

        city_median_annualized_revenue = (
            selected_city_median * 365
        )

        revenue_card_html = (
            f'<div class="revenue-card">'
            f'<div class="revenue-eyebrow">'
            f'Annualized revenue potential'
            f'</div>'
            f'<div class="revenue-main">'
            f'${annualized_revenue:,.0f}'
            f'</div>'
            f'<div class="revenue-sub">'
            f'Adjusted RevPAR × 365 available nights'
            f'</div>'
            f'<div class="revenue-grid">'
            f'<div class="revenue-stat">'
            f'<div class="revenue-stat-value">'
            f'${monthly_revenue:,.0f}'
            f'</div>'
            f'<div class="revenue-stat-label">'
            f'Monthly equivalent'
            f'</div>'
            f'</div>'
            f'<div class="revenue-stat">'
            f'<div class="revenue-stat-value">'
            f'${city_median_annualized_revenue:,.0f}'
            f'</div>'
            f'<div class="revenue-stat-label">'
            f'{selected_city_label} median annualized'
            f'</div>'
            f'</div>'
            f'</div>'
            f'<div class="revenue-context">'
            f'{pretty_label(selected_property)} '
            f'in {pretty_label(selected_city)} · '
            f'occupancy-adjusted estimate, '
            f'not gross booking revenue'
            f'</div>'
            f'</div>'
        )

        st.markdown(
            revenue_card_html,
            unsafe_allow_html=True,
        )

# ============================================================
# FULL-WIDTH CONTRIBUTION CHART
# ============================================================

st.markdown(
    '<hr class="rule">',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-label">'
    'Largest Factors Influencing This Prediction'
    '</div>',
    unsafe_allow_html=True,
)

chart_df = (
    contribution_df.reindex(
        contribution_df[
            "Contribution"
        ]
        .abs()
        .sort_values(
            ascending=False
        )
        .index
    )
    .head(15)
    .sort_values(
        "Contribution"
    )
)


if chart_df.empty:

    st.info(
        "No nonzero feature contributions "
        "for the current inputs."
    )

else:

    colors = np.where(
        chart_df["Contribution"] >= 0,
        COLOR_POSITIVE,
        COLOR_NEGATIVE,
    )

    contribution_min = min(
        float(
            chart_df[
                "Contribution"
            ].min()
        ),
        0.0,
    )

    contribution_max = max(
        float(
            chart_df[
                "Contribution"
            ].max()
        ),
        0.0,
    )

    contribution_span = max(
        contribution_max
        - contribution_min,
        0.05,
    )

    bar_label_gap = (
        contribution_span * 0.018
    )

    left_padding = (
        contribution_span * 0.16
    )

    right_padding = (
        contribution_span * 0.16
    )

    figure = go.Figure(
        go.Bar(
            x=chart_df[
                "Contribution"
            ],
            y=chart_df[
                "Label"
            ],
            orientation="h",
            marker_color=colors,
            customdata=chart_df[
                "Display impact"
            ],
            hovertemplate=(
                "%{y}: "
                "%{customdata}"
                "<extra></extra>"
            ),
        )
    )

    for _, row in chart_df.iterrows():

        value = float(
            row["Contribution"]
        )

        if value >= 0:

            annotation_x = (
                value
                + bar_label_gap
            )

            xanchor = "left"

        else:

            annotation_x = (
                value
                - bar_label_gap
            )

            xanchor = "right"

        figure.add_annotation(
            x=annotation_x,
            y=row["Label"],
            text=row[
                "Display impact"
            ],
            showarrow=False,
            xanchor=xanchor,
            yanchor="middle",
            font=dict(
                size=12,
            ),
        )


    figure.add_vline(
        x=0,
        line_width=1,
        line_color="#8b95a5",
    )


    figure.update_layout(
        height=max(
            500,
            len(chart_df) * 38,
        ),
        margin=dict(
            l=260,
            r=105,
            t=10,
            b=10,
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[
                contribution_min
                - left_padding,
                contribution_max
                + right_padding,
            ],
        ),
        yaxis=dict(
            showgrid=False,
            automargin=True,
            tickfont=dict(
                size=12,
            ),
        ),
        plot_bgcolor=(
            "rgba(0,0,0,0)"
        ),
        paper_bgcolor=(
            "rgba(0,0,0,0)"
        ),
        bargap=0.30,
    )


    st.plotly_chart(
        figure,
        use_container_width=True,
        config={
            "displayModeBar": False
        },
    )

st.caption(
    "Positive values increase the predicted adjusted RevPAR "
    "relative to the average listing, while negative values decrease it."
)


# ============================================================
# ACTIONABLE IMPROVEMENT OPPORTUNITIES
# ============================================================

st.markdown(
    '<hr class="rule">',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-label">'
    'Potential Listing Improvements'
    '</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Each opportunity changes one input at a time and reruns the model. "
    "The estimates show predictive associations, not guaranteed causal gains."
)

if actionable_opportunities:

    opportunity_cards = []

    for rank, opportunity in enumerate(
        actionable_opportunities,
        start=1,
    ):

        opportunity_cards.append(
            f'<div class="opportunity-card">'
            f'<div class="opportunity-rank">'
            f'Opportunity {rank}'
            f'</div>'
            f'<div class="opportunity-title">'
            f'{escape(str(opportunity["Title"]))}'
            f'</div>'
            f'<div class="opportunity-impact">'
            f'+${opportunity["Uplift"]:,.0f} '
            f'RevPAR per night'
            f'</div>'
            f'<div class="opportunity-annual">'
            f'+${opportunity["Annual uplift"]:,.0f} '
            f'annualized model estimate'
            f'</div>'
            f'<div class="opportunity-detail">'
            f'{escape(str(opportunity["Detail"]))}'
            f'</div>'
            f'</div>'
        )

    opportunity_grid_html = (
        '<div class="opportunity-grid">'
        + "".join(opportunity_cards)
        + '</div>'
    )

    st.markdown(
        opportunity_grid_html,
        unsafe_allow_html=True,
    )

else:

    st.info(
        "None of the tested one-step changes produced a meaningful "
        "increase in the current model prediction."
    )


# ============================================================
# HISTORICAL COMPARABLE LISTINGS MAP
# ============================================================

st.markdown(
    '<hr class="rule">',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-label">'
    'Comparable Properties'
    '</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Historical listings are ranked using 65% similarity in the fitted "
    "model's standardized feature space and 35% closeness of actual adjusted "
    "TTM RevPAR to the current prediction. Listings with a missing core "
    "property, capacity, location, photo-count, minimum-stay, or RevPAR field "
    "are excluded before ranking."
)

comparables_df = load_comparables(
    COMPARABLES_PATH
)

if comparables_df.empty:

    st.info(
        "The comparable-listings map is unavailable because "
        "airroi_comparables.csv could not be loaded or is missing "
        "the required city, coordinate, and adjusted RevPAR fields."
    )

else:

    (
        comparable_rows,
        used_property_filter,
    ) = select_comparable_listings(
        bundle=bundle,
        comparables=comparables_df,
        raw_row=raw_row,
        selected_city=selected_city,
        selected_property=selected_property,
        predicted_revpar=revpar,
        count=8,
    )

    if comparable_rows.empty:

        st.info(
            "No historical comparable listings were available "
            "for the selected market."
        )

    else:

        scope_caption = (
            "The search is limited to the selected city and "
            "matching property category."
            if used_property_filter
            else (
                "This market has too few listings in the selected "
                "property category, so the search uses the closest "
                "listings across the selected city."
            )
        )

        st.caption(
            scope_caption
            + " Coordinates are rounded to approximate locations."
        )

        st.markdown(
            '<div style="'
            'font-size:.76rem;'
            'margin:.35rem 0 .7rem 0;'
            '">'
            '<span style="color:#2f9e62;">●</span> '
            'Historical RevPAR above estimate'
            '&nbsp;&nbsp;'
            '<span style="color:#9ca3af;">●</span> '
            'Within 5%'
            '&nbsp;&nbsp;'
            '<span style="color:#b83232;">●</span> '
            'Below estimate'
            '</div>',
            unsafe_allow_html=True,
        )

        comparable_map = build_comparable_map(
            comparable_rows,
            revpar,
        )

        st_folium(
            comparable_map,
            height=560,
            use_container_width=True,
            returned_objects=[],
            key=(
                "comparable_map_"
                f"{selected_city}_"
                f"{selected_property}"
            ),
        )

        comparable_table = pd.DataFrame(
            {
                "Rank": comparable_rows[
                    "Rank"
                ].astype(int),
                "Property category": comparable_rows[
                    "Property group"
                ].map(pretty_label),
                "Actual adjusted RevPAR": comparable_rows[
                    "ttm_adjusted_revpar"
                ],
                "Difference vs prediction": comparable_rows[
                    "RevPAR difference"
                ],
                "Bedrooms": comparable_rows[
                    "bedrooms"
                ],
                "Beds": comparable_rows[
                    "beds"
                ],
                "Baths": comparable_rows[
                    "baths"
                ],
                "Guests": comparable_rows[
                    "guests"
                ],
                "Match index": comparable_rows[
                    "Match index"
                ],
            }
        ).reset_index(
            drop=True
        )

        st.dataframe(
            comparable_table,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Actual adjusted RevPAR": (
                    st.column_config.NumberColumn(
                        format="$%.1f",
                    )
                ),
                "Difference vs prediction": (
                    st.column_config.NumberColumn(
                        format="$%+.1f",
                    )
                ),
                "Match index": (
                    st.column_config.NumberColumn(
                        format="%.0f / 100",
                    )
                ),
            },
        )

        st.caption(
            "The proposed listing is not plotted because the app does not "
            "collect an address or coordinates. Marker locations represent "
            "rounded historical coordinates and should be treated as "
            "approximate."
        )


with st.expander(
    "About This Model",
    expanded=False,
):
    st.markdown(
        """
### What is adjusted TTM RevPAR?

TTM means **trailing 12 months**, while RevPAR means **revenue per available rental night**. RevPAR combines pricing and occupancy into one measure of listing performance. The prediction represents estimated daily revenue performance based on the adjusted TTM RevPAR measure in the source data.

The annualized figure shown in the app is calculated as predicted RevPAR multiplied by 365. It should not be interpreted as profit, cash flow, or a complete investment return.

### What is Elastic Net?

Elastic Net is a regression method that combines Ridge and Lasso regularization. This allows it to work effectively with numerous, potentially related listing characteristics while limiting the influence of noisy predictors. Unlike many more complex machine-learning models, its relationships remain relatively transparent and explainable.

### Why was this model selected?

We evaluated several predictive approaches, including Random Forest, XGBoost, Adaptive Lasso, and Elastic Net. Although the ensemble models achieved somewhat higher validation performance, Elastic Net provided the strongest balance of predictive usefulness, interpretability, and application functionality.

Its transparent structure allows the app to explain individual predictions and evaluate potential listing changes.

### How can this tool help?

The application allows users to:

- Estimate a listing's adjusted RevPAR
- Compare the estimate with the selected city's median
- See which characteristics most influence the prediction
- Explore potential improvements by changing listing features one at a time
- View historical comparable listings with similar model characteristics

### How should the results be interpreted?

Predictions are estimates based on patterns found in historical short-term-rental data. The factors and potential improvements shown represent model associations, not guaranteed or causal effects. Actual performance may also depend on seasonality, local events, pricing strategy, competition, operating quality, and market conditions not fully captured by the model.

This tool evaluates **revenue performance**, not complete investment returns. It does not account for acquisition price, financing, taxes, maintenance, management fees, or other operating expenses.
        """
    )

with st.expander(
    "View exact model input row"
):

    st.dataframe(
        raw_row.T.rename(
            columns={
                0: "Value"
            }
        ),
        use_container_width=True,
    )

st.caption(
    "Predictions are generated from an Elastic Net model trained on "
    "historical STR listings and should be interpreted as estimates "
    "rather than guarantees."
)

st.markdown(
    f"""
    <div class="footer">
        Elastic Net alpha: {bundle["model"].alpha_:.5f}
        &nbsp;|&nbsp; L1 ratio: {bundle["model"].l1_ratio_:.2f}
        &nbsp;|&nbsp; {len(bundle["feature_names"])} predictors
        &nbsp;|&nbsp; scikit-learn {stored_version}
    </div>
    """,
    unsafe_allow_html=True,
)
