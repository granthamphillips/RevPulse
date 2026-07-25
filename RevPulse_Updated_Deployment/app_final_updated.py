"""
RevPulse Streamlit predictor for the fitted Elastic Net model.

Required files in the same directory:
    elastic_net_streamlit_bundle.pkl
    airroi_comparables.csv
    revpulse_interval_calibration.csv
    revpulse_property_score_reference.csv
    revpulse_market_score_reference.csv
    revpulse_feature_category_map.csv

The bundle must contain the fitted model, imputer, scaler, training feature
schema, category metadata, and amenity group definitions. Use the companion
export cell generated for the modeling notebook.
"""

from __future__ import annotations

import base64
import pickle
import re
from io import StringIO
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
INTERVAL_CALIBRATION_PATH = APP_DIR / "revpulse_interval_calibration.csv"
PROPERTY_SCORE_REFERENCE_PATH = APP_DIR / "revpulse_property_score_reference.csv"
MARKET_SCORE_REFERENCE_PATH = APP_DIR / "revpulse_market_score_reference.csv"
FEATURE_CATEGORY_MAP_PATH = APP_DIR / "revpulse_feature_category_map.csv"

COLOR_POSITIVE = "#2f9e62"
COLOR_NEGATIVE = "#b83232"
COLOR_NEUTRAL = "#9ca3af"



EMBEDDED_INTERVAL_CALIBRATION_CSV = r"""
scope,city,property_type_group,n,q50_log,q80_log,q90_log,empirical_50_coverage,empirical_80_coverage,empirical_90_coverage,mae_revpar,rmse_revpar
global,,,2128,0.4051379883088963,0.8297047511498432,1.155985323367767,0.5009398496240601,0.8012218045112782,0.9013157894736842,35.3751435731973,53.09699164514554
city,asheville,,197,0.4325630381577268,0.8801500826191946,1.1506428993611726,0.5076142131979695,0.8121827411167513,0.9137055837563453,31.07616791437921,45.140281096904495
city,carolina_beach,,204,0.396828999815968,0.776932415206602,1.154616880957394,0.5098039215686274,0.8088235294117647,0.9117647058823528,46.1281181814361,62.431869363455455
city,charlotte,,207,0.3421216575506349,0.8430464987763182,1.190766716900029,0.5072463768115942,0.8115942028985508,0.9130434782608696,29.59314256106504,41.6977554320844
city,durham,,168,0.4015603228087903,0.8790482275386273,1.412832442948869,0.5119047619047619,0.8154761904761905,0.9166666666666666,29.569444515218574,45.6643916728782
city,gatlinburg,,224,0.4656132712507914,0.9469434939759802,1.2177828545427165,0.5089285714285714,0.8080357142857143,0.9107142857142856,53.26149151843232,77.29503306934147
city,myrtle_beach,,253,0.4151495803656102,0.901553308042466,1.228376182911493,0.5059288537549407,0.8102766798418972,0.9090909090909092,43.12751514264776,70.28720911149216
city,pigeon_forge,,221,0.3571906339316202,0.7118038557895696,0.910575279024444,0.5067873303167421,0.8099547511312217,0.9095022624434388,33.00973804236717,50.284989086199246
city,raleigh,,224,0.4026191077379284,0.863008008414277,1.2833083455763037,0.5089285714285714,0.8080357142857143,0.9107142857142856,28.511298728622965,38.15565559095159
city,williamsburg,,215,0.5567519344870395,1.1196325352834715,1.3198307552672466,0.5069767441860465,0.8093023255813954,0.9116279069767442,26.222661058573447,36.81020862032668
city,wilmington,,215,0.3712452655974543,0.6606319925393729,0.8942912318117346,0.5069767441860465,0.8093023255813954,0.9116279069767442,30.192216583880416,39.07219660883501
city_property,asheville,entire_place,180,0.4228248006440784,0.8499644470839263,1.1050774572892692,0.5111111111111111,0.8111111111111111,0.9111111111111112,32.30968230185368,46.70862821000067
city_property,carolina_beach,entire_place,201,0.4048143637305728,0.7853272454196176,1.154616880957394,0.5074626865671642,0.8109452736318408,0.9104477611940298,46.7384409355303,62.89262661395716
city_property,charlotte,entire_place,199,0.3385301291362568,0.8209357273516265,1.1851876399171015,0.507537688442211,0.8090452261306532,0.9095477386934674,30.03299628421388,42.31140783952996
city_property,durham,entire_place,157,0.3981867985661402,0.8790482275386273,1.3799161226917045,0.5095541401273885,0.8152866242038217,0.9171974522292994,30.026904422070277,46.525214165550445
city_property,gatlinburg,entire_place,222,0.4492594727686496,0.9469434939759802,1.2177828545427165,0.509009009009009,0.8108108108108109,0.90990990990991,53.3885257855628,77.54756479010801
city_property,myrtle_beach,entire_place,249,0.4151495803656102,0.8940058225544991,1.228376182911493,0.5060240963855421,0.8072289156626506,0.9076305220883534,43.379526945738405,70.65969532838946
city_property,pigeon_forge,entire_place,208,0.3621432145833037,0.7145745660308154,0.910575279024444,0.5096153846153846,0.8125,0.9134615384615384,34.37370867769072,51.684249865313646
city_property,raleigh,entire_place,203,0.3801998356442206,0.7631138252473839,1.144306103777069,0.5073891625615764,0.812807881773399,0.9113300492610836,29.33456320208116,39.31733641106736
city_property,williamsburg,entire_place,180,0.5414907502727302,1.0873896067124855,1.2515944481312666,0.5111111111111111,0.8111111111111111,0.9111111111111112,25.35921363620846,35.75708238357865
city_property,williamsburg,private_room,31,0.6375419289783437,1.2044275692495532,1.958735886611378,0.5483870967741935,0.8709677419354839,0.967741935483871,26.64851443183737,37.41483615743152
city_property,wilmington,entire_place,207,0.3712452655974543,0.6628973343107658,0.8942912318117346,0.5072463768115942,0.8115942028985508,0.9130434782608696,30.678914349831626,39.63170204445655
"""

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

        .brand-parent {
            margin-top: .35rem;
            font-size: .7rem;
            font-weight: 650;
            letter-spacing: .8px;
            text-transform: uppercase;
            color: inherit;
            opacity: .56;
        }

        .insight-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .9rem;
            margin: 1rem 0 .1rem 0;
            align-items: stretch;
        }

        .insight-card {
            box-sizing: border-box;
            min-width: 0;
            height: 100%;
            padding: .95rem 1rem;
            background: rgba(127, 127, 127, .08);
            color: inherit;
            border: 1px solid rgba(127, 127, 127, .28);
            border-radius: 11px;
            box-shadow: 0 8px 22px rgba(0, 0, 0, .08);
        }

        .insight-title {
            font-size: .66rem;
            font-weight: 700;
            letter-spacing: .85px;
            text-transform: uppercase;
            color: inherit;
            opacity: .64;
            margin-bottom: .4rem;
        }

        .insight-value {
            font-size: 1.45rem;
            font-weight: 800;
            line-height: 1.12;
            letter-spacing: -.35px;
            color: inherit;
            overflow-wrap: anywhere;
        }

        .insight-value-positive {
            color: #2f9e62;
        }

        .insight-detail {
            font-size: .72rem;
            line-height: 1.42;
            color: inherit;
            opacity: .70;
            margin-top: .38rem;
            overflow-wrap: anywhere;
        }

        .score-list {
            list-style: none;
            margin: .6rem 0 0 0;
            padding: 0;
        }

        .score-list li {
            display: flex;
            justify-content: space-between;
            gap: .45rem;
            margin: .24rem 0;
            font-size: .72rem;
            line-height: 1.3;
        }

        .score-stars {
            white-space: nowrap;
            letter-spacing: .45px;
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
            margin: 0;
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

        .summary-bottom-space {
            height: 1.35rem;
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

            .insight-grid {
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


@st.cache_data
def load_validation_table(
    path: Path,
    required_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Load a validation export and return an empty frame if it is unusable."""
    if not path.exists():
        return pd.DataFrame()

    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()

    if not set(required_columns).issubset(frame.columns):
        return pd.DataFrame()

    return frame


@st.cache_data
def load_interval_calibration(path: Path) -> pd.DataFrame:
    """Load the 50/80/90% validation calibration, with a built-in data-backed fallback."""
    required_columns = {
        "scope",
        "city",
        "property_type_group",
        "n",
        "q50_log",
        "q80_log",
        "q90_log",
        "empirical_50_coverage",
        "empirical_80_coverage",
        "empirical_90_coverage",
    }

    frame = pd.DataFrame()
    if path.exists():
        try:
            frame = pd.read_csv(path, low_memory=False)
        except Exception:
            frame = pd.DataFrame()

    if frame.empty or not required_columns.issubset(frame.columns):
        frame = pd.read_csv(StringIO(EMBEDDED_INTERVAL_CALIBRATION_CSV))

    return frame


def percentile_rank(
    reference_values: pd.Series,
    value: float,
) -> float:
    """Return a mid-rank empirical percentile on a 0-100 scale."""
    values = pd.to_numeric(reference_values, errors="coerce").dropna()
    if values.empty or not np.isfinite(value):
        return 50.0

    below = float((values < value).sum())
    equal = float((values == value).sum())
    return float(np.clip(100.0 * (below + 0.5 * equal) / len(values), 0.0, 100.0))


def percentile_label(value: float) -> str:
    rounded = int(np.clip(round(value), 0, 100))
    if 10 <= rounded % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(rounded % 10, "th")
    return f"{rounded}{suffix} percentile"


def percentile_stars(value: float) -> str:
    filled = int(np.clip(np.ceil(value / 20.0), 1, 5))
    return "★" * filled + "☆" * (5 - filled)


def choose_reference_group(
    property_reference: pd.DataFrame,
    selected_city: str,
    selected_property: str,
    minimum_exact_rows: int = 20,
) -> tuple[pd.DataFrame, str]:
    """Prefer city/property peers, then city peers, then the global sample."""
    if property_reference.empty:
        return property_reference.copy(), "historical listings"

    exact = property_reference.loc[
        property_reference["city"].eq(selected_city)
        & property_reference["property_type_group"].eq(selected_property)
    ].copy()

    if len(exact) >= minimum_exact_rows:
        return exact, (
            f"{pretty_label(selected_city)} "
            f"{pretty_label(selected_property).lower()} listings"
        )

    city_rows = property_reference.loc[
        property_reference["city"].eq(selected_city)
    ].copy()

    if len(city_rows) >= minimum_exact_rows:
        return city_rows, f"{pretty_label(selected_city)} listings"

    return property_reference.copy(), "all historical listings"


def select_interval_row(
    interval_calibration: pd.DataFrame,
    selected_city: str,
    selected_property: str,
) -> pd.Series | None:
    """Use the most specific conformal calibration group with adequate data."""
    if interval_calibration.empty:
        return None

    exact = interval_calibration.loc[
        interval_calibration["scope"].eq("city_property")
        & interval_calibration["city"].eq(selected_city)
        & interval_calibration["property_type_group"].eq(selected_property)
        & interval_calibration["n"].ge(30)
    ]
    if not exact.empty:
        return exact.iloc[0]

    city = interval_calibration.loc[
        interval_calibration["scope"].eq("city")
        & interval_calibration["city"].eq(selected_city)
    ]
    if not city.empty:
        return city.iloc[0]

    global_rows = interval_calibration.loc[
        interval_calibration["scope"].eq("global")
    ]
    if not global_rows.empty:
        return global_rows.iloc[0]

    return None


def conformal_prediction_range(
    predicted_revpar: float,
    interval_row: pd.Series | None,
    coverage: int = 80,
) -> tuple[float, float, float, int] | None:
    """Create an asymmetric log-scale conformal prediction interval."""
    if interval_row is None or predicted_revpar < 0:
        return None

    quantile_column = f"q{coverage}_log"
    empirical_column = f"empirical_{coverage}_coverage"
    if quantile_column not in interval_row.index:
        return None

    quantile = float(interval_row[quantile_column])
    predicted_log = float(np.log1p(predicted_revpar))
    lower = max(0.0, float(np.expm1(predicted_log - quantile)))
    upper = max(lower, float(np.expm1(predicted_log + quantile)))
    empirical_coverage = float(interval_row.get(empirical_column, coverage / 100.0))
    sample_size = int(interval_row.get("n", 0))
    return lower, upper, empirical_coverage, sample_size


def current_category_contributions(
    raw_contribution_df: pd.DataFrame,
    feature_category_map: pd.DataFrame,
) -> dict[str, float]:
    """Sum the fitted model's log-scale contributions into product categories."""
    if raw_contribution_df.empty or feature_category_map.empty:
        return {}

    merged = raw_contribution_df.merge(
        feature_category_map,
        how="left",
        left_on="Feature",
        right_on="feature",
    )

    category_totals = (
        merged.dropna(subset=["category"])
        .groupby("category")["Contribution"]
        .sum()
    )

    return {
        str(category): float(value)
        for category, value in category_totals.items()
    }


def build_data_backed_scores(
    selected_city: str,
    reference_group: pd.DataFrame,
    market_reference: pd.DataFrame,
    category_contributions: dict[str, float],
    predicted_revpar: float,
) -> dict[str, float]:
    """Calculate transparent empirical percentile scores from validation data."""
    market_percentile = 50.0
    if not market_reference.empty:
        city_row = market_reference.loc[
            market_reference["city"].eq(selected_city)
        ]
        if not city_row.empty:
            market_percentile = float(city_row.iloc[0]["market_percentile"])

    amenities_percentile = percentile_rank(
        reference_group.get(
            "amenities_contribution_log",
            pd.Series(dtype=float),
        ),
        category_contributions.get("Amenities", 0.0),
    )
    policies_percentile = percentile_rank(
        reference_group.get(
            "policies_contribution_log",
            pd.Series(dtype=float),
        ),
        category_contributions.get("Policies", 0.0),
    )
    listing_quality_percentile = percentile_rank(
        reference_group.get(
            "listing_quality_contribution_log",
            pd.Series(dtype=float),
        ),
        category_contributions.get("Listing Quality", 0.0),
    )
    performance_percentile = percentile_rank(
        reference_group.get(
            "actual_adjusted_revpar",
            pd.Series(dtype=float),
        ),
        predicted_revpar,
    )

    overall_score = float(np.mean([
        market_percentile,
        amenities_percentile,
        policies_percentile,
        listing_quality_percentile,
    ]))

    return {
        "Market": market_percentile,
        "Amenities": amenities_percentile,
        "Policies": policies_percentile,
        "Listing Quality": listing_quality_percentile,
        "Performance Percentile": performance_percentile,
        "Overall": overall_score,
    }


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

    # Give the neutral Carto basemap a subtle RevPulse-green cast while
    # leaving markers, labels, popups, and controls at their normal colors.
    comparable_map.get_root().header.add_child(
        folium.Element(
            """
            <style>
                .leaflet-container {
                    background: #e8f1e6;
                }

                .leaflet-tile-pane {
                    filter:
                        sepia(18%)
                        saturate(125%)
                        hue-rotate(62deg)
                        brightness(1.03)
                        contrast(.94);
                }
            </style>
            """
        )
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

interval_calibration = load_interval_calibration(
    INTERVAL_CALIBRATION_PATH
)

property_score_reference = load_validation_table(
    PROPERTY_SCORE_REFERENCE_PATH,
    (
        "city",
        "property_type_group",
        "actual_adjusted_revpar",
        "amenities_contribution_log",
        "policies_contribution_log",
        "listing_quality_contribution_log",
    ),
)

market_score_reference = load_validation_table(
    MARKET_SCORE_REFERENCE_PATH,
    (
        "city",
        "market_percentile",
    ),
)

feature_category_map = load_validation_table(
    FEATURE_CATEGORY_MAP_PATH,
    (
        "feature",
        "category",
    ),
)

validation_exports_ready = all(
    not frame.empty
    for frame in [
        property_score_reference,
        market_score_reference,
        feature_category_map,
    ]
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
            <div class="brand-parent">Powered by AirROI</div>
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
            <div class="brand-parent">Powered by AirROI</div>
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
    expanded=False,
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
    expanded=False,
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
    expanded=False,
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

st.markdown(
    '<div class="section-label">Amenities</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Amenities are grouped exactly as they were during model training. "
    "Use Select all or Clear all within each group to speed up entry."
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

    with amenity_columns[
        group_index % 3
    ]:
        with st.expander(
            group_title,
            expanded=False,
        ):
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
                        f"amenity_v2_{group_feature}_{amenity}"
                    ] = True

            if clear_all_clicked:
                for amenity in members:
                    st.session_state[
                        f"amenity_v2_{group_feature}_{amenity}"
                    ] = False

            checkbox_cols = st.columns(2)

            for index, amenity in enumerate(
                members
            ):
                amenity_key = (
                    f"amenity_v2_"
                    f"{group_feature}_"
                    f"{amenity}"
                )

                with checkbox_cols[index % 2]:
                    amenity_checks[
                        amenity
                    ] = st.checkbox(
                        pretty_label(amenity),
                        value=False,
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
    with st.expander(
        "Standalone amenities",
        expanded=False,
    ):
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

reference_group, reference_group_label = choose_reference_group(
    property_reference=property_score_reference,
    selected_city=selected_city,
    selected_property=selected_property,
)

interval_row = select_interval_row(
    interval_calibration=interval_calibration,
    selected_city=selected_city,
    selected_property=selected_property,
)

prediction_range = conformal_prediction_range(
    predicted_revpar=revpar,
    interval_row=interval_row,
    coverage=50,
)

category_contributions = current_category_contributions(
    raw_contribution_df=raw_contribution_df,
    feature_category_map=feature_category_map,
)

property_scores = build_data_backed_scores(
    selected_city=selected_city,
    reference_group=reference_group,
    market_reference=market_score_reference,
    category_contributions=category_contributions,
    predicted_revpar=revpar,
)

amenity_actions = [
    opportunity
    for opportunity in actionable_opportunities
    if "amenity" in str(opportunity.get("Category", "")).lower()
]

top_amenity_action = (
    max(amenity_actions, key=lambda item: item["Uplift"])
    if amenity_actions
    else None
)


# ============================================================
# RENDER PREDICTION SUMMARY IN THE RESERVED TOP POSITION
# ============================================================

with summary_placeholder:
    st.markdown(
        '<hr class="rule">',
        unsafe_allow_html=True,
    )

    annualized_revenue = revpar * 365
    monthly_revenue = annualized_revenue / 12
    city_median_annualized_revenue = selected_city_median * 365

    summary_left, summary_right = st.columns(
        [1.28, 0.72],
        gap="large",
        vertical_alignment="center",
    )

    with summary_left:
        revenue_card_html = (
            f'<div class="revenue-card">'
            f'<div class="revenue-eyebrow">Annualized revenue potential</div>'
            f'<div class="revenue-main">${annualized_revenue:,.0f}</div>'
            f'<div class="revenue-sub">Adjusted RevPAR × 365 available nights</div>'
            f'<div class="revenue-grid">'
            f'<div class="revenue-stat">'
            f'<div class="revenue-stat-value">${monthly_revenue:,.0f}</div>'
            f'<div class="revenue-stat-label">Monthly equivalent</div>'
            f'</div>'
            f'<div class="revenue-stat">'
            f'<div class="revenue-stat-value">${city_median_annualized_revenue:,.0f}</div>'
            f'<div class="revenue-stat-label">{selected_city_label} median annualized</div>'
            f'</div>'
            f'</div>'
            f'<div class="revenue-context">'
            f'{pretty_label(selected_property)} in {pretty_label(selected_city)} · '
            f'occupancy-adjusted estimate, not gross booking revenue'
            f'</div>'
            f'</div>'
        )

        st.markdown(
            revenue_card_html,
            unsafe_allow_html=True,
        )

    with summary_right:
        st.markdown(
            '<div class="section-label">Prediction</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="hero-value">${revpar:,.0f}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="hero-label">Predicted adjusted RevPAR per night</div>',
            unsafe_allow_html=True,
        )

        direction = "above" if difference >= 0 else "below"
        sign = "+" if difference >= 0 else "-"

        st.markdown(
            f'<div class="hero-context">'
            f'{sign}${abs(difference):,.0f} '
            f'({sign}{abs(difference_pct):.0f}%) '
            f'{direction} the {selected_city_label} median'
            f'</div>',
            unsafe_allow_html=True,
        )

    if prediction_range is not None:
        (
            lower_revpar,
            upper_revpar,
            empirical_coverage,
            interval_sample_size,
        ) = prediction_range
        lower_annual = lower_revpar * 365
        upper_annual = upper_revpar * 365
        interval_value = f'${lower_annual:,.0f} – ${upper_annual:,.0f}'
        interval_detail = (
            f'Middle 50% out-of-sample predicted range · '
            f'{empirical_coverage * 100:.1f}% observed calibration coverage '
            f'across {interval_sample_size:,} validation rows.'
        )
    else:
        interval_value = 'Unavailable'
        interval_detail = 'No compatible 50% validation calibration row was found for this market.'

    performance_percentile = property_scores.get("Performance Percentile", 50.0)
    amenity_strength_score = property_scores.get("Amenities", 50.0)
    overall_property_score = property_scores.get("Overall", 50.0)

    amenity_detail = (
        f'Model-based amenity contribution versus '
        f'{escape(reference_group_label)} (n={len(reference_group):,}).'
    )

    score_rows = "".join(
        f'<li><span>{escape(display_label)}</span>'
        f'<span class="score-stars">{percentile_stars(property_scores.get(score_key, 50.0))}</span></li>'
        for display_label, score_key in [
            ("Market", "Market"),
            ("Amenities", "Amenities"),
            ("Policies", "Policies"),
            ("Listing Setup", "Listing Quality"),
        ]
    )

    insight_cards_html = (
        '<div class="insight-grid">'
        '<div class="insight-card">'
        '<div class="insight-title">Typical revenue range</div>'
        f'<div class="insight-value">{interval_value}</div>'
        f'<div class="insight-detail">{escape(interval_detail)}</div>'
        '</div>'
        '<div class="insight-card">'
        '<div class="insight-title">Market percentile</div>'
        f'<div class="insight-value insight-value-positive">{performance_percentile:.0f}th</div>'
        f'<div class="insight-detail">Predicted RevPAR versus actual historical performance among {escape(reference_group_label)} (n={len(reference_group):,}).</div>'
        '</div>'
        '<div class="insight-card">'
        '<div class="insight-title">Amenity strength percentile</div>'
        f'<div class="insight-value insight-value-positive">{amenity_strength_score:.0f}th</div>'
        f'<div class="insight-detail">{amenity_detail}</div>'
        '</div>'
        '<div class="insight-card">'
        '<div class="insight-title">Property score</div>'
        f'<div class="insight-value">{overall_property_score:.0f} / 100</div>'
        f'<ul class="score-list">{score_rows}</ul>'
        f'<div class="insight-detail">Average of four empirical category percentiles.</div>'
        '</div>'
        '</div>'
    )

    st.markdown(
        insight_cards_html,
        unsafe_allow_html=True,
    )

    if not validation_exports_ready:
        st.warning(
            "One or more RevPulse score-reference files are missing. "
            "The percentile score cards cannot be fully calculated."
        )

    st.markdown(
        '<div class="summary-bottom-space"></div>',
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

### How is the revenue prediction range calculated?

The displayed range is the **middle 50% out-of-sample prediction range**, calibrated from nested five-fold cross-validation errors on 2,128 historical listings. It is intentionally presented as a typical range rather than a broad high-coverage interval. The app uses the most specific reliable calibration group available: selected city and property type when there are at least 30 validation rows, then city, then the global validation sample.

Because the Elastic Net predicts `log1p(RevPAR)`, the interval is calculated symmetrically around the prediction in log space and transformed back to dollars. This creates the appropriate asymmetric dollar range. The annual range is the resulting lower and upper RevPAR bounds multiplied by 365.

### How are the Property Score and amenity strength percentile calculated?

The Property Score is based on empirical percentiles rather than hand-assigned quality rules. Amenities, Policies, and Listing Setup are calculated by summing the current listing's fitted model contributions in each category and comparing them with historical listings in the closest available city/property reference group. Internally, the validation export retains the original `Listing Quality` category name, but the interface labels it **Listing Setup** because it reflects capacity, photo count, and property type rather than subjective quality. Market reflects how the selected city's median adjusted RevPAR ranks across the ten modeled markets.

The overall Property Score is the simple average of those four category percentiles. The amenity card shows the current listing's **amenity strength percentile** directly. Potential amenity improvements remain in the separate listing-improvements section, where each change is tested by rerunning the fitted model one input at a time.

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
