

# ============================================================
# URIP — NAIROBI URBAN RESILIENCE INTELLIGENCE PLATFORM
# Streamlit dashboard for the frozen URIP analytical model
#
# IMPORTANT:
# This application VISUALIZES frozen analytical outputs.
# It does NOT recalculate the URIP model.
#
# Expected files:
#   app.py
#   URIP_MODEL.pkl.gz
#
# Optional:
#   URIP_MODEL.pkl
# ============================================================

from pathlib import Path
import base64
import io
import pickle
import gzip
import html
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium import plugins
from shapely.geometry import LineString
import streamlit as st
from streamlit_folium import st_folium

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="URIP — Urban Resilience Intelligence Platform",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN SYSTEM
# ============================================================

NAVY = "#0B2545"
NAVY_2 = "#15335C"
TEAL = "#0EA5A8"
BLUE = "#2563EB"
GREEN = "#16A34A"
AMBER = "#F59E0B"
ORANGE = "#F97316"
RED = "#DC2626"
DARK_RED = "#7F0000"
PURPLE = "#7C3AED"

WHITE = "#FFFFFF"
OFFWHITE = "#F8FAFC"
LIGHT = "#F1F5F9"
BORDER = "#E2E8F0"
TEXT = "#0F172A"
MUTED = "#64748B"

ROAD_COLORS = {
    "Passable": "#2ca25f",
    "Minor Disruption": "#fdae6b",
    "Severe Disruption": "#e34a33",
    "Closed": "#7f0000",
    "No Data": "#969696",
}

SCENARIO_COLORS = {
    "Normal": GREEN,
    "Moderate Rainfall": AMBER,
    "Heavy Rainfall": ORANGE,
    "Severe Rainfall": RED,
}

SCENARIOS = [
    "Normal",
    "Moderate Rainfall",
    "Heavy Rainfall",
    "Severe Rainfall",
]

SCENARIO_RAINFALL = {
    "Normal": 10,
    "Moderate Rainfall": 30,
    "Heavy Rainfall": 50,
    "Severe Rainfall": 75,
}

ANALYSIS_CRS = "EPSG:32737"
DISPLAY_CRS = "EPSG:4326"

MAP_CENTER = [-1.2864, 36.8172]
MAP_ZOOM = 13


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    f"""
    <style>

    .stApp {{
        background: {OFFWHITE};
    }}

    [data-testid="stSidebar"] {{
        background: {NAVY};
    }}

    [data-testid="stSidebar"] * {{
        color: white;
    }}

    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }}

    .urip-header {{
        background: linear-gradient(
            135deg,
            {NAVY},
            {NAVY_2}
        );
        color: white;
        padding: 24px 28px;
        border-radius: 14px;
        margin-bottom: 18px;
    }}

    .urip-title {{
        font-size: 30px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }}

    .urip-subtitle {{
        font-size: 14px;
        margin-top: 7px;
        color: #CBD5E1;
    }}

    .section-header {{
        margin-top: 22px;
        margin-bottom: 10px;
    }}

    .section-title {{
        font-size: 21px;
        font-weight: 750;
        color: {TEXT};
    }}

    .section-subtitle {{
        color: {MUTED};
        font-size: 13px;
        margin-top: 3px;
    }}

    .kpi-card {{
        background: white;
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 14px 15px;
        min-height: 105px;
        box-shadow: 0 1px 3px rgba(15,23,42,0.04);
    }}

    .kpi-label {{
        font-size: 11px;
        color: {MUTED};
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 700;
    }}

    .kpi-value {{
        font-size: 23px;
        font-weight: 800;
        color: {TEXT};
        margin-top: 5px;
    }}

    .kpi-note {{
        font-size: 11px;
        color: {MUTED};
        margin-top: 3px;
    }}

    .status-card {{
        background: white;
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 15px;
        min-height: 145px;
    }}

    .status-scenario {{
        font-size: 12px;
        font-weight: 700;
        color: {MUTED};
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }}

    .status-value {{
        font-size: 20px;
        font-weight: 800;
        color: {TEXT};
        margin-top: 6px;
    }}

    .facility-card {{
        border-radius: 12px;
        padding: 16px;
        margin-top: 12px;
        margin-bottom: 7px;
        background: white;
    }}

    .route-card {{
        margin-left: 20px;
        border-radius: 8px;
        padding: 12px 15px;
        margin-bottom: 8px;
        background: white;
    }}

    .badge {{
        display: inline-block;
        color: white;
        padding: 4px 9px;
        border-radius: 999px;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.4px;
    }}

    .alert-card {{
        padding: 14px 17px;
        border-radius: 10px;
        margin: 12px 0;
        font-size: 14px;
    }}

    .model-note {{
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 10px;
        padding: 13px 16px;
        color: #1E3A8A;
        font-size: 12px;
        margin-top: 20px;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_GZ = BASE_DIR / "URIP_MODEL.pkl.gz"
MODEL_PKL = BASE_DIR / "URIP_MODEL.pkl"


@st.cache_resource
def load_model():

    if MODEL_GZ.exists():

        with gzip.open(MODEL_GZ, "rb") as f:
            model = pickle.load(f)

        return model, MODEL_GZ.name

    if MODEL_PKL.exists():

        with open(MODEL_PKL, "rb") as f:
            model = pickle.load(f)

        return model, MODEL_PKL.name

    raise FileNotFoundError(
        "No frozen URIP model was found. "
        "Upload URIP_MODEL.pkl.gz to the repository."
    )


try:

    MODEL, MODEL_FILENAME = load_model()

except Exception as exc:

    st.error("URIP model could not be loaded.")

    st.code(str(exc))

    st.stop()


# ============================================================
# MODEL VALIDATION
# ============================================================

REQUIRED_TOP_LEVEL = [
    "metadata",
    "scenarios",
    "flood",
    "roads",
    "facilities",
    "population",
    "incidents",
    "facility_options",
    "routes",
    "traffic",
    "emergency",
    "summary",
]

missing_sections = [
    key
    for key in REQUIRED_TOP_LEVEL
    if key not in MODEL
]

if missing_sections:

    st.error(
        "The frozen URIP model is incomplete."
    )

    st.write(
        "Missing sections:",
        missing_sections
    )

    st.stop()


# ============================================================
# AUTHORITATIVE FROZEN TABLES
# ============================================================

EMERGENCY = MODEL["emergency"]

INCIDENT_REPORT = EMERGENCY["incident_report"]
INCIDENT_COMPARISON = EMERGENCY["incident_comparison"]
SITUATION_REPORT = EMERGENCY["situation_report"]
EMERGENCY_KPIS = EMERGENCY["kpis"]

FACILITY_OPTIONS = MODEL["facility_options"]

ROUTES_ALTERNATIVES = MODEL["routes"]["alternatives"]
ROUTES_RECOMMENDED = MODEL["routes"]["recommended"]

INCIDENT_POINTS = MODEL["incidents"]["points"]
INCIDENT_FLOOD = MODEL["incidents"]["flood"]

FACILITIES = MODEL["facilities"]
TRAFFIC = MODEL["traffic"]


# ============================================================
# DATA ACCESS HELPERS
# ============================================================

def safe_float(value, default=0.0):

    try:

        if pd.isna(value):
            return default

        return float(value)

    except Exception:

        return default


def safe_int(value, default=0):

    try:

        if pd.isna(value):
            return default

        return int(value)

    except Exception:

        return default


def get_incident_ids():

    if (
        INCIDENT_POINTS is not None
        and not INCIDENT_POINTS.empty
        and "incident_id" in INCIDENT_POINTS.columns
    ):

        return sorted(
            INCIDENT_POINTS["incident_id"]
            .astype(str)
            .unique()
            .tolist()
        )

    if "incident_id" in INCIDENT_REPORT.columns:

        return sorted(
            INCIDENT_REPORT["incident_id"]
            .astype(str)
            .unique()
            .tolist()
        )

    return []


def get_incident_report(
    scenario,
    incident_id
):

    if INCIDENT_REPORT.empty:
        return pd.DataFrame()

    return INCIDENT_REPORT[
        (INCIDENT_REPORT["scenario"] == scenario)
        &
        (INCIDENT_REPORT["incident_id"] == incident_id)
    ].copy()


def get_incident_comparison(
    scenario,
    incident_id
):

    if INCIDENT_COMPARISON.empty:
        return pd.DataFrame()

    return INCIDENT_COMPARISON[
        (INCIDENT_COMPARISON["scenario"] == scenario)
        &
        (INCIDENT_COMPARISON["incident_id"] == incident_id)
    ].copy()


def get_situation_report(scenario):

    if SITUATION_REPORT.empty:
        return None

    rows = SITUATION_REPORT[
        SITUATION_REPORT["scenario"] == scenario
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


def get_emergency_kpi(scenario):

    if EMERGENCY_KPIS.empty:
        return None

    rows = EMERGENCY_KPIS[
        EMERGENCY_KPIS["scenario"] == scenario
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


def get_facility_options(
    scenario,
    incident_id
):

    if FACILITY_OPTIONS.empty:
        return pd.DataFrame()

    result = FACILITY_OPTIONS[
        (FACILITY_OPTIONS["scenario"] == scenario)
        &
        (FACILITY_OPTIONS["incident_id"] == incident_id)
    ].copy()

    if "facility_rank" in result.columns:

        result = result.sort_values(
            "facility_rank"
        )

    return result


def get_route_options(
    scenario,
    incident_id,
    facility_rank=1,
    route_rank=None
):

    if ROUTES_ALTERNATIVES.empty:
        return pd.DataFrame()

    result = ROUTES_ALTERNATIVES[
        (ROUTES_ALTERNATIVES["scenario"] == scenario)
        &
        (ROUTES_ALTERNATIVES["incident_id"] == incident_id)
        &
        (ROUTES_ALTERNATIVES["facility_rank"] == facility_rank)
    ].copy()

    if route_rank is not None:

        result = result[
            result["route_rank"] == route_rank
        ].copy()

    if "route_rank" in result.columns:

        result = result.sort_values(
            "route_rank"
        )

    return result


def get_roads(scenario):

    if scenario not in TRAFFIC:
        return gpd.GeoDataFrame()

    roads = TRAFFIC[scenario]

    if roads is None:
        return gpd.GeoDataFrame()

    return roads.copy()


def get_facilities(scenario):

    if scenario not in FACILITIES:
        return gpd.GeoDataFrame()

    facilities = FACILITIES[scenario]

    if facilities is None:
        return gpd.GeoDataFrame()

    return facilities.copy()


def get_incidents():

    if INCIDENT_POINTS is None:
        return gpd.GeoDataFrame()

    return INCIDENT_POINTS.copy()


# ============================================================
# CBD POPULATION EXTRACTION
#
# Uses frozen population objects only.
# No population recalculation is performed.
# ============================================================

def _population_from_dataframe(df, scenario):

    if not isinstance(df, pd.DataFrame):
        return None

    if "scenario" not in df.columns:
        return None

    rows = df[
        df["scenario"] == scenario
    ]

    if rows.empty:
        return None

    row = rows.iloc[0]

    candidates = [
        "flood_exposed_population",
        "access_disrupted_population",
        "priority_population",
        "high_priority_population",
    ]

    result = {}

    for col in candidates:

        if col in row.index:
            result[col] = safe_float(
                row[col]
            )

    if result:
        return result

    return None


def get_cbd_population_metrics(scenario):

    """
    Read frozen CBD-wide population outputs.

    Priority order:
    1. explicitly named authoritative_population
    2. population exposure/access/priority result tables
    3. summary tables
    """

    population_section = MODEL.get(
        "population",
        {}
    )

    if not isinstance(
        population_section,
        dict
    ):
        population_section = {}

    candidate_names = [
        "authoritative_population",
        "population_exposure_results",
        "population_access_results",
        "population_priority_results",
        "population",
        "pop_result",
        "state_population",
    ]

    result = {}

    for name in candidate_names:

        obj = population_section.get(name)

        if isinstance(obj, pd.DataFrame):

            if "scenario" not in obj.columns:
                continue

            rows = obj[
                obj["scenario"] == scenario
            ]

            if rows.empty:
                continue

            # Look for aggregate rows if available.
            row = rows.iloc[0]

            for key in [
                "flood_exposed_population",
                "access_disrupted_population",
                "newly_unreachable_population",
                "priority_population",
                "high_priority_population",
            ]:

                if key in row.index:

                    result[key] = safe_float(
                        row[key]
                    )

    # If some metrics were not available from named tables,
    # search the summary section.
    summary = MODEL.get("summary")

    if isinstance(summary, dict):

        for obj in summary.values():

            if not isinstance(
                obj,
                pd.DataFrame
            ):
                continue

            if "scenario" not in obj.columns:
                continue

            rows = obj[
                obj["scenario"] == scenario
            ]

            if rows.empty:
                continue

            row = rows.iloc[0]

            for key in [
                "flood_exposed_population",
                "access_disrupted_population",
                "newly_unreachable_population",
                "priority_population",
                "high_priority_population",
            ]:

                if (
                    key not in result
                    and key in row.index
                ):

                    result[key] = safe_float(
                        row[key]
                    )

    # Safe defaults.
    for key in [
        "flood_exposed_population",
        "access_disrupted_population",
        "newly_unreachable_population",
        "priority_population",
        "high_priority_population",
    ]:

        result.setdefault(
            key,
            0.0
        )

    return result


# ============================================================
# TRAFFIC METRICS
# ============================================================

def get_traffic_metrics(scenario):

    roads = get_roads(scenario)

    if roads.empty:
        return {
            "affected": 0,
            "closed": 0,
            "mean_vc": 0,
        }

    affected = 0
    closed = 0

    if "affected_pct" in roads.columns:

        affected = int(
            (
                roads["affected_pct"]
                .fillna(0)
                > 0
            ).sum()
        )

    elif "access_status" in roads.columns:

        affected = int(
            (
                roads["access_status"]
                .astype(str)
                .isin([
                    "Minor Disruption",
                    "Severe Disruption",
                    "Closed",
                ])
            ).sum()
        )

    if "passability" in roads.columns:

        closed = int(
            (
                roads["passability"]
                .fillna(1)
                <= 0
            ).sum()
        )

    elif "access_status" in roads.columns:

        closed = int(
            (
                roads["access_status"]
                .astype(str)
                == "Closed"
            ).sum()
        )

    if "final_vc_ratio" in roads.columns:

        vc = pd.to_numeric(
            roads["final_vc_ratio"],
            errors="coerce"
        )

        mean_vc = safe_float(
            vc.replace(
                [np.inf, -np.inf],
                np.nan
            ).mean()
        )

    else:

        mean_vc = 0.0

    return {
        "affected": affected,
        "closed": closed,
        "mean_vc": mean_vc,
    }


# ============================================================
# FACILITY METRICS
# ============================================================

def get_facility_metrics(scenario):

    facilities = get_facilities(scenario)

    if facilities.empty:
        return {
            "affected": 0,
            "unreachable": 0,
        }

    if "operationally_affected" in facilities.columns:

        affected = int(
            pd.to_numeric(
                facilities[
                    "operationally_affected"
                ],
                errors="coerce"
            )
            .fillna(0)
            .astype(bool)
            .sum()
        )

    else:

        affected = 0

    if "access_status" in facilities.columns:

        unreachable = int(
            (
                facilities[
                    "access_status"
                ]
                .astype(str)
                .str.lower()
                .isin([
                    "unreachable",
                    "no access",
                ])
            ).sum()
        )

    else:

        unreachable = 0

    return {
        "affected": affected,
        "unreachable": unreachable,
    }


# ============================================================
# INCIDENT METRICS
# ============================================================

def get_incident_metrics(
    scenario,
    incident_id
):

    report = get_incident_report(
        scenario,
        incident_id
    )

    if report.empty:

        return {}

    row = report.iloc[0]

    return {
        "flood_impact": safe_float(
            row.get(
                "incident_flood_impact"
            )
        ),
        "flood_class": str(
            row.get(
                "incident_flood_class",
                "Unknown"
            )
        ),
        "population_in_catchment": safe_float(
            row.get(
                "population_in_catchment"
            )
        ),
        "flood_exposed_population": safe_float(
            row.get(
                "flood_exposed_population"
            )
        ),
        "access_disrupted_population": safe_float(
            row.get(
                "access_disrupted_population"
            )
        ),
        "priority_population": safe_float(
            row.get(
                "priority_population"
            )
        ),
        "high_priority_population": safe_float(
            row.get(
                "high_priority_population"
            )
        ),
        "recommended_facility": str(
            row.get(
                "recommended_facility_name",
                "Unavailable"
            )
        ),
        "recommended_facility_id": str(
            row.get(
                "recommended_facility_id",
                ""
            )
        ),
        "recommended_route_time": safe_float(
            row.get(
                "recommended_route_response_time_min"
            )
        ),
        "recommended_route_length": safe_float(
            row.get(
                "recommended_route_length_m"
            )
        ),
    }


# ============================================================
# GEO HELPERS
# ============================================================

def ensure_display_crs(gdf):

    if gdf is None:
        return gpd.GeoDataFrame()

    if len(gdf) == 0:
        return gdf

    result = gdf.copy()

    if result.crs is None:

        result = result.set_crs(
            ANALYSIS_CRS,
            allow_override=True
        )

    if str(result.crs) != DISPLAY_CRS:

        result = result.to_crs(
            DISPLAY_CRS
        )

    return result


def route_geometry_to_gdf(
    route_row
):

    geometry = route_row.get(
        "geometry"
    )

    if geometry is None:
        return None

    try:

        if isinstance(
            geometry,
            gpd.GeoSeries
        ):

            geometry = geometry.iloc[0]

        return gpd.GeoDataFrame(
            [route_row],
            geometry=[geometry],
            crs=ANALYSIS_CRS
        ).to_crs(
            DISPLAY_CRS
        )

    except Exception:

        return None


# ============================================================
# FLOOD IMAGE
# ============================================================

def flood_overlay(
    fmap,
    scenario
):

    flood_obj = MODEL.get(
        "flood",
        {}
    )

    if not isinstance(
        flood_obj,
        dict
    ):
        return

    flood_array = flood_obj.get(
        scenario
    )

    if flood_array is None:
        return

    try:

        array = np.asarray(
            flood_array,
            dtype=float
        )

        if array.ndim != 2:
            return

        valid = np.isfinite(array)

        if not valid.any():
            return

        vmin = float(
            np.nanpercentile(
                array[valid],
                5
            )
        )

        vmax = float(
            np.nanpercentile(
                array[valid],
                95
            )
        )

        if vmax <= vmin:
            vmax = vmin + 1

        import matplotlib

        cmap = matplotlib.colormaps.get_cmap(
            "turbo"
        )

        normalized = np.clip(
            (array - vmin)
            / (vmax - vmin),
            0,
            1
        )

        rgba = (
            cmap(normalized)
            * 255
        ).astype(np.uint8)

        rgba[..., 3] = np.where(
            valid,
            155,
            0
        ).astype(np.uint8)

        from PIL import Image

        image = Image.fromarray(
            rgba,
            mode="RGBA"
        )

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="PNG"
        )

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode()

        bounds = [
            [-1.3149999999999973, 36.79],
            [-1.2600000000000051, 36.85],
        ]

        folium.raster_layers.ImageOverlay(
            image=(
                "data:image/png;base64,"
                + encoded
            ),
            bounds=bounds,
            opacity=0.48,
            interactive=True,
            cross_origin=False,
            zindex=2,
            name=f"Flood scenario — {scenario}",
        ).add_to(fmap)

    except Exception:

        pass


# ============================================================
# ROAD MAP
# ============================================================

def add_road_layer(
    fmap,
    roads,
    scenario
):

    if roads.empty:
        return

    roads_display = ensure_display_crs(
        roads
    )

    for _, row in roads_display.iterrows():

        geometry = row.geometry

        if geometry is None:
            continue

        status = str(
            row.get(
                "access_status",
                "No Data"
            )
        )

        color = ROAD_COLORS.get(
            status,
            ROAD_COLORS["No Data"]
        )

        weight = 2.5

        if status == "Closed":
            weight = 4

        elif status == "Severe Disruption":
            weight = 3.5

        popup = folium.Popup(
            f"""
            <b>Road ID:</b>
            {html.escape(str(row.get("road_id", "")))}<br>
            <b>Status:</b>
            {html.escape(status)}<br>
            <b>Flood impact:</b>
            {safe_float(row.get("mean_flood_impact")):.3f}<br>
            <b>Passability:</b>
            {safe_float(row.get("passability")):.2f}<br>
            <b>Traffic V/C:</b>
            {safe_float(row.get("final_vc_ratio")):.2f}<br>
            <b>Flood-adjusted time:</b>
            {safe_float(row.get("traffic_adjusted_time_min")):.2f} min
            """,
            max_width=320,
        )

        folium.GeoJson(
            geometry.__geo_interface__,
            style_function=lambda feature,
            color=color,
            weight=weight,
            opacity=0.78: {
                "color": color,
                "weight": weight,
                "opacity": opacity,
            },
            tooltip=folium.Tooltip(
                f"{status}"
            ),
            popup=popup,
            name="Road network",
        ).add_to(fmap)


# ============================================================
# FACILITY MAP
# ============================================================

def add_facility_layer(
    fmap,
    facilities
):

    if facilities.empty:
        return

    facilities_display = ensure_display_crs(
        facilities
    )

    layer = folium.FeatureGroup(
        name="Emergency facilities",
        show=True,
    )

    for _, row in facilities_display.iterrows():

        point = row.geometry

        if point is None:
            continue

        facility_type = str(
            row.get(
                "facility_type",
                "Facility"
            )
        )

        name = str(
            row.get(
                "name",
                "Unknown"
            )
        )

        status = str(
            row.get(
                "access_status",
                "Unknown"
            )
        )

        if status.lower() in [
            "unreachable",
            "no access",
        ]:

            color = RED

        elif row.get(
            "operationally_affected",
            False
        ):

            color = ORANGE

        else:

            color = BLUE

        folium.CircleMarker(
            location=[
                point.y,
                point.x
            ],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=1,
            tooltip=(
                f"{name} • {facility_type}"
            ),
            popup=folium.Popup(
                f"""
                <b>{html.escape(name)}</b><br>
                Type: {html.escape(facility_type)}<br>
                Status: {html.escape(status)}<br>
                Response:
                {safe_float(row.get("response_time")):.2f} min
                """,
                max_width=300,
            ),
        ).add_to(layer)

    layer.add_to(fmap)


# ============================================================
# INCIDENT MAP
# ============================================================

def add_incident_layer(
    fmap,
    incident_points,
    selected_incident
):

    if incident_points.empty:
        return

    incidents_display = ensure_display_crs(
        incident_points
    )

    layer = folium.FeatureGroup(
        name="Incidents",
        show=True,
    )

    for _, row in incidents_display.iterrows():

        point = row.geometry

        if point is None:
            continue

        incident_id = str(
            row.get(
                "incident_id",
                ""
            )
        )

        selected = (
            incident_id
            == selected_incident
        )

        color = RED if selected else PURPLE
        radius = 9 if selected else 6

        folium.CircleMarker(
            location=[
                point.y,
                point.x
            ],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            weight=2,
            tooltip=incident_id,
            popup=folium.Popup(
                f"""
                <b>Incident:</b>
                {html.escape(incident_id)}
                """,
                max_width=250,
            ),
        ).add_to(layer)

    layer.add_to(fmap)


# ============================================================
# ROUTE MAP
# ============================================================

def add_selected_routes(
    fmap,
    scenario,
    incident_id,
    selected_facility_rank=1
):

    routes = get_route_options(
        scenario,
        incident_id,
        selected_facility_rank
    )

    if routes.empty:
        return

    layer = folium.FeatureGroup(
        name="Route alternatives",
        show=True,
    )

    for _, route in routes.iterrows():

        route_rank = safe_int(
            route.get("route_rank")
        )

        route_gdf = route_geometry_to_gdf(
            route
        )

        if route_gdf is None:
            continue

        geometry = route_gdf.geometry.iloc[0]

        if geometry is None:
            continue

        recommended = (
            route_rank == 1
            and selected_facility_rank == 1
        )

        color = GREEN if recommended else BLUE
        weight = 6 if recommended else 3

        popup = folium.Popup(
            f"""
            <b>Route {route_rank}</b><br>
            Response:
            {safe_float(route.get("response_time_min")):.2f} min<br>
            Length:
            {safe_float(route.get("route_length_m")):,.0f} m<br>
            Affected segments:
            {safe_int(route.get("affected_segments"))}<br>
            Closed segments:
            {safe_int(route.get("closed_segments"))}<br>
            Delay:
            +{safe_float(route.get("response_delay_min")):.2f} min
            """,
            max_width=300,
        )

        folium.GeoJson(
            geometry.__geo_interface__,
            style_function=lambda feature,
            color=color,
            weight=weight: {
                "color": color,
                "weight": weight,
                "opacity": 0.9,
            },
            tooltip=(
                f"Route {route_rank}"
                + (
                    " — RECOMMENDED"
                    if recommended
                    else ""
                )
            ),
            popup=popup,
        ).add_to(layer)

    layer.add_to(fmap)


# ============================================================
# MAP LEGEND
# ============================================================

def add_road_legend(fmap):

    legend_html = """
    <div style="
        position: fixed;
        bottom: 25px;
        left: 25px;
        z-index: 9999;
        background: white;
        padding: 12px 14px;
        border-radius: 9px;
        border: 1px solid #CBD5E1;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        font-size: 12px;
    ">

        <div style="
            font-weight:700;
            margin-bottom:8px;
        ">
            Road passability
        </div>

        <div>
            <span style="
                display:inline-block;
                width:12px;
                height:12px;
                background:#2ca25f;
                margin-right:6px;
            "></span>
            Passable
        </div>

        <div>
            <span style="
                display:inline-block;
                width:12px;
                height:12px;
                background:#fdae6b;
                margin-right:6px;
            "></span>
            Minor disruption
        </div>

        <div>
            <span style="
                display:inline-block;
                width:12px;
                height:12px;
                background:#e34a33;
                margin-right:6px;
            "></span>
            Severe disruption
        </div>

        <div>
            <span style="
                display:inline-block;
                width:12px;
                height:12px;
                background:#7f0000;
                margin-right:6px;
            "></span>
            Closed
        </div>

    </div>
    """

    fmap.get_root().html.add_child(
        folium.Element(
            legend_html
        )
    )


# ============================================================
# DASHBOARD HEADER
# ============================================================

st.markdown(
    """
    <div class="urip-header">

        <div class="urip-title">
            URIP — Urban Resilience Intelligence Platform
        </div>

        <div class="urip-subtitle">
            Nairobi urban flood, traffic and emergency-response intelligence
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.markdown(
    """
    <div style="
        font-size:24px;
        font-weight:800;
        margin-bottom:3px;
    ">
        URIP
    </div>

    <div style="
        font-size:12px;
        color:#CBD5E1;
        margin-bottom:20px;
    ">
        Scenario intelligence
    </div>
    """,
    unsafe_allow_html=True,
)


selected_scenario = st.sidebar.selectbox(
    "Rainfall scenario",
    SCENARIOS,
    index=0,
)


incident_ids = get_incident_ids()

if not incident_ids:

    st.error(
        "No incident records are available in the frozen model."
    )

    st.stop()


selected_incident = st.sidebar.selectbox(
    "Incident",
    incident_ids,
    index=0,
)


map_mode = st.sidebar.radio(
    "Map mode",
    [
        "Flood Scenario",
        "Emergency Response",
    ],
)


# ============================================================
# SELECTED SCENARIO DATA
# ============================================================

situation = get_situation_report(
    selected_scenario
)

emergency_kpi = get_emergency_kpi(
    selected_scenario
)

incident_metrics = get_incident_metrics(
    selected_scenario,
    selected_incident
)

traffic_metrics = get_traffic_metrics(
    selected_scenario
)

facility_metrics = get_facility_metrics(
    selected_scenario
)

population_metrics = get_cbd_population_metrics(
    selected_scenario
)


# ============================================================
# KPI VALUES
# ============================================================

if situation is not None:

    mean_flood = safe_float(
        situation.get(
            "mean_incident_flood_impact"
        )
    )

    mean_response = safe_float(
        situation.get(
            "mean_response_time_min"
        )
    )

else:

    mean_flood = 0
    mean_response = 0


roads_affected = traffic_metrics[
    "affected"
]

roads_closed = traffic_metrics[
    "closed"
]

mean_vc = traffic_metrics[
    "mean_vc"
]

facilities_affected = facility_metrics[
    "affected"
]

population_exposed = population_metrics[
    "flood_exposed_population"
]

population_access = population_metrics[
    "access_disrupted_population"
]

population_priority = population_metrics[
    "priority_population"
]

population_high_priority = population_metrics[
    "high_priority_population"
]


# ============================================================
# SCENARIO SUMMARY
# ============================================================

st.markdown(
    """
    <div class="section-header">

        <div class="section-title">
            Scenario intelligence
        </div>

        <div class="section-subtitle">
            Frozen analytical outputs for the selected rainfall scenario
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


kpi_data = [
    (
        "Flood impact",
        f"{mean_flood:.3f}",
        selected_scenario,
    ),
    (
        "Roads affected",
        f"{roads_affected:,}",
        "road segments",
    ),
    (
        "Roads closed",
        f"{roads_closed:,}",
        "road segments",
    ),
    (
        "Mean V/C",
        f"{mean_vc:.2f}",
        "network ratio",
    ),
    (
        "Facilities affected",
        f"{facilities_affected:,}",
        "operationally affected",
    ),
    (
        "Population exposed",
        f"{population_exposed:,.0f}",
        "CBD-wide",
    ),
    (
        "Access disrupted",
        f"{population_access:,.0f}",
        "CBD-wide",
    ),
    (
        "Priority population",
        f"{population_priority:,.0f}",
        "CBD-wide",
    ),
    (
        "High priority",
        f"{population_high_priority:,.0f}",
        "CBD-wide",
    ),
    (
        "Mean response",
        f"{mean_response:.2f} min",
        "all incidents",
    ),
]


columns = st.columns(
    5
)

for index, (
    label,
    value,
    note,
) in enumerate(kpi_data):

    column = columns[
        index % 5
    ]

    with column:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-label">
                    {label}
                </div>

                <div class="kpi-value">
                    {value}
                </div>

                <div class="kpi-note">
                    {note}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    if index % 5 == 4 and index != len(kpi_data) - 1:

        st.write("")


# ============================================================
# EMERGENCY STATUS
# ============================================================

status_text = "Operational"
response_network = "Routes available"
facility_status = "Facility recommendations stable"

if situation is not None:

    status_text = str(
        situation.get(
            "situation_status",
            "Operational"
        )
    )

    response_network = str(
        situation.get(
            "response_network_status",
            "Routes available"
        )
    )

    facility_status = str(
        situation.get(
            "facility_status",
            "Facility recommendations stable"
        )
    )


if status_text == "Operational":

    status_color = GREEN

elif status_text == "Access Disrupted":

    status_color = AMBER

elif status_text in [
    "Priority",
    "Flood Exposed",
]:

    status_color = ORANGE

else:

    status_color = RED


st.markdown(
    """
    <div class="section-header">

        <div class="section-title">
            Emergency situation
        </div>

        <div class="section-subtitle">
            Network-level emergency response condition
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


status_columns = st.columns(
    3
)

status_items = [
    (
        "Situation",
        status_text,
    ),
    (
        "Response network",
        response_network,
    ),
    (
        "Facility status",
        facility_status,
    ),
]


for column, (
    label,
    value,
) in zip(
    status_columns,
    status_items,
):

    with column:

        color = (
            status_color
            if label == "Situation"
            else NAVY
        )

        st.markdown(
            f"""
            <div class="status-card">

                <div class="status-scenario">
                    {label}
                </div>

                <div class="status-value"
                     style="color:{color};">
                    {html.escape(value)}
                </div>

                <div style="
                    margin-top:8px;
                    color:{MUTED};
                    font-size:12px;
                ">
                    Scenario:
                    {html.escape(selected_scenario)}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# EMERGENCY ALERT
# ============================================================

if emergency_kpi is not None:

    alert_text = str(
        emergency_kpi.get(
            "emergency_alert",
            ""
        )
    )

    if alert_text:

        if selected_scenario == "Normal":

            alert_bg = "#F0FDF4"
            alert_border = "#86EFAC"
            alert_color = "#166534"

        elif status_text == "Access Disrupted":

            alert_bg = "#FFFBEB"
            alert_border = "#FCD34D"
            alert_color = "#92400E"

        else:

            alert_bg = "#FEF2F2"
            alert_border = "#FCA5A5"
            alert_color = "#991B1B"

        st.markdown(
            f"""
            <div class="alert-card"
                 style="
                    background:{alert_bg};
                    border:1px solid {alert_border};
                    color:{alert_color};
                 ">

                <b>Emergency intelligence:</b>
                {html.escape(alert_text)}

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# MAP SECTION
# ============================================================

st.markdown(
    """
    <div class="section-header">

        <div class="section-title">
            Spatial intelligence
        </div>

        <div class="section-subtitle">
            Flood scenario, road disruption, emergency facilities,
            incidents and response routes
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


roads = get_roads(
    selected_scenario
)

facilities = get_facilities(
    selected_scenario
)

incidents = get_incidents()


m = folium.Map(
    location=MAP_CENTER,
    zoom_start=MAP_ZOOM,
    control_scale=True,
    tiles=None,
)


# ------------------------------------------------------------
# Base maps
# ------------------------------------------------------------

folium.TileLayer(
    tiles="OpenStreetMap",
    name="Street map",
    control=True,
).add_to(m)


folium.TileLayer(
    tiles=(
        "https://server.arcgisonline.com/"
        "ArcGIS/rest/services/World_Imagery/"
        "MapServer/tile/{z}/{y}/{x}"
    ),
    attr="Esri",
    name="Satellite",
    control=True,
).add_to(m)


# ------------------------------------------------------------
# Flood layer
# ------------------------------------------------------------

if map_mode == "Flood Scenario":

    flood_overlay(
        m,
        selected_scenario
    )


# ------------------------------------------------------------
# Road layer
# ------------------------------------------------------------

add_road_layer(
    m,
    roads,
    selected_scenario
)


# ------------------------------------------------------------
# Facilities
# ------------------------------------------------------------

add_facility_layer(
    m,
    facilities
)


# ------------------------------------------------------------
# Incidents
# ------------------------------------------------------------

add_incident_layer(
    m,
    incidents,
    selected_incident
)


# ------------------------------------------------------------
# Emergency routes
# ------------------------------------------------------------

if map_mode == "Emergency Response":

    incident_row = get_incident_report(
        selected_scenario,
        selected_incident
    )

    if not incident_row.empty:

        incident = incident_row.iloc[0]

        facility_rank = safe_int(
            incident.get(
                "facility_rank",
                1
            ),
            1
        )

        add_selected_routes(
            m,
            selected_scenario,
            selected_incident,
            facility_rank
        )


# ------------------------------------------------------------
# Legend
# ------------------------------------------------------------

add_road_legend(
    m
)


# ------------------------------------------------------------
# Layer control
# ------------------------------------------------------------

folium.LayerControl(
    collapsed=False
).add_to(m)


# ------------------------------------------------------------
# Map display
# ------------------------------------------------------------

st_folium(
    m,
    width=None,
    height=650,
    returned_objects=[],
)


# ============================================================
# INCIDENT INTELLIGENCE
# ============================================================

st.markdown(
    """
    <div class="section-header">

        <div class="section-title">
            Incident intelligence
        </div>

        <div class="section-subtitle">
            Facility selection and route alternatives for the selected incident
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


incident_report = get_incident_report(
    selected_scenario,
    selected_incident
)


if incident_report.empty:

    st.warning(
        "No emergency intelligence is available for this incident."
    )

else:

    incident = incident_report.iloc[0]

    incident_columns = st.columns(
        4
    )

    incident_summary = [
        (
            "Incident flood impact",
            f"{safe_float(incident.get('incident_flood_impact')):.3f}",
        ),
        (
            "Flood class",
            str(
                incident.get(
                    "incident_flood_class",
                    "Unknown"
                )
            ),
        ),
        (
            "Catchment population",
            f"{safe_float(incident.get('population_in_catchment')):,.0f}",
        ),
        (
            "Priority population",
            f"{safe_float(incident.get('priority_population')):,.0f}",
        ),
    ]

    for column, (
        label,
        value,
    ) in zip(
        incident_columns,
        incident_summary,
    ):

        with column:

            st.markdown(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        {label}
                    </div>

                    <div class="kpi-value">
                        {html.escape(value)}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# FACILITY OPTIONS + ROUTE ALTERNATIVES
# ============================================================

facility_options = get_facility_options(
    selected_scenario,
    selected_incident
)


if facility_options.empty:

    st.info(
        "No facility options are available for this incident."
    )

else:

    recommended_facility_id = str(
        incident.get(
            "recommended_facility_id",
            ""
        )
    )

    recommended_facility_rank = safe_int(
        incident.get(
            "facility_rank",
            1
        ),
        1
    )

    recommended_route_rank = safe_int(
        incident.get(
            "route_rank",
            1
        ),
        1
    )

    for _, facility in facility_options.iterrows():

        facility_rank = safe_int(
            facility.get(
                "facility_rank",
                0
            )
        )

        facility_id = str(
            facility.get(
                "facility_id",
                ""
            )
        )

        facility_name = str(
            facility.get(
                "facility_name",
                "Unknown facility"
            )
        )

        facility_type = str(
            facility.get(
                "facility_type",
                "Facility"
            )
        )

        response_time = safe_float(
            facility.get(
                "travel_time_min"
            )
        )

        normal_time = safe_float(
            facility.get(
                "normal_travel_time_min"
            )
        )

        response_delay = safe_float(
            facility.get(
                "response_delay_min"
            )
        )

        delay_pct = safe_float(
            facility.get(
                "response_delay_pct"
            )
        )

        reachable = bool(
            facility.get(
                "reachable",
                True
            )
        )

        is_recommended = (
            facility_id
            == recommended_facility_id
        )

        if is_recommended:

            border_color = GREEN
            background = "#F0FDF4"
            badge_color = GREEN
            badge_text = "RECOMMENDED"

        else:

            border_color = BORDER
            background = WHITE
            badge_color = MUTED
            badge_text = (
                f"OPTION {facility_rank}"
            )

        reachable_color = (
            GREEN
            if reachable
            else RED
        )

        reachable_text = (
            "Yes"
            if reachable
            else "No"
        )

        # ----------------------------------------------------
        # Facility card
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="facility-card"
                 style="
                    border:1px solid {border_color};
                    background:{background};
                 ">

                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                    margin-bottom:9px;
                ">

                    <span class="badge"
                          style="
                            background:{badge_color};
                          ">
                        {badge_text}
                    </span>

                    <span style="
                        font-size:11px;
                        color:{MUTED};
                        font-weight:700;
                    ">
                        FACILITY RANK {facility_rank}
                    </span>

                </div>

                <div style="
                    font-size:18px;
                    font-weight:800;
                    color:{TEXT};
                ">
                    {html.escape(facility_name)}
                </div>

                <div style="
                    color:{MUTED};
                    font-size:13px;
                    margin-top:3px;
                ">
                    {html.escape(facility_type)}
                    &nbsp; • &nbsp;
                    {html.escape(facility_id)}
                </div>

                <div style="
                    display:flex;
                    flex-wrap:wrap;
                    gap:28px;
                    margin-top:13px;
                    padding-top:11px;
                    border-top:1px solid {BORDER};
                ">

                    <div>
                        <div class="kpi-label">
                            RESPONSE
                        </div>
                        <div style="
                            font-size:18px;
                            font-weight:800;
                            color:{TEXT};
                        ">
                            {response_time:.2f} min
                        </div>
                    </div>

                    <div>
                        <div class="kpi-label">
                            NORMAL
                        </div>
                        <div style="
                            font-size:18px;
                            font-weight:800;
                            color:{TEXT};
                        ">
                            {normal_time:.2f} min
                        </div>
                    </div>

                    <div>
                        <div class="kpi-label">
                            DELAY
                        </div>
                        <div style="
                            font-size:18px;
                            font-weight:800;
                            color:{RED};
                        ">
                            +{response_delay:.2f} min
                        </div>
                    </div>

                    <div>
                        <div class="kpi-label">
                            REACHABLE
                        </div>
                        <div style="
                            font-size:18px;
                            font-weight:800;
                            color:{reachable_color};
                        ">
                            {reachable_text}
                        </div>
                    </div>

                </div>

                <div style="
                    margin-top:9px;
                    font-size:12px;
                    color:{MUTED};
                ">
                    Relative response delay:
                    <b>{delay_pct:,.1f}%</b>
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # Routes for facility
        # ----------------------------------------------------

        routes = get_route_options(
            selected_scenario,
            selected_incident,
            facility_rank
        )

        if routes.empty:

            st.markdown(
                f"""
                <div style="
                    margin-left:20px;
                    color:{MUTED};
                    font-size:12px;
                    padding:6px 0 10px 0;
                ">
                    No route alternatives are available
                    for this facility.
                </div>
                """,
                unsafe_allow_html=True,
            )

            continue

        st.markdown(
            """
            <div style="
                margin-left:20px;
                margin-top:7px;
                margin-bottom:8px;
                font-size:13px;
                font-weight:800;
                color:#334155;
            ">
                Response route alternatives
            </div>
            """,
            unsafe_allow_html=True,
        )

        for _, route in routes.iterrows():

            route_rank = safe_int(
                route.get(
                    "route_rank"
                )
            )

            route_response = safe_float(
                route.get(
                    "response_time_min"
                )
            )

            route_normal = safe_float(
                route.get(
                    "normal_travel_time_min"
                )
            )

            route_delay = safe_float(
                route.get(
                    "response_delay_min"
                )
            )

            route_delay_pct = safe_float(
                route.get(
                    "response_delay_pct"
                )
            )

            route_length = safe_float(
                route.get(
                    "route_length_m"
                )
            )

            affected_segments = safe_int(
                route.get(
                    "affected_segments"
                )
            )

            closed_segments = safe_int(
                route.get(
                    "closed_segments"
                )
            )

            route_recommended = (
                is_recommended
                and route_rank
                == recommended_route_rank
            )

            if route_recommended:

                route_border = GREEN
                route_background = "#F7FEE7"
                route_badge = GREEN
                route_badge_text = "RECOMMENDED"

            else:

                route_border = BORDER
                route_background = WHITE
                route_badge = MUTED
                route_badge_text = (
                    f"ROUTE {route_rank}"
                )

            # ------------------------------------------------
            # Route card
            # ------------------------------------------------

            st.markdown(
                f"""
                <div class="route-card"
                     style="
                        border:1px solid {route_border};
                        border-left:4px solid {route_border};
                        background:{route_background};
                     ">

                    <div style="
                        display:flex;
                        justify-content:space-between;
                        align-items:center;
                        margin-bottom:8px;
                    ">

                        <span class="badge"
                              style="
                                background:{route_badge};
                              ">
                            {route_badge_text}
                        </span>

                        <span style="
                            font-size:11px;
                            color:{MUTED};
                        ">
                            Route {route_rank}
                        </span>

                    </div>

                    <div style="
                        display:flex;
                        flex-wrap:wrap;
                        gap:22px;
                        font-size:13px;
                        color:#334155;
                    ">

                        <div>
                            <span style="color:{MUTED};">
                                Response
                            </span><br>
                            <b>
                                {route_response:.2f} min
                            </b>
                        </div>

                        <div>
                            <span style="color:{MUTED};">
                                Length
                            </span><br>
                            <b>
                                {route_length:,.0f} m
                            </b>
                        </div>

                        <div>
                            <span style="color:{MUTED};">
                                Affected segments
                            </span><br>
                            <b>
                                {affected_segments}
                            </b>
                        </div>

                        <div>
                            <span style="color:{MUTED};">
                                Closed segments
                            </span><br>
                            <b>
                                {closed_segments}
                            </b>
                        </div>

                        <div>
                            <span style="color:{MUTED};">
                                Delay
                            </span><br>
                            <b style="color:{RED};">
                                +{route_delay:.2f} min
                            </b>
                        </div>

                    </div>

                    <div style="
                        margin-top:8px;
                        padding-top:8px;
                        border-top:1px solid {BORDER};
                        color:{MUTED};
                        font-size:11px;
                    ">

                        Normal route time:
                        <b>{route_normal:.2f} min</b>

                        &nbsp; • &nbsp;

                        Relative delay:
                        <b>{route_delay_pct:,.1f}%</b>

                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# SCENARIO COMPARISON
# ============================================================

st.markdown(
    """
    <div class="section-header">

        <div class="section-title">
            Scenario status
        </div>

        <div class="section-subtitle">
            Comparison of emergency conditions across rainfall scenarios
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


scenario_columns = st.columns(
    4
)


for column, scenario_name in zip(
    scenario_columns,
    SCENARIOS,
):

    row = get_situation_report(
        scenario_name
    )

    if row is None:
        continue

    scenario_status = str(
        row.get(
            "situation_status",
            "Unknown"
        )
    )

    scenario_flood = safe_float(
        row.get(
            "mean_incident_flood_impact"
        )
    )

    scenario_high = safe_int(
        row.get(
            "high_very_high_incidents"
        )
    )

    scenario_response = safe_float(
        row.get(
            "mean_response_time_min"
        )
    )

    scenario_color = SCENARIO_COLORS.get(
        scenario_name,
        NAVY
    )

    with column:

        st.markdown(
            f"""
            <div class="status-card"
                 style="
                    border-top:4px solid {scenario_color};
                 ">

                <div class="status-scenario">
                    {html.escape(scenario_name)}
                </div>

                <div class="status-value">
                    {html.escape(scenario_status)}
                </div>

                <div style="
                    margin-top:9px;
                    color:{MUTED};
                    font-size:12px;
                    line-height:1.65;
                ">

                    Rainfall:
                    <b>
                        {SCENARIO_RAINFALL[scenario_name]}
                        mm/hr
                    </b>

                    <br>

                    Flood impact:
                    <b>
                        {scenario_flood:.3f}
                    </b>

                    <br>

                    High / very high incidents:
                    <b>
                        {scenario_high}
                    </b>

                    <br>

                    Mean response:
                    <b>
                        {scenario_response:.2f} min
                    </b>

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown(
    f"""
    <div class="model-note">

        <b>Frozen URIP analytical model</b><br>

        Dashboard outputs are read directly from
        <b>{html.escape(MODEL_FILENAME)}</b>.
        The Streamlit interface does not recalculate flood,
        traffic, population, facility or route analysis.

        <br><br>

        Analytical chain:

        <b>
        rainfall scenario →
        flood impact →
        road disruption →
        passability →
        flood-adjusted travel time →
        traffic →
        rerouting →
        emergency response →
        facilities →
        population
        </b>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#94A3B8;
        font-size:11px;
        margin-top:25px;
        padding-top:12px;
        border-top:1px solid #E2E8F0;
    ">
        URIP — Urban Resilience Intelligence Platform
        • Nairobi CBD prototype
    </div>
    """,
    unsafe_allow_html=True,
)
        
