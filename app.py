

import gzip
import pickle
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="URIP | Urban Resilience Intelligence Platform",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_GZ = BASE_DIR / "URIP_MODEL.pkl.gz"
MODEL_PKL = BASE_DIR / "URIP_MODEL.pkl"

ANALYSIS_CRS = "EPSG:32737"
DISPLAY_CRS = "EPSG:4326"

MAP_CENTER = [-1.2864, 36.8172]
MAP_ZOOM = 13

DEFAULT_SCENARIO = "Normal"

EXPECTED_SECTIONS = [
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


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .urip-title {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 0.15rem;
    }

    .urip-subtitle {
        font-size: 1.05rem;
        opacity: 0.72;
        margin-bottom: 0.5rem;
    }

    .urip-context {
        margin-top: 0.8rem;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        background: rgba(128, 128, 128, 0.08);
        font-size: 0.9rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 750;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }

    .kpi-card {
        padding: 1rem 1rem 0.9rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.18);
        background: rgba(128,128,128,0.045);
        min-height: 125px;
    }

    .kpi-label {
        font-size: 0.82rem;
        opacity: 0.68;
        margin-bottom: 0.35rem;
        font-weight: 600;
    }

    .kpi-value {
        font-size: 1.65rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .kpi-unit {
        font-size: 0.72rem;
        opacity: 0.58;
        margin-top: 0.35rem;
    }

    .status-card {
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.18);
        background: rgba(128,128,128,0.045);
        margin-bottom: 0.7rem;
    }

    .status-title {
        font-size: 0.75rem;
        opacity: 0.65;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .status-value {
        font-size: 1.05rem;
        font-weight: 750;
        margin-top: 0.2rem;
    }

    .model-note {
        font-size: 0.78rem;
        opacity: 0.62;
        padding-top: 1rem;
        border-top: 1px solid rgba(128,128,128,0.18);
        margin-top: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================

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

    return None, None


URIP_MODEL, MODEL_FILE = load_model()


# ============================================================
# MODEL VALIDATION
# ============================================================

if URIP_MODEL is None:

    st.error("Frozen URIP model not found.")

    st.markdown(
        f"""
        The application expects one of the following files:

        - `{MODEL_GZ.name}`
        - `{MODEL_PKL.name}`

        Current application directory:

        `{BASE_DIR}`
        """
    )

    st.stop()


if not isinstance(URIP_MODEL, dict):

    st.error("Invalid URIP model format.")

    st.write(
        f"The loaded object is `{type(URIP_MODEL).__name__}`, "
        "but the dashboard requires a dictionary."
    )

    st.stop()


MISSING_SECTIONS = [
    section
    for section in EXPECTED_SECTIONS
    if section not in URIP_MODEL
]


# ============================================================
# GENERAL HELPERS
# ============================================================

def model_get(section, default=None):
    """Safely retrieve a top-level model section."""
    if not isinstance(URIP_MODEL, dict):
        return default

    return URIP_MODEL.get(section, default)


def as_dataframe(obj):
    """Convert supported objects to a DataFrame where possible."""

    if obj is None:
        return None

    if isinstance(obj, pd.DataFrame):
        return obj.copy()

    if isinstance(obj, pd.Series):
        return obj.to_frame().T

    if isinstance(obj, gpd.GeoDataFrame):
        return obj.copy()

    if isinstance(obj, list):
        try:
            return pd.DataFrame(obj)
        except Exception:
            return None

    if isinstance(obj, dict):

        try:
            return pd.DataFrame(obj)
        except Exception:
            return None

    return None


def row_like(obj):
    """Represent a dictionary or Series as a Series."""

    if isinstance(obj, pd.Series):
        return obj

    if isinstance(obj, dict):
        return pd.Series(obj)

    return None


def get_value(row, aliases, default=None):

    if row is None:
        return default

    for alias in aliases:

        try:
            if alias in row.index:

                value = row[alias]

                if pd.notna(value):
                    return value

        except Exception:
            pass

        try:
            if isinstance(row, dict) and alias in row:

                value = row[alias]

                if value is not None:
                    return value

        except Exception:
            pass

    return default


def find_scenario_row(table, scenario):

    if table is None:
        return None

    if isinstance(table, pd.DataFrame):

        if "scenario" in table.columns:

            match = table[
                table["scenario"].astype(str) == str(scenario)
            ]

            if not match.empty:
                return match.iloc[0]

        return None

    if isinstance(table, dict):

        if scenario in table:
            return row_like(table[scenario])

    return None


def merge_rows(*rows):

    result = pd.Series(dtype=object)

    for row in rows:

        if row is None:
            continue

        current = row_like(row)

        if current is None:
            continue

        for key, value in current.items():

            if key not in result.index:
                result.loc[key] = value
            else:
                try:
                    if pd.isna(result.loc[key]) and pd.notna(value):
                        result.loc[key] = value
                except Exception:
                    pass

    return result


# ============================================================
# SCENARIOS
# ============================================================

SCENARIO_DATA = model_get("scenarios", {})

if isinstance(SCENARIO_DATA, dict):

    SCENARIOS = list(SCENARIO_DATA.keys())

elif isinstance(SCENARIO_DATA, pd.DataFrame):

    if "scenario" in SCENARIO_DATA.columns:

        SCENARIOS = (
            SCENARIO_DATA["scenario"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    else:
        SCENARIOS = []

else:
    SCENARIOS = []


if not SCENARIOS:

    SCENARIOS = [
        "Normal",
        "Moderate Rainfall",
        "Heavy Rainfall",
        "Severe Rainfall",
    ]


if DEFAULT_SCENARIO not in SCENARIOS:
    DEFAULT_SCENARIO = SCENARIOS[0]


RAINFALL_DEFAULTS = {
    "Normal": 10,
    "Moderate Rainfall": 30,
    "Heavy Rainfall": 50,
    "Severe Rainfall": 75,
}


def get_scenario_row(scenario):

    summary = model_get("summary")
    scenarios = model_get("scenarios")
    emergency = model_get("emergency")

    summary_row = find_scenario_row(summary, scenario)
    scenario_row = find_scenario_row(scenarios, scenario)

    emergency_kpis = None

    if isinstance(emergency, dict):
        emergency_kpis = find_scenario_row(
            emergency.get("kpis"),
            scenario
        )

    return merge_rows(
        summary_row,
        scenario_row,
        emergency_kpis,
    )


def get_emergency_kpi(scenario):

    emergency = model_get("emergency")

    if isinstance(emergency, dict):

        return find_scenario_row(
            emergency.get("kpis"),
            scenario
        )

    return None


def get_emergency_situation(scenario):

    emergency = model_get("emergency")

    if isinstance(emergency, dict):

        return find_scenario_row(
            emergency.get("situation_report"),
            scenario
        )

    return None


# ============================================================
# FORMATTING
# ============================================================

def fmt_number(value, decimals=0):

    if value is None:
        return "—"

    try:

        if pd.isna(value):
            return "—"

        return f"{float(value):,.{decimals}f}"

    except Exception:
        return str(value)


def fmt_integer(value):
    return fmt_number(value, 0)


def fmt_decimal(value):
    return fmt_number(value, 3)


def fmt_percent(value):

    if value is None:
        return "—"

    try:

        if pd.isna(value):
            return "—"

        return f"{float(value):.1f}%"

    except Exception:
        return str(value)


# ============================================================
# SCENARIO DATA
# ============================================================

def scenario_rainfall(scenario):

    row = get_scenario_row(scenario)

    value = get_value(
        row,
        [
            "rainfall_mm_hr",
            "rainfall",
            "rainfall_intensity",
        ],
        None,
    )

    if value is None:
        value = RAINFALL_DEFAULTS.get(scenario)

    return value


def dashboard_kpi(scenario, aliases):

    row = get_scenario_row(scenario)

    return get_value(
        row,
        aliases,
        None,
    )


# ============================================================
# INCIDENT HELPERS
# ============================================================

def get_incident_table():

    emergency = model_get("emergency")

    if isinstance(emergency, dict):

        table = emergency.get("incident_report")

        if table is not None:
            return as_dataframe(table)

    table = model_get("incident_report")

    return as_dataframe(table)


def get_incidents_gdf():

    incidents = model_get("incidents")

    if isinstance(incidents, dict):

        candidates = [
            incidents.get("points"),
            incidents.get("gdf"),
            incidents.get("base"),
        ]

        for candidate in candidates:

            if isinstance(candidate, gpd.GeoDataFrame):
                return prepare_gdf(candidate)

    if isinstance(incidents, gpd.GeoDataFrame):
        return prepare_gdf(incidents)

    return None


# ============================================================
# FACILITY HELPERS
# ============================================================

def get_facilities_gdf(scenario):

    facilities = model_get("facilities")

    if facilities is None:
        return None

    if isinstance(facilities, dict):

        candidate = facilities.get(scenario)

        if candidate is None:
            candidate = facilities.get("base")

        if isinstance(candidate, dict):

            candidate = candidate.get("gdf", candidate.get("data"))

        if isinstance(candidate, gpd.GeoDataFrame):
            return prepare_gdf(candidate)

        if isinstance(candidate, pd.DataFrame):

            if "geometry" in candidate.columns:

                try:
                    return prepare_gdf(
                        gpd.GeoDataFrame(
                            candidate,
                            geometry="geometry",
                            crs=ANALYSIS_CRS,
                        )
                    )
                except Exception:
                    pass

    if isinstance(facilities, gpd.GeoDataFrame):
        return prepare_gdf(facilities)

    return None


# ============================================================
# ROAD HELPERS
# ============================================================

def get_roads_gdf(scenario):

    roads = model_get("roads")

    if roads is None:
        return None

    candidate = roads

    if isinstance(roads, dict):

        candidate = roads.get(scenario)

        if candidate is None:
            candidate = roads.get("base")

        if isinstance(candidate, dict):

            candidate = candidate.get(
                "gdf",
                candidate.get("data")
            )

    if isinstance(candidate, gpd.GeoDataFrame):
        return prepare_gdf(candidate)

    if isinstance(candidate, pd.DataFrame):

        if "geometry" in candidate.columns:

            try:

                return prepare_gdf(
                    gpd.GeoDataFrame(
                        candidate,
                        geometry="geometry",
                        crs=ANALYSIS_CRS,
                    )
                )

            except Exception:
                return None

    return None


# ============================================================
# GEO DATA PREPARATION
# ============================================================

def prepare_gdf(gdf):

    if gdf is None:
        return None

    result = gdf.copy()

    if result.crs is None:

        result = result.set_crs(
            ANALYSIS_CRS,
            allow_override=True,
        )

    if str(result.crs) != DISPLAY_CRS:

        try:
            result = result.to_crs(DISPLAY_CRS)
        except Exception:
            pass

    return result


# ============================================================
# FLOOD RASTER
# ============================================================

FLOOD_BOUNDS = [
    [-1.3149999999999973, 36.79],
    [-1.2600000000000051, 36.85],
]


def get_flood_raster(scenario):

    flood = model_get("flood")

    if flood is None:
        return None

    if isinstance(flood, dict):

        raster = flood.get(scenario)

        if isinstance(raster, dict):

            raster = raster.get(
                "raster",
                raster.get("flood_raster")
            )

        return raster

    return None


# ============================================================
# ROUTE HELPERS
# ============================================================

def get_route_table():

    routes = model_get("routes")

    if isinstance(routes, dict):

        candidates = [
            routes.get("alternatives"),
            routes.get("routes"),
            routes.get("data"),
        ]

        for candidate in candidates:

            df = as_dataframe(candidate)

            if df is not None:
                return df

    return as_dataframe(routes)


# ============================================================
# FACILITY OPTION HELPERS
# ============================================================

def get_facility_options():

    options = model_get("facility_options")

    if isinstance(options, dict):

        candidates = [
            options.get("top3"),
            options.get("options"),
            options.get("data"),
        ]

        for candidate in candidates:

            df = as_dataframe(candidate)

            if df is not None:
                return df

    return as_dataframe(options)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## URIP Controls")

    scenario = st.selectbox(
        "Rainfall Scenario",
        SCENARIOS,
        index=SCENARIOS.index(DEFAULT_SCENARIO),
    )

    map_mode = st.radio(
        "Map Mode",
        [
            "Flood Scenario",
            "Emergency Response",
        ],
    )

    incidents_gdf = get_incidents_gdf()

    incident_ids = []

    if incidents_gdf is not None:

        if "incident_id" in incidents_gdf.columns:

            incident_ids = (
                incidents_gdf["incident_id"]
                .dropna()
                .astype(str)
                .tolist()
            )

    incident_table = get_incident_table()

    if not incident_ids and incident_table is not None:

        if "incident_id" in incident_table.columns:

            incident_ids = (
                incident_table["incident_id"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

    if not incident_ids:

        incident_ids = [
            f"INC_{i:03d}"
            for i in range(1, 21)
        ]

    incident_id = st.selectbox(
        "Emergency Incident",
        incident_ids,
    )

    st.markdown("---")

    st.caption(
        f"Model file: {MODEL_FILE}"
    )

    if MISSING_SECTIONS:

        with st.expander("Model diagnostics"):

            st.warning(
                "The loaded model is missing dashboard output sections."
            )

            st.write("Missing sections:")

            for section in MISSING_SECTIONS:
                st.write(f"- `{section}`")

            st.write("Available sections:")

            for section in URIP_MODEL.keys():
                st.write(f"- `{section}`")

    else:

        with st.expander("Model diagnostics"):

            st.success(
                "Complete frozen URIP dashboard package detected."
            )

            st.write(
                f"Sections loaded: {len(URIP_MODEL)}"
            )


# ============================================================
# SCENARIO INFORMATION
# ============================================================

rainfall = scenario_rainfall(scenario)

scenario_row = get_scenario_row(scenario)

situation_row = get_emergency_situation(scenario)

emergency_kpi = get_emergency_kpi(scenario)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="urip-title">URIP</div>

    <div class="urip-subtitle">
        Urban Resilience Intelligence Platform
        &nbsp;|&nbsp;
        Nairobi CBD Emergency Response Simulation
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="urip-context">
        Scenario:
        <b>{scenario}</b>
        &nbsp; • &nbsp;
        Rainfall:
        <b>{fmt_number(rainfall, 0)} mm/hr</b>
        &nbsp; • &nbsp;
        Map:
        <b>{map_mode}</b>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FLOOD / RESILIENCE KPIs
# ============================================================

st.markdown(
    '<div class="section-title">Urban Resilience Indicators</div>',
    unsafe_allow_html=True,
)

flood_impact = dashboard_kpi(
    scenario,
    [
        "mean_flood_impact",
        "flood_impact",
        "mean_flood_score",
    ],
)

roads_affected = dashboard_kpi(
    scenario,
    [
        "roads_affected",
        "affected_roads",
    ],
)

roads_closed = dashboard_kpi(
    scenario,
    [
        "roads_closed",
        "closed_roads",
    ],
)

mean_vc = dashboard_kpi(
    scenario,
    [
        "mean_vc",
        "mean_final_vc",
        "final_vc",
        "mean_volume_capacity",
    ],
)

facilities_affected = dashboard_kpi(
    scenario,
    [
        "facilities_affected",
        "operationally_affected_facilities",
    ],
)


kpi_cols = st.columns(5)


kpi_data = [
    (
        "Flood Impact",
        fmt_decimal(flood_impact),
        "mean score",
    ),
    (
        "Roads Affected",
        fmt_integer(roads_affected),
        "roads",
    ),
    (
        "Roads Closed",
        fmt_integer(roads_closed),
        "roads",
    ),
    (
        "Mean V/C",
        fmt_decimal(mean_vc),
        "volume / capacity",
    ),
    (
        "Facilities Affected",
        fmt_integer(facilities_affected),
        "facilities",
    ),
]


for col, (label, value, unit) in zip(kpi_cols, kpi_data):

    with col:

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-unit">{unit}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# EMERGENCY KPIs
# ============================================================

st.markdown(
    '<div class="section-title">Resilience & Emergency Indicators</div>',
    unsafe_allow_html=True,
)


population_exposed = dashboard_kpi(
    scenario,
    [
        "population_exposed",
        "flood_exposed_population",
    ],
)

access_disrupted = dashboard_kpi(
    scenario,
    [
        "access_disrupted",
        "access_disrupted_population",
    ],
)

priority_population = dashboard_kpi(
    scenario,
    [
        "priority_population",
    ],
)

high_priority = dashboard_kpi(
    scenario,
    [
        "high_priority_population",
    ],
)

mean_response = dashboard_kpi(
    scenario,
    [
        "mean_response_time_min",
        "mean_response_time",
        "response_time_min",
    ],
)


emergency_cols = st.columns(5)


emergency_data = [
    (
        "Population Exposed",
        fmt_integer(population_exposed),
        "people",
    ),
    (
        "Access Disrupted",
        fmt_integer(access_disrupted),
        "people",
    ),
    (
        "Priority Population",
        fmt_integer(priority_population),
        "people",
    ),
    (
        "High Priority",
        fmt_integer(high_priority),
        "people",
    ),
    (
        "Mean Response Time",
        fmt_decimal(mean_response),
        "minutes",
    ),
]


for col, (label, value, unit) in zip(
    emergency_cols,
    emergency_data,
):

    with col:

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-unit">{unit}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# EMERGENCY SITUATION
# ============================================================

st.markdown(
    '<div class="section-title">Emergency Situation</div>',
    unsafe_allow_html=True,
)


situation_status = get_value(
    situation_row,
    [
        "situation_status",
        "status",
    ],
    None,
)

response_network = get_value(
    situation_row,
    [
        "network",
        "response_network",
        "network_status",
    ],
    None,
)

facility_status = get_value(
    situation_row,
    [
        "facility_status",
        "facility_recommendation_status",
    ],
    None,
)

route_availability = get_value(
    situation_row,
    [
        "route_availability_pct",
        "route_availability",
    ],
    None,
)


if situation_status is None:

    situation_status = get_value(
        emergency_kpi,
        ["situation_status", "status"],
        "Unavailable",
    )


if response_network is None:

    response_network = get_value(
        emergency_kpi,
        ["route_availability_note"],
        "Unavailable",
    )


if facility_status is None:

    facility_status = "Unavailable"


if route_availability is None:

    route_availability_display = "—"

else:

    try:

        route_availability_display = (
            f"{float(route_availability):.1f}%"
        )

    except Exception:

        route_availability_display = str(
            route_availability
        )


status_cols = st.columns(4)


status_data = [
    (
        "Situation",
        situation_status,
    ),
    (
        "Response Network",
        response_network,
    ),
    (
        "Facility Status",
        facility_status,
    ),
    (
        "Route Availability",
        route_availability_display,
    ),
]


for col, (label, value) in zip(
    status_cols,
    status_data,
):

    with col:

        st.markdown(
            f"""
            <div class="status-card">
                <div class="status-title">{label}</div>
                <div class="status-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# MODEL COMPLETENESS NOTICE
# ============================================================

if MISSING_SECTIONS:

    st.warning(
        "The current uploaded model is not the complete frozen "
        "URIP dashboard package. The analytical values available "
        "in the loaded model are displayed above, but map layers "
        "and emergency outputs that depend on missing sections "
        "cannot be displayed yet."
    )


# ============================================================
# MAP
# ============================================================

st.markdown(
    '<div class="section-title">Urban Intelligence Map</div>',
    unsafe_allow_html=True,
)


roads_gdf = get_roads_gdf(scenario)
facilities_gdf = get_facilities_gdf(scenario)
incidents_gdf = get_incidents_gdf()
flood_raster = get_flood_raster(scenario)


m = folium.Map(
    location=MAP_CENTER,
    zoom_start=MAP_ZOOM,
    control_scale=True,
    tiles=None,
)


# ------------------------------------------------------------
# BASEMAPS
# ------------------------------------------------------------

folium.TileLayer(
    tiles=(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    ),
    attr="Esri",
    name="ESRI Satellite",
).add_to(m)


folium.TileLayer(
    tiles="OpenStreetMap",
    name="OpenStreetMap",
).add_to(m)


# ------------------------------------------------------------
# FLOOD RASTER
# ------------------------------------------------------------

if flood_raster is not None:

    try:

        raster_array = np.asarray(
            flood_raster,
            dtype=float,
        )

        raster_array = np.nan_to_num(
            raster_array,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        if raster_array.ndim == 2:

            normalized = (
                raster_array - np.nanmin(raster_array)
            )

            denominator = (
                np.nanmax(raster_array)
                - np.nanmin(raster_array)
            )

            if denominator > 0:
                normalized = normalized / denominator

            normalized = np.clip(
                normalized,
                0,
                1,
            )

            folium.raster_layers.ImageOverlay(
                image=normalized,
                bounds=FLOOD_BOUNDS,
                opacity=0.60 if map_mode == "Flood Scenario" else 0.28,
                interactive=True,
                cross_origin=False,
                zindex=2,
                colormap=lambda value: (
                    0.1,
                    0.45,
                    0.9,
                    value,
                ),
                name=f"Flood Impact — {scenario}",
            ).add_to(m)

    except Exception:
        pass


# ------------------------------------------------------------
# FLOOD ANALYSIS BOUNDARY
# ------------------------------------------------------------

folium.Rectangle(
    bounds=FLOOD_BOUNDS,
    color="#ffffff",
    weight=3,
    opacity=0.95,
    fill=False,
    dash_array="8,6",
    tooltip="Flood Analysis Boundary",
    name="Flood Analysis Boundary",
).add_to(m)


# ------------------------------------------------------------
# ROADS
# ------------------------------------------------------------

if roads_gdf is not None and not roads_gdf.empty:

    status_column = None

    for candidate in [
        "passability_class",
        "passability",
        "road_status",
        "disruption_class",
        "road_disruption",
    ]:

        if candidate in roads_gdf.columns:
            status_column = candidate
            break


    ROAD_COLORS = {
        "Passable": "#2ca25f",
        "Minor Disruption": "#fdae6b",
        "Severe Disruption": "#e34a33",
        "Closed": "#7f0000",
        "No Data": "#969696",
    }


    def road_style(feature):

        properties = feature.get(
            "properties",
            {},
        )

        status = "No Data"

        if status_column is not None:

            value = properties.get(
                status_column
            )

            if value is not None:
                status = str(value)

        color = ROAD_COLORS.get(
            status,
            "#969696",
        )

        return {
            "color": color,
            "weight": 2.2,
            "opacity": 0.82,
        }


    tooltip_fields = []

    if status_column is not None:
        tooltip_fields.append(status_column)

    try:

        folium.GeoJson(
            roads_gdf.to_json(),
            name=f"Road Disruption — {scenario}",
            style_function=road_style,
            tooltip=(
                folium.GeoJsonTooltip(
                    fields=tooltip_fields
                )
                if tooltip_fields
                else None
            ),
        ).add_to(m)

    except Exception:
        pass


# ------------------------------------------------------------
# FACILITIES
# ------------------------------------------------------------

if facilities_gdf is not None and not facilities_gdf.empty:

    facility_group = folium.FeatureGroup(
        name="Emergency Facilities",
        show=True,
    )

    for _, row in facilities_gdf.iterrows():

        geometry = row.geometry

        if geometry is None:
            continue

        try:

            centroid = geometry.centroid

            facility_id = row.get(
                "facility_id",
                "Facility",
            )

            facility_name = row.get(
                "name",
                row.get(
                    "facility_name",
                    "",
                ),
            )

            facility_type = row.get(
                "facility_type",
                "",
            )

            operational_status = row.get(
                "operationally_affected",
                None,
            )

            popup_html = (
                f"<b>{facility_id}</b><br>"
                f"{facility_name}<br>"
                f"Type: {facility_type}"
            )

            if operational_status is not None:

                popup_html += (
                    f"<br>Operationally affected: "
                    f"{operational_status}"
                )

            folium.CircleMarker(
                location=[
                    centroid.y,
                    centroid.x,
                ],
                radius=4,
                weight=1,
                fill=True,
                fill_opacity=0.9,
                popup=folium.Popup(
                    popup_html,
                    max_width=300,
                ),
            ).add_to(facility_group)

        except Exception:
            continue

    facility_group.add_to(m)


# ------------------------------------------------------------
# INCIDENTS
# ------------------------------------------------------------

if incidents_gdf is not None and not incidents_gdf.empty:

    incident_group = folium.FeatureGroup(
        name="Emergency Incidents",
        show=True,
    )

    for _, row in incidents_gdf.iterrows():

        geometry = row.geometry

        if geometry is None:
            continue

        try:

            incident_id_value = row.get(
                "incident_id",
                "Incident",
            )

            popup_text = str(
                incident_id_value
            )

            folium.Marker(
                location=[
                    geometry.y,
                    geometry.x,
                ],
                popup=popup_text,
                tooltip=popup_text,
            ).add_to(incident_group)

        except Exception:
            continue

    incident_group.add_to(m)


# ------------------------------------------------------------
# SELECTED INCIDENT
# ------------------------------------------------------------

if incidents_gdf is not None:

    if "incident_id" in incidents_gdf.columns:

        selected = incidents_gdf[
            incidents_gdf["incident_id"].astype(str)
            == str(incident_id)
        ]

        if not selected.empty:

            point = selected.iloc[0].geometry

            try:

                folium.CircleMarker(
                    location=[
                        point.y,
                        point.x,
                    ],
                    radius=9,
                    color="#ff0000",
                    weight=3,
                    fill=True,
                    fill_opacity=0.25,
                    tooltip=f"Selected: {incident_id}",
                    name="Selected Incident",
                ).add_to(m)

            except Exception:
                pass


# ------------------------------------------------------------
# MAP CONTROLS
# ------------------------------------------------------------

folium.LayerControl(
    collapsed=False,
).add_to(m)


st_folium(
    m,
    use_container_width=True,
    height=650,
)


# ============================================================
# FOOTER
# ============================================================

metadata = model_get(
    "metadata",
    {},
)

model_version = "Frozen URIP analytical model"

if isinstance(metadata, dict):

    model_version = metadata.get(
        "model_version",
        metadata.get(
            "version",
            model_version,
        ),
    )


st.markdown(
    f"""
    <div class="model-note">
        <b>URIP model:</b> {model_version}<br>
        <b>Study area:</b> Nairobi CBD<br>
        <b>Analysis CRS:</b> {ANALYSIS_CRS}<br>
        <b>Display CRS:</b> {DISPLAY_CRS}<br>
        Analytical outputs are loaded from the frozen URIP model;
        the dashboard does not recalculate the underlying analysis.
    </div>
    """,
    unsafe_allow_html=True,
)
        
