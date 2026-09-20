

# ============================================================
# URIP — URBAN RESILIENCE INTELLIGENCE PLATFORM
# Nairobi CBD Emergency Response Simulation
#
# UI / VISUALIZATION LAYER ONLY
# No analytical recalculation is performed here.
# ============================================================

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium

from shapely.geometry import mapping
from streamlit_folium import st_folium
from folium import Element


warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="URIP | Urban Resilience Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_PATH = "URIP_MODEL.pkl"

CRS_METRIC = "EPSG:32737"
CRS_GEO = "EPSG:4326"

MAP_CENTER = [-1.2864, 36.8172]
MAP_ZOOM = 13

ROAD_COLORS = {
    "Passable": "#2ca25f",
    "Minor Disruption": "#fdae6b",
    "Severe Disruption": "#e34a33",
    "Closed": "#7f0000",
    "No Data": "#969696",
}


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f9fb;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .urip-header {
        background:
            linear-gradient(
                135deg,
                #0b3d2e 0%,
                #145a43 55%,
                #1f7a59 100%
            );
        padding: 1.35rem 1.6rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.1rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.12);
    }

    .urip-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }

    .urip-subtitle {
        font-size: 0.95rem;
        opacity: 0.88;
    }

    .kpi-card {
        background: white;
        border: 1px solid #e5e9ee;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        min-height: 105px;
        box-shadow: 0 2px 7px rgba(0,0,0,0.05);
    }

    .kpi-label {
        color: #6b7280;
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .kpi-value {
        color: #17202a;
        font-size: 1.55rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }

    .kpi-unit {
        color: #6b7280;
        font-size: 0.72rem;
    }

    .status-card {
        background: white;
        border-radius: 12px;
        border: 1px solid #e5e9ee;
        padding: 1rem 1.1rem;
        min-height: 100px;
    }

    .status-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #6b7280;
        font-weight: 600;
    }

    .status-value {
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 0.35rem;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #17202a;
        margin-top: 1.1rem;
        margin-bottom: 0.6rem;
    }

    .info-box {
        background: #eef6f2;
        border-left: 4px solid #1f7a59;
        padding: 0.8rem 1rem;
        border-radius: 7px;
        color: #24352d;
        font-size: 0.86rem;
    }

    .alert-box {
        background: #fff4e5;
        border-left: 4px solid #f39c12;
        padding: 0.8rem 1rem;
        border-radius: 7px;
        color: #593b08;
        font-size: 0.86rem;
    }

    .danger-box {
        background: #fff0f0;
        border-left: 4px solid #c0392b;
        padding: 0.8rem 1rem;
        border-radius: 7px;
        color: #5b1914;
        font-size: 0.86rem;
    }

    .footer {
        color: #7f8c8d;
        font-size: 0.75rem;
        text-align: center;
        padding-top: 1.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD FROZEN MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Frozen model not found: {MODEL_PATH}"
        )
        st.stop()

    with open(
        MODEL_PATH,
        "rb"
    ) as f:

        model = pickle.load(f)

    return model


URIP_MODEL = load_model()


# ============================================================
# MODEL METADATA
# ============================================================

METADATA = URIP_MODEL.get(
    "metadata",
    {}
)

SCENARIO_TABLE = URIP_MODEL[
    "scenarios"
].copy()

SCENARIOS = (
    SCENARIO_TABLE[
        "scenario"
    ]
    .tolist()
)


# ============================================================
# HELPERS
# ============================================================

def safe_float(
    value,
    default=0.0
):

    try:

        if pd.isna(value):
            return default

        return float(value)

    except Exception:

        return default


def format_number(
    value,
    decimals=0
):

    if value is None:
        return "—"

    if isinstance(
        value,
        (float, np.floating)
    ):

        return f"{value:,.{decimals}f}"

    try:

        return f"{float(value):,.{decimals}f}"

    except:

        return str(value)


def get_summary_row(
    scenario
):

    summary = URIP_MODEL[
        "summary"
    ]

    row = summary[
        summary["scenario"] == scenario
    ]

    if row.empty:
        return None

    return row.iloc[0]


def get_emergency_kpi(
    scenario
):

    table = URIP_MODEL[
        "emergency"
    ][
        "kpis"
    ]

    row = table[
        table["scenario"] == scenario
    ]

    if row.empty:
        return None

    return row.iloc[0]


def get_emergency_situation(
    scenario
):

    table = URIP_MODEL[
        "emergency"
    ][
        "situation_report"
    ]

    row = table[
        table["scenario"] == scenario
    ]

    if row.empty:
        return None

    return row.iloc[0]


def get_incident_report(
    scenario,
    incident_id
):

    table = URIP_MODEL[
        "emergency"
    ][
        "incident_report"
    ]

    row = table[
        (table["scenario"] == scenario)
        &
        (table["incident_id"] == incident_id)
    ]

    if row.empty:
        return None

    return row.iloc[0]


# ============================================================
# INCIDENTS
# ============================================================

INCIDENTS = (
    URIP_MODEL[
        "incidents"
    ][
        "points"
    ][
        "incident_id"
    ]
    .tolist()
)


# ============================================================
# FACILITY OPTIONS
# ============================================================

def get_facility_options(
    scenario,
    incident_id
):

    table = URIP_MODEL[
        "facility_options"
    ]

    result = table[
        (table["scenario"] == scenario)
        &
        (table["incident_id"] == incident_id)
    ].copy()

    return (
        result
        .sort_values("facility_rank")
        .head(3)
    )


# ============================================================
# ROUTE OPTIONS
# ============================================================

def get_routes_for_facility(
    scenario,
    incident_id,
    facility_rank
):

    table = URIP_MODEL[
        "routes"
    ][
        "alternatives"
    ]

    result = table[
        (table["scenario"] == scenario)
        &
        (table["incident_id"] == incident_id)
        &
        (
            table["facility_rank"]
            == facility_rank
        )
    ].copy()

    return (
        result
        .sort_values("route_rank")
        .head(3)
    )


# ============================================================
# KPI CARD
# ============================================================

def kpi_card(
    label,
    value,
    unit=""
):

    return f"""
    <div class="kpi-card">

        <div class="kpi-label">
            {label}
        </div>

        <div class="kpi-value">
            {value}
        </div>

        <div class="kpi-unit">
            {unit}
        </div>

    </div>
    """


# ============================================================
# MAP — FLOOD RASTER
# ============================================================

def add_flood_layer(
    m,
    scenario
):

    raster = np.asarray(
        URIP_MODEL[
            "flood"
        ][scenario],
        dtype=float
    )

    if raster.ndim != 2:
        return

    valid = np.isfinite(raster)

    if not valid.any():
        return

    values = raster[valid]

    vmin = np.nanmin(values)
    vmax = np.nanmax(values)

    if vmax > vmin:

        norm = (
            raster - vmin
        ) / (
            vmax - vmin
        )

    else:

        norm = np.zeros_like(
            raster
        )

    norm = np.clip(
        norm,
        0,
        1
    )

    rgba = np.zeros(
        (
            raster.shape[0],
            raster.shape[1],
            4
        ),
        dtype=np.uint8
    )

    # Blue flood ramp
    rgba[:, :, 0] = (
        210 - 180 * norm
    ).astype(np.uint8)

    rgba[:, :, 1] = (
        235 - 170 * norm
    ).astype(np.uint8)

    rgba[:, :, 2] = 255

    rgba[:, :, 3] = np.where(
        valid,
        50 + 180 * norm,
        0
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Recover authoritative flood bounds.
    # --------------------------------------------------------

    bounds = [
        [-1.3149999999999973, 36.79],
        [-1.2600000000000051, 36.85]
    ]

    folium.raster_layers.ImageOverlay(
        image=rgba,
        bounds=bounds,
        opacity=0.70,
        interactive=True,
        cross_origin=False,
        zindex=2,
        name="Flood Impact"
    ).add_to(m)


# ============================================================
# MAP — ANALYSIS BOUNDARY
# ============================================================

def add_analysis_boundary(
    m,
    scenario
):

    bounds = [
        [-1.3149999999999973, 36.79],
        [-1.2600000000000051, 36.85]
    ]

    south = bounds[0][0]
    west = bounds[0][1]

    north = bounds[1][0]
    east = bounds[1][1]

    coordinates = [
        [south, west],
        [south, east],
        [north, east],
        [north, west],
        [south, west]
    ]

    group = folium.FeatureGroup(
        name="Flood Analysis Boundary",
        show=True
    )

    folium.PolyLine(
        coordinates,
        color="#1f2937",
        weight=5,
        opacity=0.4
    ).add_to(group)

    folium.PolyLine(
        coordinates,
        color="#ffffff",
        weight=3,
        opacity=1,
        dash_array="8,6",
        tooltip=(
            f"Flood Analysis Boundary — "
            f"{scenario}"
        )
    ).add_to(group)

    group.add_to(m)


# ============================================================
# MAP — ROADS
# ============================================================

def add_roads(
    m,
    scenario
):

    roads = (
        URIP_MODEL[
            "roads"
        ][scenario]
        .copy()
    )

    if roads.crs != CRS_GEO:
        roads = roads.to_crs(
            CRS_GEO
        )

    roads = roads[
        roads.geometry.notna()
        &
        ~roads.geometry.is_empty
    ]

    candidates = [
        "passability_class",
        "passability",
        "road_status",
        "disruption_class",
        "road_disruption",
    ]

    passability_column = None

    for col in candidates:

        if col in roads.columns:
            passability_column = col
            break

    # --------------------------------------------------------
    # Only serialize useful attributes.
    # --------------------------------------------------------

    columns = [
        "geometry",
        "road_id",
        "road_type",
        "highway",
        "flood_impact",
        "passability_class",
        "passability",
        "road_status",
        "disruption_class",
        "travel_time_min",
        "final_vc",
        "congestion_class",
    ]

    columns = [
        c for c in columns
        if c in roads.columns
    ]

    roads_map = roads[
        columns
    ].copy()

    fields = [
        c for c in [
            "road_id",
            "flood_impact",
            passability_column,
            "final_vc",
            "congestion_class"
        ]
        if (
            c is not None
            and c in roads_map.columns
        )
    ]

    aliases = [
        c.replace(
            "_",
            " "
        ).title()
        for c in fields
    ]

    def style_function(
        feature
    ):

        if passability_column:

            status = feature[
                "properties"
            ].get(
                passability_column,
                "No Data"
            )

        else:

            status = "No Data"

        return {
            "color": ROAD_COLORS.get(
                str(status),
                "#969696"
            ),
            "weight": (
                1.2
                if str(status) == "Passable"
                else 2.8
            ),
            "opacity": 0.85
        }

    group = folium.FeatureGroup(
        name="Road Network / Passability",
        show=True
    )

    folium.GeoJson(
        roads_map.to_json(),
        style_function=style_function,
        highlight_function=lambda f: {
            "color": "#ffffff",
            "weight": 4,
            "opacity": 1
        },
        tooltip=(
            folium.GeoJsonTooltip(
                fields=fields,
                aliases=aliases,
                localize=True
            )
            if fields
            else None
        )
    ).add_to(group)

    group.add_to(m)


# ============================================================
# MAP — FACILITIES
# ============================================================

def add_facilities(
    m,
    scenario,
    selected_facility_id=None
):

    facilities = (
        URIP_MODEL[
            "facilities"
        ][scenario]
        .copy()
    )

    if facilities.crs != CRS_GEO:
        facilities = facilities.to_crs(
            CRS_GEO
        )

    group = folium.FeatureGroup(
        name="Emergency Facilities",
        show=True
    )

    for _, row in facilities.iterrows():

        geometry = row.geometry

        if geometry is None:
            continue

        facility_id = str(
            row.get(
                "facility_id",
                ""
            )
        )

        name = str(
            row.get(
                "name",
                facility_id
            )
        )

        facility_type = str(
            row.get(
                "facility_type",
                "Facility"
            )
        )

        affected = bool(
            row.get(
                "operationally_affected",
                False
            )
        )

        selected = (
            facility_id
            == str(
                selected_facility_id
            )
        )

        popup = f"""
        <div style="font-family:Arial;width:270px">

        <h4>{name}</h4>

        <b>Type:</b> {facility_type}<br>
        <b>ID:</b> {facility_id}<br>

        <b>Flood class:</b>
        {row.get("flood_class", "—")}<br>

        <b>Access:</b>
        {row.get("access_status", "—")}<br>

        <b>Response time:</b>
        {safe_float(
            row.get("response_time", np.nan),
            np.nan
        ):.2f} min<br>

        <b>Response delay:</b>
        {safe_float(
            row.get("response_delay", np.nan),
            np.nan
        ):.2f} min<br>

        <b>Operationally affected:</b>
        {"Yes" if affected else "No"}

        </div>
        """

        folium.CircleMarker(
            location=[
                geometry.y,
                geometry.x
            ],
            radius=(
                9
                if selected
                else 5
            ),
            color=(
                "#00ff88"
                if selected
                else (
                    "#c0392b"
                    if affected
                    else "#2471a3"
                )
            ),
            fill=True,
            fill_color=(
                "#00ff88"
                if selected
                else (
                    "#c0392b"
                    if affected
                    else "#2471a3"
                )
            ),
            fill_opacity=0.95,
            weight=2,
            tooltip=(
                f"{name} — "
                f"{facility_type}"
            ),
            popup=folium.Popup(
                popup,
                max_width=320
            )
        ).add_to(group)

    group.add_to(m)


# ============================================================
# MAP — INCIDENTS
# ============================================================

def add_incidents(
    m,
    scenario,
    incident_id
):

    incidents = (
        URIP_MODEL[
            "incidents"
        ][
            "points"
        ]
        .copy()
    )

    if incidents.crs != CRS_GEO:
        incidents = incidents.to_crs(
            CRS_GEO
        )

    report = (
        URIP_MODEL[
            "emergency"
        ][
            "incident_report"
        ]
    )

    group = folium.FeatureGroup(
        name="Emergency Incidents",
        show=True
    )

    for _, row in incidents.iterrows():

        current_id = str(
            row["incident_id"]
        )

        selected = (
            current_id
            == str(incident_id)
        )

        details = report[
            (report["scenario"] == scenario)
            &
            (
                report["incident_id"]
                == current_id
            )
        ]

        if not details.empty:

            d = details.iloc[0]

            popup = f"""
            <div style="font-family:Arial;width:270px">

            <h4>{current_id}</h4>

            <b>Scenario:</b> {scenario}<br>

            <b>Incident flood class:</b>
            {d.get(
                "incident_flood_class",
                "—"
            )}<br>

            <b>Flood impact:</b>
            {safe_float(
                d.get(
                    "incident_flood_impact",
                    np.nan
                ),
                np.nan
            ):.3f}<br>

            <b>Catchment population:</b>
            {safe_float(
                d.get(
                    "population_in_catchment",
                    0
                )
            ):,.0f}<br>

            <b>Flood exposed:</b>
            {safe_float(
                d.get(
                    "flood_exposed_population",
                    0
                )
            ):,.0f}<br>

            <b>Access disrupted:</b>
            {safe_float(
                d.get(
                    "access_disrupted_population",
                    0
                )
            ):,.0f}<br>

            <b>Priority population:</b>
            {safe_float(
                d.get(
                    "priority_population",
                    0
                )
            ):,.0f}

            </div>
            """

        else:

            popup = (
                f"<b>{current_id}</b>"
            )

        folium.CircleMarker(
            location=[
                row.geometry.y,
                row.geometry.x
            ],
            radius=10 if selected else 6,
            color="#ffffff",
            weight=3 if selected else 2,
            fill=True,
            fill_color=(
                "#ff0000"
                if selected
                else "#ffcc00"
            ),
            fill_opacity=0.95,
            tooltip=(
                current_id
                +
                (
                    " — SELECTED"
                    if selected
                    else ""
                )
            ),
            popup=folium.Popup(
                popup,
                max_width=320
            )
        ).add_to(group)

    group.add_to(m)


# ============================================================
# MAP — RESPONSE ROUTE
# ============================================================

def add_response_route(
    m,
    scenario,
    incident_id
):

    routes = (
        URIP_MODEL[
            "routes"
        ][
            "alternatives"
        ]
    )

    result = routes[
        (routes["scenario"] == scenario)
        &
        (
            routes["incident_id"]
            == incident_id
        )
        &
        (
            routes["facility_rank"]
            == 1
        )
        &
        (
            routes["route_rank"]
            == 1
        )
    ]

    if result.empty:
        return None

    route = result.iloc[0]

    facility_id = route[
        "facility_id"
    ]

    geometry = route.get(
        "geometry",
        None
    )

    group = folium.FeatureGroup(
        name="Emergency Response Route",
        show=True
    )

    if geometry is not None:

        route_gdf = gpd.GeoDataFrame(
            [route],
            geometry="geometry",
            crs=CRS_METRIC
        ).to_crs(
            CRS_GEO
        )

        route_geometry = (
            route_gdf.geometry.iloc[0]
        )

        folium.GeoJson(
            mapping(
                route_geometry
            ),
            style_function=lambda f: {
                "color": "#00ffff",
                "weight": 6,
                "opacity": 0.95
            },
            tooltip=(
                "Recommended response route | "
                f"{safe_float(
                    route['response_time_min']
                ):.2f} min"
            )
        ).add_to(group)

    group.add_to(m)

    return facility_id


# ============================================================
# BUILD MAP
# ============================================================

def build_map(
    scenario,
    map_mode,
    incident_id
):

    m = folium.Map(
        location=MAP_CENTER,
        zoom_start=MAP_ZOOM,
        control_scale=True,
        tiles=None
    )

    # --------------------------------------------------------
    # Basemaps
    # --------------------------------------------------------

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/"
            "{z}/{y}/{x}"
        ),
        attr=(
            "Esri, Maxar, Earthstar Geographics, "
            "and the GIS User Community"
        ),
        name="ESRI Satellite",
        overlay=False
    ).add_to(m)

    folium.TileLayer(
        "OpenStreetMap",
        name="Street Map",
        overlay=False
    ).add_to(m)

    # --------------------------------------------------------
    # Flood
    # --------------------------------------------------------

    add_flood_layer(
        m,
        scenario
    )

    add_analysis_boundary(
        m,
        scenario
    )

    # --------------------------------------------------------
    # Roads
    # --------------------------------------------------------

    add_roads(
        m,
        scenario
    )

    # --------------------------------------------------------
    # Incident
    # --------------------------------------------------------

    incident_report = get_incident_report(
        scenario,
        incident_id
    )

    selected_facility_id = None

    if incident_report is not None:

        selected_facility_id = (
            incident_report.get(
                "recommended_facility_id",
                None
            )
        )

    # --------------------------------------------------------
    # Facilities
    # --------------------------------------------------------

    add_facilities(
        m,
        scenario,
        selected_facility_id
    )

    # --------------------------------------------------------
    # Incidents
    # --------------------------------------------------------

    add_incidents(
        m,
        scenario,
        incident_id
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    add_response_route(
        m,
        scenario,
        incident_id
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    legend = """
    <div style="
        position:fixed;
        bottom:25px;
        left:25px;
        z-index:9999;
        background:white;
        padding:13px 16px;
        border-radius:9px;
        box-shadow:0 2px 8px rgba(0,0,0,0.25);
        font-family:Arial;
        font-size:12px;
        line-height:19px;
    ">

    <b style="font-size:13px">
        URIP MAP LEGEND
    </b>

    <hr style="margin:6px 0">

    <b>Flood Impact</b><br>

    <span style="
        background:#d2ebff;
        width:22px;height:12px;
        display:inline-block;
    "></span>
    Low<br>

    <span style="
        background:#8ac6ff;
        width:22px;height:12px;
        display:inline-block;
    "></span>
    Moderate<br>

    <span style="
        background:#3b8edb;
        width:22px;height:12px;
        display:inline-block;
    "></span>
    High<br>

    <span style="
        background:#0057a8;
        width:22px;height:12px;
        display:inline-block;
    "></span>
    Very High

    <br><br>

    <b>Road Status</b><br>

    <span style="
        background:#2ca25f;
        width:22px;height:4px;
        display:inline-block;
    "></span>
    Passable<br>

    <span style="
        background:#fdae6b;
        width:22px;height:4px;
        display:inline-block;
    "></span>
    Minor Disruption<br>

    <span style="
        background:#e34a33;
        width:22px;height:4px;
        display:inline-block;
    "></span>
    Severe Disruption<br>

    <span style="
        background:#7f0000;
        width:22px;height:4px;
        display:inline-block;
    "></span>
    Closed

    <br><br>

    <b>Emergency</b><br>

    <span style="
        background:#ffcc00;
        width:12px;height:12px;
        border-radius:50%;
        display:inline-block;
    "></span>
    Incident<br>

    <span style="
        background:#00ffff;
        width:22px;height:5px;
        display:inline-block;
    "></span>
    Response Route<br>

    <span style="
        border:2px dashed #1f2937;
        width:20px;height:10px;
        display:inline-block;
    "></span>
    Analysis Boundary

    </div>
    """

    m.get_root().html.add_child(
        Element(legend)
    )

    folium.LayerControl(
        collapsed=False
    ).add_to(m)

    return m


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## URIP Controls"
    )

    st.caption(
        "Urban Resilience Intelligence Platform"
    )

    scenario = st.selectbox(
        "Rainfall Scenario",
        SCENARIOS,
        index=SCENARIOS.index(
            "Normal"
        )
        if "Normal" in SCENARIOS
        else 0
    )

    map_mode = st.radio(
        "Map Mode",
        [
            "Flood Scenario",
            "Emergency Response"
        ]
    )

    incident_id = st.selectbox(
        "Emergency Incident",
        INCIDENTS,
        index=0
    )

    st.markdown("---")

    metadata_items = [
        (
            "Model",
            METADATA.get(
                "model_version",
                "URIP"
            )
        ),
        (
            "Study Area",
            METADATA.get(
                "study_area",
                "Nairobi CBD"
            )
        ),
        (
            "Analysis CRS",
            METADATA.get(
                "analysis_crs",
                CRS_METRIC
            )
        ),
        (
            "Population",
            METADATA.get(
                "population_type",
                "Estimated Daytime Population"
            )
        ),
        (
            "Traffic",
            METADATA.get(
                "traffic_type",
                "Modelled Traffic Volume"
            )
        ),
    ]

    for label, value in metadata_items:

        st.caption(
            f"**{label}:** {value}"
        )


# ============================================================
# HEADER
# ============================================================

scenario_row = get_summary_row(
    scenario
)

emergency_kpi = get_emergency_kpi(
    scenario
)

emergency_situation = get_emergency_situation(
    scenario
)

incident = get_incident_report(
    scenario,
    incident_id
)


st.markdown(
    f"""
    <div class="urip-header">

        <div class="urip-title">
            URIP
        </div>

        <div class="urip-subtitle">
            Urban Resilience Intelligence Platform
            &nbsp;|&nbsp;
            Nairobi CBD Emergency Response Simulation
        </div>

        <div style="
            margin-top:10px;
            font-size:0.85rem;
            opacity:0.82;
        ">
            Scenario: <b>{scenario}</b>
            &nbsp; • &nbsp;
            Rainfall:
            <b>{safe_float(
                scenario_row.get(
                    "rainfall_mm_hr",
                    0
                )
            ):,.0f} mm/hr</b>
            &nbsp; • &nbsp;
            Map: <b>{map_mode}</b>
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CORE KPI ROW
# ============================================================

if scenario_row is not None:

    cols = st.columns(5)

    values = [
        (
            "Flood Impact",
            format_number(
                scenario_row.get(
                    "mean_flood_impact",
                    0
                ),
                3
            ),
            "mean score"
        ),
        (
            "Roads Affected",
            format_number(
                scenario_row.get(
                    "roads_affected",
                    0
                )
            ),
            "roads"
        ),
        (
            "Roads Closed",
            format_number(
                scenario_row.get(
                    "roads_closed",
                    0
                )
            ),
            "roads"
        ),
        (
            "Mean V/C",
            format_number(
                scenario_row.get(
                    "mean_vc",
                    0
                ),
                2
            ),
            "volume / capacity"
        ),
        (
            "Facilities Affected",
            format_number(
                scenario_row.get(
                    "facilities_affected",
                    0
                )
            ),
            "facilities"
        ),
    ]

    for col, item in zip(
        cols,
        values
    ):

        with col:

            st.markdown(
                kpi_card(
                    *item
                ),
                unsafe_allow_html=True
            )


# ============================================================
# POPULATION / EMERGENCY KPIS
# ============================================================

st.markdown(
    '<div class="section-title">Resilience & Emergency Indicators</div>',
    unsafe_allow_html=True
)

cols = st.columns(5)

population_values = [
    (
        "Population Exposed",
        scenario_row.get(
            "population_exposed",
            0
        ),
        "people"
    ),
    (
        "Access Disrupted",
        scenario_row.get(
            "access_disrupted",
            0
        ),
        "people"
    ),
    (
        "Priority Population",
        scenario_row.get(
            "priority_population",
            0
        ),
        "people"
    ),
    (
        "High Priority",
        scenario_row.get(
            "high_priority_population",
            0
        ),
        "people"
    ),
    (
        "Mean Response Time",
        emergency_kpi.get(
            "mean_response_time_min",
            0
        ),
        "minutes"
    ),
]

for col, item in zip(
    cols,
    population_values
):

    with col:

        decimals = (
            2
            if item[0]
            == "Mean Response Time"
            else 0
        )

        st.markdown(
            kpi_card(
                item[0],
                format_number(
                    item[1],
                    decimals
                ),
                item[2]
            ),
            unsafe_allow_html=True
        )


# ============================================================
# SITUATION STATUS
# ============================================================

st.markdown(
    '<div class="section-title">Emergency Situation</div>',
    unsafe_allow_html=True
)

status_cols = st.columns(4)

status_values = [
    (
        "Situation",
        emergency_situation.get(
            "situation_status",
            "Unknown"
        )
    ),
    (
        "Response Network",
        emergency_situation.get(
            "response_network_status",
            "Unknown"
        )
    ),
    (
        "Facility Status",
        emergency_situation.get(
            "facility_status",
            "Unknown"
        )
    ),
    (
        "Route Availability",
        f"{safe_float(
            emergency_situation.get(
                "route_coverage_pct",
                0
            )
        ):.1f}%"
    ),
]

for col, item in zip(
    status_cols,
    status_values
):

    with col:

        st.markdown(
            f"""
            <div class="status-card">

                <div class="status-title">
                    {item[0]}
                </div>

                <div class="status-value">
                    {item[1]}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# ALERT
# ============================================================

alert = emergency_kpi.get(
    "emergency_alert",
    ""
)

if alert:

    if (
        "No incident-level"
        in str(alert)
    ):

        st.markdown(
            f"""
            <div class="info-box">
                <b>Emergency intelligence:</b>
                {alert}
            </div>
            """,
            unsafe_allow_html=True
        )

    elif (
        "High-priority"
        in str(alert)
    ):

        st.markdown(
            f"""
            <div class="danger-box">
                <b>Emergency alert:</b>
                {alert}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="alert-box">
                <b>Emergency alert:</b>
                {alert}
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# MAP
# ============================================================

st.markdown(
    '<div class="section-title">Spatial Intelligence</div>',
    unsafe_allow_html=True
)

m = build_map(
    scenario,
    map_mode,
    incident_id
)

st_folium(
    m,
    width=None,
    height=680,
    returned_objects=[]
)


# ============================================================
# INCIDENT INTELLIGENCE
# ============================================================

st.markdown(
    '<div class="section-title">Emergency Incident Intelligence</div>',
    unsafe_allow_html=True
)


if incident is not None:

    incident_cols = st.columns(5)

    incident_values = [
        (
            "Incident Flood Class",
            incident.get(
                "incident_flood_class",
                "—"
            )
        ),
        (
            "Incident Flood Impact",
            format_number(
                incident.get(
                    "incident_flood_impact",
                    np.nan
                ),
                3
            )
        ),
        (
            "Catchment Population",
            format_number(
                incident.get(
                    "population_in_catchment",
                    0
                )
            )
        ),
        (
            "Priority Population",
            format_number(
                incident.get(
                    "priority_population",
                    0
                )
            )
        ),
        (
            "High Priority",
            format_number(
                incident.get(
                    "high_priority_population",
                    0
                )
            )
        ),
    ]

    for col, item in zip(
        incident_cols,
        incident_values
    ):

        with col:

            st.markdown(
                kpi_card(
                    item[0],
                    item[1],
                    ""
                ),
                unsafe_allow_html=True
            )


# ============================================================
# FACILITY + ROUTE TABLE
# ============================================================

facility_options = get_facility_options(
    scenario,
    incident_id
)

table_rows = []


for _, facility in facility_options.iterrows():

    rank = int(
        facility[
            "facility_rank"
        ]
    )

    routes = get_routes_for_facility(
        scenario,
        incident_id,
        rank
    )

    if routes.empty:

        table_rows.append({
            "Option": f"Option {rank}",
            "Facility": facility[
                "facility_name"
            ],
            "Type": facility[
                "facility_type"
            ],
            "Facility Time": (
                safe_float(
                    facility[
                        "travel_time_min"
                    ]
                )
            ),
            "Facility Delay": (
                safe_float(
                    facility[
                        "response_delay_min"
                    ]
                )
            ),
            "Route": "No route",
            "Route Time": np.nan,
            "Route Delay": np.nan,
            "Length": np.nan,
            "Affected": np.nan,
            "Closed": np.nan
        })

    else:

        for _, route in routes.iterrows():

            table_rows.append({
                "Option": f"Option {rank}",
                "Facility": facility[
                    "facility_name"
                ],
                "Type": facility[
                    "facility_type"
                ],
                "Facility Time": (
                    safe_float(
                        facility[
                            "travel_time_min"
                        ]
                    )
                ),
                "Facility Delay": (
                    safe_float(
                        facility[
                            "response_delay_min"
                        ]
                    )
                ),
                "Route": (
                    f"Route "
                    f"{int(
                        route['route_rank']
                    )}"
                ),
                "Route Time": (
                    safe_float(
                        route[
                            "response_time_min"
                        ]
                    )
                ),
                "Route Delay": (
                    safe_float(
                        route[
                            "response_delay_min"
                        ]
                    )
                ),
                "Length": (
                    safe_float(
                        route[
                            "route_length_m"
                        ]
                    )
                ),
                "Affected": (
                    safe_float(
                        route[
                            "affected_segments"
                        ]
                    )
                ),
                "Closed": (
                    safe_float(
                        route[
                            "closed_segments"
                        ]
                    )
                )
            })


# ------------------------------------------------------------
# Display table
# ------------------------------------------------------------

if table_rows:

    display_table = pd.DataFrame(
        table_rows
    )

    display_table = display_table.rename(
        columns={
            "Option": "Option",
            "Facility": "Facility",
            "Type": "Type",
            "Facility Time": "Facility Time (min)",
            "Facility Delay": "Facility Delay (min)",
            "Route": "Route",
            "Route Time": "Route Time (min)",
            "Route Delay": "Route Delay (min)",
            "Length": "Length (m)",
            "Affected": "Affected Segments",
            "Closed": "Closed Segments"
        }
    )

    # Numeric formatting
    for col in [
        "Facility Time (min)",
        "Facility Delay (min)",
        "Route Time (min)",
        "Route Delay (min)",
        "Length (m)"
    ]:

        display_table[col] = (
            display_table[col]
            .apply(
                lambda x:
                    "—"
                    if pd.isna(x)
                    else f"{x:,.2f}"
            )
        )

    for col in [
        "Affected Segments",
        "Closed Segments"
    ]:

        display_table[col] = (
            display_table[col]
            .apply(
                lambda x:
                    "—"
                    if pd.isna(x)
                    else f"{x:,.0f}"
            )
        )

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
        height=390
    )


# ============================================================
# MODEL NOTE
# ============================================================

st.markdown(
    f"""
    <div class="info-box">

    <b>Model note:</b>
    URIP is displaying frozen model outputs from
    <b>{METADATA.get(
        "model_version",
        "URIP"
    )}</b>.
    No flood, traffic, routing, population or facility
    calculations are performed by the dashboard.

    The emergency population catchment is
    <b>{METADATA.get(
        "catchment_distance_m",
        1000
    )} m network distance</b>
    using the normal network as the reference catchment.

    The emergency V/C threshold is a
    <b>URIP modelling assumption</b>,
    not an official Nairobi operational standard.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        URIP — Urban Resilience Intelligence Platform
        · Nairobi CBD Emergency Response Simulation
        · Frozen analytical model / interactive decision-support interface
    </div>
    """,
    unsafe_allow_html=True
)

        
