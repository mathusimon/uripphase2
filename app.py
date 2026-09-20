

# ============================================================
# URIP — URBAN RESILIENCE INTELLIGENCE PLATFORM
# Nairobi CBD Emergency Response Simulation
#
# STREAMLIT VISUALIZATION LAYER
#
# IMPORTANT:
# - This application does NOT recalculate the URIP model.
# - It only loads and visualizes frozen outputs.
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
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Frozen model not found: {MODEL_PATH}"
        )
        st.stop()

    try:

        with open(
            MODEL_PATH,
            "rb"
        ) as f:

            model = pickle.load(f)

    except Exception as exc:

        st.error(
            "The frozen URIP model could not be loaded."
        )

        st.exception(exc)

        st.stop()

    if not isinstance(model, dict):

        st.error(
            "URIP_MODEL.pkl does not contain the expected dictionary package."
        )

        st.stop()

    return model


URIP_MODEL = load_model()


# ============================================================
# GENERIC HELPERS
# ============================================================

def safe_float(
    value,
    default=0.0
):

    try:

        if value is None:
            return default

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

    try:

        value = float(value)

        return f"{value:,.{decimals}f}"

    except Exception:

        return str(value)


def get_dict(
    parent,
    key
):

    if isinstance(parent, dict):

        value = parent.get(
            key,
            {}
        )

        if isinstance(value, dict):
            return value

    return {}


def get_table(
    parent,
    key
):

    if not isinstance(parent, dict):
        return None

    value = parent.get(key)

    if isinstance(value, pd.DataFrame):
        return value

    if isinstance(value, gpd.GeoDataFrame):
        return value

    return None


def filter_scenario(
    table,
    scenario
):

    if table is None:
        return None

    if not hasattr(
        table,
        "columns"
    ):
        return None

    if "scenario" not in table.columns:
        return table

    result = table[
        table["scenario"] == scenario
    ].copy()

    return result


def first_row(
    table
):

    if table is None:
        return None

    if len(table) == 0:
        return None

    return table.iloc[0]


def value_from(
    row,
    *keys,
    default=None
):

    if row is None:
        return default

    for key in keys:

        try:

            if isinstance(
                row,
                dict
            ):

                if key in row:
                    return row[key]

            elif key in row.index:

                value = row[key]

                if not pd.isna(value):
                    return value

        except Exception:

            pass

    return default


# ============================================================
# MODEL STRUCTURE
# ============================================================

METADATA = get_dict(
    URIP_MODEL,
    "metadata"
)

THRESHOLDS = get_dict(
    URIP_MODEL,
    "thresholds"
)

SCENARIO_DATA = get_dict(
    URIP_MODEL,
    "scenarios"
)

POPULATION_DATA = get_dict(
    URIP_MODEL,
    "population"
)

FACILITY_DATA = get_dict(
    URIP_MODEL,
    "facilities"
)


# ============================================================
# SCENARIOS
# ============================================================

SCENARIOS = list(
    SCENARIO_DATA.keys()
)

if not SCENARIOS:

    SCENARIOS = [
        "Normal",
        "Moderate Rainfall",
        "Heavy Rainfall",
        "Severe Rainfall"
    ]

DEFAULT_SCENARIO = (
    "Normal"
    if "Normal" in SCENARIOS
    else SCENARIOS[0]
)


# ============================================================
# SCENARIO ACCESS
# ============================================================

def get_scenario_object(
    scenario
):

    return SCENARIO_DATA.get(
        scenario,
        {}
    )


def get_scenario_table(
    scenario,
    possible_keys=None
):

    scenario_object = (
        get_scenario_object(
            scenario
        )
    )

    if isinstance(
        scenario_object,
        pd.DataFrame
    ):

        return scenario_object

    if isinstance(
        scenario_object,
        gpd.GeoDataFrame
    ):

        return scenario_object

    if isinstance(
        scenario_object,
        dict
    ):

        keys = (
            possible_keys
            or []
        )

        for key in keys:

            value = (
                scenario_object.get(
                    key
                )
            )

            if isinstance(
                value,
                (
                    pd.DataFrame,
                    gpd.GeoDataFrame
                )
            ):

                return value

    return None


# ============================================================
# SUMMARY ACCESS
# ============================================================

def get_summary_row(
    scenario
):

    summary = URIP_MODEL.get(
        "summary"
    )

    if isinstance(
        summary,
        pd.DataFrame
    ):

        if "scenario" in summary.columns:

            rows = summary[
                summary["scenario"]
                == scenario
            ]

            return first_row(rows)

        return first_row(summary)

    scenario_object = (
        get_scenario_object(
            scenario
        )
    )

    if isinstance(
        scenario_object,
        dict
    ):

        for key in [
            "summary",
            "kpi",
            "kpis",
            "dashboard",
            "dashboard_summary"
        ]:

            value = (
                scenario_object.get(
                    key
                )
            )

            if isinstance(
                value,
                pd.DataFrame
            ):

                return first_row(value)

            if isinstance(
                value,
                dict
            ):

                return value

    return scenario_object


# ============================================================
# EMERGENCY ACCESSORS
# ============================================================

def get_emergency_section():

    section = URIP_MODEL.get(
        "emergency"
    )

    if isinstance(
        section,
        dict
    ):

        return section

    return {}


def get_emergency_kpi(
    scenario
):

    emergency = (
        get_emergency_section()
    )

    table = emergency.get(
        "kpis"
    )

    if isinstance(
        table,
        pd.DataFrame
    ):

        return first_row(
            table[
                table["scenario"]
                == scenario
            ]
        )

    return None


def get_emergency_situation(
    scenario
):

    emergency = (
        get_emergency_section()
    )

    table = emergency.get(
        "situation_report"
    )

    if isinstance(
        table,
        pd.DataFrame
    ):

        return first_row(
            table[
                table["scenario"]
                == scenario
            ]
        )

    return None


def get_incident_report(
    scenario,
    incident_id
):

    emergency = (
        get_emergency_section()
    )

    table = emergency.get(
        "incident_report"
    )

    if not isinstance(
        table,
        pd.DataFrame
    ):

        return None

    if (
        "scenario" not in table.columns
        or
        "incident_id" not in table.columns
    ):

        return None

    rows = table[
        (table["scenario"] == scenario)
        &
        (
            table["incident_id"]
            == incident_id
        )
    ]

    return first_row(rows)


# ============================================================
# INCIDENT ACCESS
# ============================================================

def get_incidents():

    incidents = URIP_MODEL.get(
        "incidents"
    )

    if isinstance(
        incidents,
        dict
    ):

        points = incidents.get(
            "points"
        )

        if isinstance(
            points,
            (
                pd.DataFrame,
                gpd.GeoDataFrame
            )
        ):

            if "incident_id" in points.columns:

                return points

    for scenario in SCENARIOS:

        obj = get_scenario_object(
            scenario
        )

        if isinstance(
            obj,
            dict
        ):

            for key in [
                "incidents",
                "incident_points",
                "points"
            ]:

                points = obj.get(key)

                if isinstance(
                    points,
                    (
                        pd.DataFrame,
                        gpd.GeoDataFrame
                    )
                ):

                    if "incident_id" in points.columns:

                        return points

    return None


INCIDENTS_GDF = get_incidents()

if INCIDENTS_GDF is not None:

    INCIDENTS = (
        INCIDENTS_GDF[
            "incident_id"
        ]
        .astype(str)
        .tolist()
    )

else:

    INCIDENTS = [
        f"INC_{i:03d}"
        for i in range(1, 21)
    ]


# ============================================================
# FACILITY ACCESS
# ============================================================

def get_facilities():

    base = FACILITY_DATA.get(
        "base"
    )

    if isinstance(
        base,
        (
            pd.DataFrame,
            gpd.GeoDataFrame
        )
    ):

        return base.copy()

    if isinstance(
        base,
        dict
    ):

        for key in [
            "gdf",
            "facilities",
            "data"
        ]:

            value = base.get(key)

            if isinstance(
                value,
                (
                    pd.DataFrame,
                    gpd.GeoDataFrame
                )
            ):

                return value.copy()

    return None


FACILITIES_GDF = get_facilities()


def get_facility_options(
    scenario,
    incident_id
):

    table = URIP_MODEL.get(
        "facility_options"
    )

    if not isinstance(
        table,
        pd.DataFrame
    ):

        return pd.DataFrame()

    required = [
        "scenario",
        "incident_id"
    ]

    if not all(
        c in table.columns
        for c in required
    ):

        return pd.DataFrame()

    result = table[
        (table["scenario"] == scenario)
        &
        (
            table["incident_id"]
            == incident_id
        )
    ].copy()

    if "facility_rank" in result.columns:

        result = (
            result
            .sort_values(
                "facility_rank"
            )
            .head(3)
        )

    return result


# ============================================================
# ROUTE ACCESS
# ============================================================

def get_routes_for_facility(
    scenario,
    incident_id,
    facility_rank
):

    routes_section = URIP_MODEL.get(
        "routes"
    )

    if not isinstance(
        routes_section,
        dict
    ):

        return pd.DataFrame()

    table = routes_section.get(
        "alternatives"
    )

    if not isinstance(
        table,
        pd.DataFrame
    ):

        return pd.DataFrame()

    required = [
        "scenario",
        "incident_id",
        "facility_rank"
    ]

    if not all(
        c in table.columns
        for c in required
    ):

        return pd.DataFrame()

    result = table[
        (table["scenario"] == scenario)
        &
        (
            table["incident_id"]
            == incident_id
        )
        &
        (
            table["facility_rank"]
            == facility_rank
        )
    ].copy()

    if "route_rank" in result.columns:

        result = (
            result
            .sort_values(
                "route_rank"
            )
            .head(3)
        )

    return result


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
# FLOOD MAP
# ============================================================

def add_flood_layer(
    m,
    scenario
):

    flood = URIP_MODEL.get(
        "flood"
    )

    if not isinstance(
        flood,
        dict
    ):

        return

    raster = flood.get(
        scenario
    )

    if raster is None:
        return

    try:

        raster = np.asarray(
            raster,
            dtype=float
        )

    except Exception:

        return

    if raster.ndim != 2:
        return

    valid = np.isfinite(
        raster
    )

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

    bounds = [
        [-1.315, 36.79],
        [-1.260, 36.85]
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
# FLOOD ANALYSIS BOUNDARY
# ============================================================

def add_analysis_boundary(
    m
):

    bounds = [
        [-1.315, 36.79],
        [-1.260, 36.85]
    ]

    coordinates = [
        [bounds[0][0], bounds[0][1]],
        [bounds[0][0], bounds[1][1]],
        [bounds[1][0], bounds[1][1]],
        [bounds[1][0], bounds[0][1]],
        [bounds[0][0], bounds[0][1]]
    ]

    group = folium.FeatureGroup(
        name="Flood Analysis Boundary",
        show=True
    )

    folium.PolyLine(
        coordinates,
        color="#1f2937",
        weight=5,
        opacity=0.45
    ).add_to(group)

    folium.PolyLine(
        coordinates,
        color="#ffffff",
        weight=3,
        opacity=1,
        dash_array="8,6"
    ).add_to(group)

    group.add_to(m)


# ============================================================
# ROADS
# ============================================================

def get_roads(
    scenario
):

    roads_section = URIP_MODEL.get(
        "roads"
    )

    if isinstance(
        roads_section,
        dict
    ):

        roads = roads_section.get(
            scenario
        )

        if isinstance(
            roads,
            (
                pd.DataFrame,
                gpd.GeoDataFrame
            )
        ):

            return roads.copy()

    scenario_object = (
        get_scenario_object(
            scenario
        )
    )

    if isinstance(
        scenario_object,
        dict
    ):

        for key in [
            "roads",
            "road_network",
            "road_results"
        ]:

            roads = scenario_object.get(
                key
            )

            if isinstance(
                roads,
                (
                    pd.DataFrame,
                    gpd.GeoDataFrame
                )
            ):

                return roads.copy()

    return None


def add_roads(
    m,
    scenario
):

    roads = get_roads(
        scenario
    )

    if roads is None:
        return

    if (
        not isinstance(
            roads,
            gpd.GeoDataFrame
        )
        or
        roads.crs is None
    ):

        return

    if roads.crs != CRS_GEO:

        roads = roads.to_crs(
            CRS_GEO
        )

    roads = roads[
        roads.geometry.notna()
        &
        ~roads.geometry.is_empty
    ]

    if roads.empty:
        return

    candidates = [
        "passability_class",
        "passability",
        "road_status",
        "disruption_class",
        "road_disruption"
    ]

    status_column = None

    for column in candidates:

        if column in roads.columns:

            status_column = column
            break

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
        "congestion_class"
    ]

    columns = [
        c for c in columns
        if c in roads.columns
    ]

    roads_map = roads[
        columns
    ].copy()

    tooltip_fields = [
        c for c in [
            "road_id",
            "flood_impact",
            status_column,
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
        for c in tooltip_fields
    ]

    def style_function(
        feature
    ):

        properties = (
            feature.get(
                "properties",
                {}
            )
        )

        status = (
            properties.get(
                status_column,
                "No Data"
            )
            if status_column
            else "No Data"
        )

        return {
            "color": ROAD_COLORS.get(
                str(status),
                "#969696"
            ),
            "weight": (
                1.2
                if str(status)
                == "Passable"
                else 2.8
            ),
            "opacity": 0.85
        }

    group = folium.FeatureGroup(
        name="Road Network / Passability",
        show=True
    )

    tooltip = None

    if tooltip_fields:

        tooltip = folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=aliases,
            localize=True
        )

    folium.GeoJson(
        roads_map.to_json(),
        style_function=style_function,
        highlight_function=lambda feature: {
            "color": "#ffffff",
            "weight": 4,
            "opacity": 1
        },
        tooltip=tooltip
    ).add_to(group)

    group.add_to(m)


# ============================================================
# FACILITIES
# ============================================================

def add_facilities(
    m,
    scenario,
    selected_facility_id=None
):

    facilities = FACILITIES_GDF

    if facilities is None:
        return

    facilities = facilities.copy()

    if not isinstance(
        facilities,
        gpd.GeoDataFrame
    ):

        return

    if facilities.crs is None:
        return

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

        if geometry.is_empty:
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

        color = (
            "#00ff88"
            if selected
            else (
                "#c0392b"
                if affected
                else "#2471a3"
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
            row.get(
                "response_time",
                np.nan
            ),
            np.nan
        ):.2f} min<br>

        <b>Response delay:</b>
        {safe_float(
            row.get(
                "response_delay",
                np.nan
            ),
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
            radius=9 if selected else 5,
            color=color,
            fill=True,
            fill_color=color,
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
# INCIDENTS
# ============================================================

def add_incidents(
    m,
    scenario,
    incident_id
):

    incidents = INCIDENTS_GDF

    if incidents is None:
        return

    if incidents.crs is None:
        return

    incidents = incidents.copy()

    if incidents.crs != CRS_GEO:

        incidents = incidents.to_crs(
            CRS_GEO
        )

    group = folium.FeatureGroup(
        name="Emergency Incidents",
        show=True
    )

    for _, row in incidents.iterrows():

        current_id = str(
            row.get(
                "incident_id",
                ""
            )
        )

        selected = (
            current_id
            == str(
                incident_id
            )
        )

        report = get_incident_report(
            scenario,
            current_id
        )

        if report is not None:

            popup = f"""
            <div style="font-family:Arial;width:270px">

            <h4>{current_id}</h4>

            <b>Scenario:</b> {scenario}<br>

            <b>Incident flood class:</b>
            {value_from(
                report,
                "incident_flood_class",
                default="—"
            )}<br>

            <b>Flood impact:</b>
            {safe_float(
                value_from(
                    report,
                    "incident_flood_impact",
                    default=np.nan
                ),
                np.nan
            ):.3f}<br>

            <b>Catchment population:</b>
            {safe_float(
                value_from(
                    report,
                    "population_in_catchment",
                    default=0
                )
            ):,.0f}<br>

            <b>Flood exposed:</b>
            {safe_float(
                value_from(
                    report,
                    "flood_exposed_population",
                    default=0
                )
            ):,.0f}<br>

            <b>Access disrupted:</b>
            {safe_float(
                value_from(
                    report,
                    "access_disrupted_population",
                    default=0
                )
            ):,.0f}<br>

            <b>Priority population:</b>
            {safe_float(
                value_from(
                    report,
                    "priority_population",
                    default=0
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
# RESPONSE ROUTE
# ============================================================

def add_response_route(
    m,
    scenario,
    incident_id
):

    routes_section = URIP_MODEL.get(
        "routes"
    )

    if not isinstance(
        routes_section,
        dict
    ):

        return None

    routes = routes_section.get(
        "alternatives"
    )

    if not isinstance(
        routes,
        pd.DataFrame
    ):

        return None

    required = [
        "scenario",
        "incident_id",
        "facility_rank",
        "route_rank"
    ]

    if not all(
        c in routes.columns
        for c in required
    ):

        return None

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

    facility_id = route.get(
        "facility_id"
    )

    geometry = route.get(
        "geometry"
    )

    if geometry is None:
        return facility_id

    try:

        route_gdf = gpd.GeoDataFrame(
            [route],
            geometry="geometry",
            crs=CRS_METRIC
        )

        route_gdf = route_gdf.to_crs(
            CRS_GEO
        )

        route_geometry = (
            route_gdf.geometry.iloc[0]
        )

        group = folium.FeatureGroup(
            name="Emergency Response Route",
            show=True
        )

        folium.GeoJson(
            mapping(route_geometry),
            style_function=lambda feature: {
                "color": "#00ffff",
                "weight": 6,
                "opacity": 0.95
            },
            tooltip=(
                "Recommended response route | "
                f"{safe_float(
                    route.get(
                        'response_time_min',
                        0
                    )
                ):.2f} min"
            )
        ).add_to(group)

        group.add_to(m)

    except Exception:

        pass

    return facility_id


# ============================================================
# MAP
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
        overlay=False,
        control=True
    ).add_to(m)

    folium.TileLayer(
        "OpenStreetMap",
        name="Street Map",
        overlay=False,
        control=True
    ).add_to(m)

    # --------------------------------------------------------
    # Flood scenario mode
    # --------------------------------------------------------

    if map_mode == "Flood Scenario":

        add_flood_layer(
            m,
            scenario
        )

        add_analysis_boundary(
            m
        )

        add_roads(
            m,
            scenario
        )

        add_facilities(
            m,
            scenario
        )

        add_incidents(
            m,
            scenario,
            incident_id
        )

    # --------------------------------------------------------
    # Emergency response mode
    # --------------------------------------------------------

    else:

        add_roads(
            m,
            scenario
        )

        add_facilities(
            m,
            scenario
        )

        add_incidents(
            m,
            scenario,
            incident_id
        )

        incident_report = (
            get_incident_report(
                scenario,
                incident_id
            )
        )

        selected_facility_id = None

        if incident_report is not None:

            selected_facility_id = value_from(
                incident_report,
                "recommended_facility_id"
            )

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
            DEFAULT_SCENARIO
        )
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

    st.caption(
        f"**Study Area:** "
        f"{METADATA.get(
            'study_area',
            'Nairobi CBD'
        )}"
    )

    st.caption(
        f"**Analysis CRS:** "
        f"{METADATA.get(
            'analysis_crs',
            CRS_METRIC
        )}"
    )

    st.caption(
        f"**Population:** "
        f"{METADATA.get(
            'population_type',
            'Estimated Daytime Population'
        )}"
    )

    st.caption(
        f"**Traffic:** "
        f"{METADATA.get(
            'traffic_type',
            'Modelled Traffic Volume'
        )}"
    )


# ============================================================
# CURRENT DATA
# ============================================================

scenario_row = get_summary_row(
    scenario
)

emergency_kpi = get_emergency_kpi(
    scenario
)

emergency_situation = (
    get_emergency_situation(
        scenario
    )
)

incident = get_incident_report(
    scenario,
    incident_id
)


# ============================================================
# HEADER
# ============================================================

rainfall = value_from(
    scenario_row,
    "rainfall_mm_hr",
    "rainfall",
    default=0
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

            Scenario:
            <b>{scenario}</b>

            &nbsp; • &nbsp;

            Rainfall:
            <b>{safe_float(
                rainfall
            ):,.0f} mm/hr</b>

            &nbsp; • &nbsp;

            Map:
            <b>{map_mode}</b>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CORE KPIs
# ============================================================

st.markdown(
    '<div class="section-title">Urban Resilience Indicators</div>',
    unsafe_allow_html=True
)

cols = st.columns(5)

core_values = [

    (
        "Flood Impact",
        value_from(
            scenario_row,
            "mean_flood_impact",
            "flood_impact",
            default=0
        ),
        "mean score",
        3
    ),

    (
        "Roads Affected",
        value_from(
            scenario_row,
            "roads_affected",
            "affected_roads",
            default=0
        ),
        "roads",
        0
    ),

    (
        "Roads Closed",
        value_from(
            scenario_row,
            "roads_closed",
            "closed_roads",
            default=0
        ),
        "roads",
        0
    ),

    (
        "Mean V/C",
        value_from(
            scenario_row,
            "mean_vc",
            "mean_final_vc",
            default=0
        ),
        "volume / capacity",
        2
    ),

    (
        "Facilities Affected",
        value_from(
            scenario_row,
            "facilities_affected",
            "operationally_affected_facilities",
            default=0
        ),
        "facilities",
        0
    )
]


for col, item in zip(
    cols,
    core_values
):

    with col:

        st.markdown(
            kpi_card(
                item[0],
                format_number(
                    item[1],
                    item[3]
                ),
                item[2]
            ),
            unsafe_allow_html=True
        )


# ============================================================
# POPULATION / EMERGENCY KPIs
# ============================================================

st.markdown(
    '<div class="section-title">Resilience & Emergency Indicators</div>',
    unsafe_allow_html=True
)

cols = st.columns(5)

population_values = [

    (
        "Population Exposed",
        value_from(
            scenario_row,
            "population_exposed",
            "exposed_population",
            default=0
        ),
        "people",
        0
    ),

    (
        "Access Disrupted",
        value_from(
            scenario_row,
            "access_disrupted",
            "access_disrupted_population",
            default=0
        ),
        "people",
        0
    ),

    (
        "Priority Population",
        value_from(
            scenario_row,
            "priority_population",
            default=0
        ),
        "people",
        0
    ),

    (
        "High Priority",
        value_from(
            scenario_row,
            "high_priority_population",
            default=0
        ),
        "people",
        0
    ),

    (
        "Mean Response Time",
        value_from(
            emergency_kpi,
            "mean_response_time_min",
            default=0
        ),
        "minutes",
        2
    )
]


for col, item in zip(
    cols,
    population_values
):

    with col:

        st.markdown(
            kpi_card(
                item[0],
                format_number(
                    item[1],
                    item[3]
                ),
                item[2]
            ),
            unsafe_allow_html=True
        )


# ============================================================
# EMERGENCY SITUATION
# ============================================================

st.markdown(
    '<div class="section-title">Emergency Situation</div>',
    unsafe_allow_html=True
)

status_cols = st.columns(4)

status_values = [

    (
        "Situation",
        value_from(
            emergency_situation,
            "situation_status",
            default="Operational"
        )
    ),

    (
        "Response Network",
        value_from(
            emergency_situation,
            "response_network_status",
            default="Unknown"
        )
    ),

    (
        "Facility Status",
        value_from(
            emergency_situation,
            "facility_status",
            default="Unknown"
        )
    ),

    (
        "Route Availability",
        f"{safe_float(
            value_from(
                emergency_situation,
                "route_coverage_pct",
                default=100
            )
        ):.1f}%"
    )
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

alert = value_from(
    emergency_kpi,
    "emergency_alert",
    default=""
)

if alert:

    alert_text = str(
        alert
    )

    if "No incident-level" in alert_text:

        st.markdown(
            f"""
            <div class="info-box">
                <b>Emergency intelligence:</b>
                {alert_text}
            </div>
            """,
            unsafe_allow_html=True
        )

    elif "High-priority" in alert_text:

        st.markdown(
            f"""
            <div class="danger-box">
                <b>Emergency alert:</b>
                {alert_text}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="alert-box">
                <b>Emergency alert:</b>
                {alert_text}
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
            value_from(
                incident,
                "incident_flood_class",
                default="—"
            ),
            ""
        ),

        (
            "Incident Flood Impact",
            format_number(
                value_from(
                    incident,
                    "incident_flood_impact",
                    default=np.nan
                ),
                3
            ),
            ""
        ),

        (
            "Catchment Population",
            format_number(
                value_from(
                    incident,
                    "population_in_catchment",
                    default=0
                )
            ),
            ""
        ),

        (
            "Priority Population",
            format_number(
                value_from(
                    incident,
                    "priority_population",
                    default=0
                )
            ),
            ""
        ),

        (
            "High Priority",
            format_number(
                value_from(
                    incident,
                    "high_priority_population",
                    default=0
                )
            ),
            ""
        )
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
                    item[2]
                ),
                unsafe_allow_html=True
            )

else:

    st.info(
        "Incident-level emergency intelligence is not available in the loaded frozen model."
    )


# ============================================================
# FACILITY + ROUTE OPTIONS
# ============================================================

facility_options = (
    get_facility_options(
        scenario,
        incident_id
    )
)

if not facility_options.empty:

    st.markdown(
        '<div class="section-title">Facility & Route Alternatives</div>',
        unsafe_allow_html=True
    )

    table_rows = []

    for _, facility in (
        facility_options.iterrows()
    ):

        rank = int(
            facility.get(
                "facility_rank",
                0
            )
        )

        routes = (
            get_routes_for_facility(
                scenario,
                incident_id,
                rank
            )
        )

        facility_name = value_from(
            facility,
            "facility_name",
            "name",
            default="Unknown facility"
        )

        facility_type = value_from(
            facility,
            "facility_type",
            default="Facility"
        )

        facility_time = value_from(
            facility,
            "travel_time_min",
            default=np.nan
        )

        facility_delay = value_from(
            facility,
            "response_delay_min",
            default=np.nan
        )

        if routes.empty:

            table_rows.append(
                {
                    "Option": f"Option {rank}",
                    "Facility": facility_name,
                    "Type": facility_type,
                    "Facility Time (min)": facility_time,
                    "Facility Delay (min)": facility_delay,
                    "Route": "No route",
                    "Route Time (min)": np.nan,
                    "Route Delay (min)": np.nan,
                    "Length (m)": np.nan,
                    "Affected Segments": np.nan,
                    "Closed Segments": np.nan
                }
            )

        else:

            for _, route in (
                routes.iterrows()
            ):

                table_rows.append(
                    {
                        "Option": f"Option {rank}",
                        "Facility": facility_name,
                        "Type": facility_type,
                        "Facility Time (min)": facility_time,
                        "Facility Delay (min)": facility_delay,
                        "Route": (
                            f"Route "
                            f"{int(
                                route.get(
                                    'route_rank',
                                    0
                                )
                            )}"
                        ),
                        "Route Time (min)": value_from(
                            route,
                            "response_time_min",
                            default=np.nan
                        ),
                        "Route Delay (min)": value_from(
                            route,
                            "response_delay_min",
                            default=np.nan
                        ),
                        "Length (m)": value_from(
                            route,
                            "route_length_m",
                            default=np.nan
                        ),
                        "Affected Segments": value_from(
                            route,
                            "affected_segments",
                            default=np.nan
                        ),
                        "Closed Segments": value_from(
                            route,
                            "closed_segments",
                            default=np.nan
                        )
                    }
                )

    display_table = pd.DataFrame(
        table_rows
    )

    numeric_two = [
        "Facility Time (min)",
        "Facility Delay (min)",
        "Route Time (min)",
        "Route Delay (min)",
        "Length (m)"
    ]

    numeric_zero = [
        "Affected Segments",
        "Closed Segments"
    ]

    for column in numeric_two:

        if column in display_table.columns:

            display_table[column] = (
                display_table[column]
                .apply(
                    lambda value:
                        "—"
                        if pd.isna(value)
                        else f"{float(value):,.2f}"
                )
            )

    for column in numeric_zero:

        if column in display_table.columns:

            display_table[column] = (
                display_table[column]
                .apply(
                    lambda value:
                        "—"
                        if pd.isna(value)
                        else f"{float(value):,.0f}"
                )
            )

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
        height=390
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">Model Information</div>',
    unsafe_allow_html=True
)

model_name = METADATA.get(
    "name",
    "Urban Resilience Intelligence Platform"
)

model_version = METADATA.get(
    "model_version",
    "URIP"
)

study_area = METADATA.get(
    "study_area",
    "Nairobi CBD"
)

catchment_distance = METADATA.get(
    "catchment_distance_m",
    1000
)

vc_cap = METADATA.get(
    "emergency_vc_cap",
    2.0
)

st.markdown(
    f"""
    <div class="info-box">

    <b>Model:</b> {model_name}<br>

    <b>Version:</b> {model_version}<br>

    <b>Study area:</b> {study_area}<br>

    <b>Emergency catchment:</b>
    {catchment_distance} m network distance<br>

    <b>Emergency V/C cap:</b>
    {vc_cap}<br><br>

    URIP is displaying frozen analytical outputs.
    No flood, traffic, routing, population or facility
    calculations are performed by this dashboard.

    The emergency catchment and V/C threshold are
    URIP modelling parameters and should not be interpreted
    as official Nairobi operational standards.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL VALIDATION MESSAGE
# ============================================================

expected_sections = [
    "flood",
    "roads",
    "incidents",
    "facility_options",
    "routes",
    "emergency",
    "summary"
]

missing_sections = [
    key
    for key in expected_sections
    if key not in URIP_MODEL
]

if missing_sections:

    st.markdown(
        '<div class="section-title">Model Package Notice</div>',
        unsafe_allow_html=True
    )

    st.warning(
        "The loaded URIP_MODEL.pkl does not contain all "
        "dashboard output sections. Missing sections: "
        + ", ".join(missing_sections)
    )

    st.caption(
        "The dashboard will display whatever frozen "
        "outputs are available, but map and emergency "
        "intelligence components requiring those sections "
        "cannot be rendered until the complete frozen "
        "dashboard package is uploaded."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        URIP — Urban Resilience Intelligence Platform
        · Nairobi CBD Emergency Response Simulation

        <br>

        Frozen analytical model
        · Interactive decision-support interface

    </div>
    """,
    unsafe_allow_html=True
)
        
