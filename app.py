
import os
import json
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium

from streamlit_folium import st_folium
from branca.colormap import linear


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="URIP — Nairobi Emergency Response",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "URIP_MODEL.pkl"
)

SUMMARY_PATH = os.path.join(
    BASE_DIR,
    "dashboard_summary.csv"
)

ROADS_PATH = os.path.join(
    BASE_DIR,
    "roads.geojson"
)

FACILITIES_PATH = os.path.join(
    BASE_DIR,
    "facilities.geojson"
)

POPULATION_PATH = os.path.join(
    BASE_DIR,
    "population_display.pkl"
)

ROUTES_PATH = os.path.join(
    BASE_DIR,
    "route_geometries.pkl"
)

ROUTE_COMPARISON_PATH = os.path.join(
    BASE_DIR,
    "route_comparisons.pkl"
)

METADATA_PATH = os.path.join(
    BASE_DIR,
    "metadata.json"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_resource
def load_model():

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_summary():

    return pd.read_csv(SUMMARY_PATH)


@st.cache_data
def load_roads():

    return gpd.read_file(ROADS_PATH)


@st.cache_data
def load_facilities():

    return gpd.read_file(FACILITIES_PATH)


@st.cache_resource
def load_population():

    with open(POPULATION_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_routes():

    with open(ROUTES_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_route_comparisons():

    with open(ROUTE_COMPARISON_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_metadata():

    with open(METADATA_PATH, "r") as f:
        return json.load(f)


MODEL = load_model()
SUMMARY = load_summary()
ROADS = load_roads()
FACILITIES = load_facilities()
POPULATION = load_population()
ROUTES = load_routes()
ROUTE_COMPARISONS = load_route_comparisons()
METADATA = load_metadata()


# ============================================================
# SCENARIOS
# ============================================================

SCENARIOS = [
    "Normal",
    "Moderate Rainfall",
    "Heavy Rainfall",
    "Severe Rainfall"
]


SCENARIO_RAINFALL = {
    "Normal": 10,
    "Moderate Rainfall": 30,
    "Heavy Rainfall": 50,
    "Severe Rainfall": 75
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def fmt(value, decimals=0):

    if value is None:
        return "—"

    try:
        value = float(value)

        if not np.isfinite(value):
            return "—"

        return f"{value:,.{decimals}f}"

    except Exception:
        return "—"


def scenario_row(scenario):

    return SUMMARY[
        SUMMARY["scenario"] == scenario
    ].iloc[0]


def get_scenario_roads(scenario):

    model_roads = MODEL[
        "scenarios"
    ][scenario]["roads"].copy()

    # Add geometry from authoritative display layer.
    display = ROADS[
        ["road_id", "geometry"]
    ].copy()

    return display.merge(
        model_roads,
        on="road_id",
        how="left"
    )


def get_scenario_facilities(scenario):

    base = FACILITIES.copy()

    scenario_facilities = MODEL[
        "scenarios"
    ][scenario]["facilities"].copy()

    columns = [
        c for c in scenario_facilities.columns
        if c != "geometry"
    ]

    return base.merge(
        scenario_facilities[columns],
        on="facility_id",
        how="left"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div style="
        padding: 0.6rem 0 0.2rem 0;
    ">
        <h1 style="margin-bottom:0;">
            🌍 URIP
        </h1>
        <p style="
            font-size:1.1rem;
            margin-top:0;
        ">
            Urban Resilience Intelligence Platform
        </p>
        <p style="
            color:#6b7280;
            margin-top:-0.5rem;
        ">
            Nairobi Emergency Response Simulation
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("Simulation Controls")

scenario = st.sidebar.selectbox(
    "Rainfall Scenario",
    SCENARIOS,
    index=0
)

mode = st.sidebar.radio(
    "Map Mode",
    [
        "Flood Scenario",
        "Emergency Response"
    ]
)


destination_id = None

if mode == "Emergency Response":

    facility_options = (
        FACILITIES[
            [
                "facility_id",
                "facility_name",
                "facility_type"
            ]
        ]
        .fillna("")
    )

    facility_options["label"] = (
        facility_options["facility_name"]
        + " ("
        + facility_options["facility_type"]
        + ") — "
        + facility_options["facility_id"]
    )

    selected_label = st.sidebar.selectbox(
        "Emergency Destination",
        facility_options["label"].tolist()
    )

    destination_id = facility_options.loc[
        facility_options["label"] == selected_label,
        "facility_id"
    ].iloc[0]


# ============================================================
# KPI SECTION
# ============================================================

row = scenario_row(scenario)

st.subheader(
    f"{scenario} — {SCENARIO_RAINFALL[scenario]} mm/hr"
)

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric(
        "Roads Affected",
        fmt(row["roads_affected"])
    )

with k2:
    st.metric(
        "Closed Roads",
        fmt(row["roads_closed"])
    )

with k3:
    st.metric(
        "Mean V/C",
        fmt(row["mean_vc"], 2)
    )

with k4:
    st.metric(
        "Facilities Affected",
        fmt(row["facilities_operationally_affected"])
    )

with k5:
    st.metric(
        "Population Exposed",
        fmt(row["population_exposed"])
    )


# ============================================================
# MAP
# ============================================================

@st.cache_data
def prepare_roads_for_map(
    roads,
    scenario
):

    roads = roads.copy()

    return roads.to_crs("EPSG:4326")


@st.cache_data
def prepare_facilities_for_map(
    facilities
):

    return facilities.to_crs("EPSG:4326")


roads_map = prepare_roads_for_map(
    get_scenario_roads(scenario),
    scenario
)

facilities_map = prepare_facilities_for_map(
    get_scenario_facilities(scenario)
)


m = folium.Map(
    location=[-1.2864, 36.8172],
    zoom_start=13,
    tiles="CartoDB positron",
    control_scale=True
)


# ============================================================
# ROAD LAYERS
# ============================================================

road_group = folium.FeatureGroup(
    name="Road Network",
    show=True
)

affected_group = folium.FeatureGroup(
    name="Affected Roads",
    show=True
)

closed_group = folium.FeatureGroup(
    name="Closed Roads",
    show=True
)


for _, road in roads_map.iterrows():

    disruption = road.get(
        "disruption_class",
        "No Data"
    )

    if disruption == "Closed":

        color = "#dc2626"
        weight = 4
        target = closed_group

    elif disruption == "Severe Disruption":

        color = "#f97316"
        weight = 3
        target = affected_group

    elif disruption == "Minor Disruption":

        color = "#f59e0b"
        weight = 2
        target = affected_group

    else:

        color = "#6b7280"
        weight = 1
        target = road_group


    name = road.get(
        "name",
        "Unnamed Road"
    )

    if isinstance(name, (list, tuple, np.ndarray)):
        name = ", ".join(
            str(x) for x in name
        )

    tooltip = (
        f"{name}<br>"
        f"Status: {disruption}"
    )

    folium.GeoJson(
        road.geometry,
        style_function=lambda feature,
        color=color,
        weight=weight: {
            "color": color,
            "weight": weight,
            "opacity": 0.8
        },
        tooltip=tooltip
    ).add_to(target)


road_group.add_to(m)
affected_group.add_to(m)
closed_group.add_to(m)


# ============================================================
# FACILITIES
# ============================================================

facility_group = folium.FeatureGroup(
    name="Emergency Facilities",
    show=True
)

for _, facility in facilities_map.iterrows():

    facility_name = facility.get(
        "facility_name",
        "Unnamed Facility"
    )

    facility_type = facility.get(
        "facility_type",
        "Emergency Facility"
    )

    folium.CircleMarker(
        location=[
            facility.geometry.y,
            facility.geometry.x
        ],
        radius=5,
        color="#2563eb",
        fill=True,
        fill_opacity=0.85,
        popup=(
            f"<b>{facility_name}</b><br>"
            f"Type: {facility_type}<br>"
            f"ID: {facility.get('facility_id', '')}"
        )
    ).add_to(facility_group)

facility_group.add_to(m)


# ============================================================
# EMERGENCY RESPONSE ROUTE
# ============================================================

if mode == "Emergency Response":

    route_data = ROUTES.get(
        scenario
    )

    if route_data is not None:

        route = route_data[
            route_data["road_id"].isin(
                route_data["road_id"]
            )
        ].copy()

        route = route.to_crs(
            "EPSG:4326"
        )

        route_group = folium.FeatureGroup(
            name="Emergency Response Route",
            show=True
        )

        for _, segment in route.iterrows():

            passability = segment.get(
                "passability",
                1
            )

            if passability <= 0:
                color = "#000000"

            elif passability < 0.25:
                color = "#dc2626"

            elif passability < 1:
                color = "#f59e0b"

            else:
                color = "#2563eb"

            folium.GeoJson(
                segment.geometry,
                style_function=lambda feature,
                color=color: {
                    "color": color,
                    "weight": 6,
                    "opacity": 0.95
                }
            ).add_to(route_group)

        route_group.add_to(m)


# ============================================================
# INCIDENT
# ============================================================

incident = MODEL.get(
    "metadata",
    {}
)

folium.Marker(
    [-1.2864, 36.8172],
    popup="Emergency Incident",
    icon=folium.Icon(
        color="red",
        icon="warning-sign"
    )
).add_to(m)


# ============================================================
# LAYER CONTROL
# ============================================================

folium.LayerControl(
    collapsed=False
).add_to(m)


# ============================================================
# DISPLAY MAP
# ============================================================

st_folium(
    m,
    width=None,
    height=650,
    returned_objects=[]
)


# ============================================================
# EMERGENCY RESPONSE SUMMARY
# ============================================================

if mode == "Emergency Response":

    st.subheader(
        "Emergency Response Intelligence"
    )

    comparison = ROUTE_COMPARISONS.get(
        scenario
    )

    if comparison is not None:

        r1, r2, r3, r4 = st.columns(4)

        with r1:
            st.metric(
                "Normal Response",
                f"{fmt(comparison.get('normal_time_min'), 2)} min"
            )

        with r2:
            st.metric(
                "Scenario Response",
                f"{fmt(comparison.get('scenario_time_min'), 2)} min"
            )

        with r3:
            st.metric(
                "Response Delay",
                f"{fmt(comparison.get('delay_min'), 2)} min"
            )

        with r4:
            st.metric(
                "Affected Route Segments",
                fmt(comparison.get("affected_segments"))
            )


# ============================================================
# SCENARIO COMPARISON
# ============================================================

st.subheader(
    "Scenario Comparison"
)

comparison_columns = [
    "scenario",
    "rainfall_mm_hr",
    "roads_affected",
    "roads_severe",
    "roads_closed",
    "mean_vc",
    "facilities_operationally_affected",
    "population_exposed",
    "population_priority",
    "population_high_priority"
]

available_columns = [
    c for c in comparison_columns
    if c in SUMMARY.columns
]

display_summary = SUMMARY[
    available_columns
].copy()

st.dataframe(
    display_summary,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MODEL NOTES
# ============================================================

with st.expander(
    "Model assumptions and limitations"
):

    st.markdown(
        """
        **URIP is a simulation and decision-support prototype.**

        - Traffic volumes are modelled estimates rather than
          observed real-time traffic counts.
        - Road capacities are modelling assumptions.
        - Population represents estimated daytime population.
        - Facility flood exposure uses point-based exposure.
        - Emergency routing uses the modelled emergency network.
        - Flood-adjusted traffic uses a BPR-style congestion model.
        - Emergency response uses a bounded V/C assumption.
        - Rerouting is a fast network approximation rather than
          a full origin-destination traffic assignment.
        """
    )
