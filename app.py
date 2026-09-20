

import gzip
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium import Element
from streamlit_folium import st_folium
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="URIP | Urban Resilience Intelligence Platform",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_GZ = BASE_DIR / "URIP_MODEL.pkl.gz"
MODEL_PKL = BASE_DIR / "URIP_MODEL.pkl"

ANALYSIS_CRS = "EPSG:32737"
DISPLAY_CRS = "EPSG:4326"

MAP_CENTER = [-1.2864, 36.8172]
MAP_ZOOM = 13

FLOOD_BOUNDS = [
    [-1.3149999999999973, 36.79],
    [-1.2600000000000051, 36.85],
]

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

ROAD_COLORS = {
    "Passable": "#2ca25f",
    "Minor Disruption": "#fdae6b",
    "Severe Disruption": "#e34a33",
    "Closed": "#7f0000",
    "No Data": "#969696",
}

FLOOD_CLASS_COLORS = {
    "Low": "#2ca25f",
    "Moderate": "#fdae6b",
    "High": "#e34a33",
    "Very High": "#7f0000",
}

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
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():
    errors = []

    if MODEL_GZ.exists():
        try:
            with gzip.open(MODEL_GZ, "rb") as f:
                model = pickle.load(f)

            return model, MODEL_GZ.name

        except Exception as e:
            errors.append(f"{MODEL_GZ.name}: {e}")

    if MODEL_PKL.exists():
        try:
            with open(MODEL_PKL, "rb") as f:
                model = pickle.load(f)

            return model, MODEL_PKL.name

        except Exception as e:
            errors.append(f"{MODEL_PKL.name}: {e}")

    raise FileNotFoundError(
        "No usable URIP model package found.\n\n"
        + "\n".join(errors)
    )


try:
    MODEL, MODEL_FILE = load_model()
except Exception as e:
    st.error("Unable to load the frozen URIP model.")
    st.exception(e)
    st.stop()


missing_sections = [
    section for section in EXPECTED_SECTIONS
    if section not in MODEL
]

if missing_sections:
    st.error(
        "The uploaded model is incomplete.\n\n"
        f"Missing sections: {', '.join(missing_sections)}"
    )
    st.stop()


# ============================================================
# GENERIC HELPERS
# ============================================================

def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if value is None:
            return default

        if pd.isna(value):
            return default

        return int(round(float(value)))

    except Exception:
        return default


def fmt_number(value, decimals=0):
    value = safe_float(value)

    if np.isnan(value):
        return "—"

    return f"{value:,.{decimals}f}"


def fmt_minutes(value):
    value = safe_float(value)

    if np.isnan(value):
        return "—"

    return f"{value:,.2f} min"


def fmt_pct(value):
    value = safe_float(value)

    if np.isnan(value):
        return "—"

    return f"{value:,.1f}%"


def get_df(obj):
    """
    Return a DataFrame / GeoDataFrame if obj is one,
    otherwise return None.
    """
    if isinstance(obj, (pd.DataFrame, gpd.GeoDataFrame)):
        return obj

    return None


def scenario_rows(df, scenario):
    """
    Return rows belonging to a scenario.
    """
    if df is None or len(df) == 0:
        return df

    if "scenario" not in df.columns:
        return df.iloc[0:0].copy()

    return df[df["scenario"].astype(str) == str(scenario)].copy()


def first_value(df, columns, default=np.nan):
    """
    Return the first non-null value from a list of possible columns.
    """
    if df is None or len(df) == 0:
        return default

    for column in columns:
        if column in df.columns:
            values = df[column].dropna()

            if len(values):
                return values.iloc[0]

    return default


def sum_value(df, columns, default=np.nan):
    """
    Sum the first matching numeric column.
    """
    if df is None or len(df) == 0:
        return default

    for column in columns:
        if column in df.columns:
            values = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            if values.notna().any():
                return values.sum()

    return default


def recursive_dataframes(obj, prefix=""):
    """
    Recursively discover DataFrames and GeoDataFrames.
    Used only for reading already-frozen outputs.
    """
    found = []

    if isinstance(obj, (pd.DataFrame, gpd.GeoDataFrame)):
        found.append((prefix, obj))
        return found

    if isinstance(obj, dict):
        for key, value in obj.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            found.extend(
                recursive_dataframes(value, child_prefix)
            )

    return found


def find_metric_in_model(
    scenario,
    preferred_columns,
    preferred_sections=None,
    aggregation="first",
):
    """
    Search frozen tables for a scenario-specific metric.

    This is a read-only extraction helper. It does not create
    or recalculate analytical outputs.
    """

    preferred_sections = preferred_sections or []

    candidates = []

    # First search explicitly preferred sections.
    for section in preferred_sections:
        if section not in MODEL:
            continue

        for path, df in recursive_dataframes(
            MODEL[section],
            section
        ):
            candidates.append((path, df))

    # Then search remaining model sections.
    for section, obj in MODEL.items():

        if section in preferred_sections:
            continue

        for path, df in recursive_dataframes(
            obj,
            section
        ):
            candidates.append((path, df))

    for path, df in candidates:

        if "scenario" not in df.columns:
            continue

        sdf = scenario_rows(df, scenario)

        if len(sdf) == 0:
            continue

        for column in preferred_columns:

            if column not in sdf.columns:
                continue

            values = pd.to_numeric(
                sdf[column],
                errors="coerce"
            )

            values = values.dropna()

            if len(values) == 0:
                continue

            if aggregation == "sum":
                return values.sum()

            if aggregation == "mean":
                return values.mean()

            if aggregation == "max":
                return values.max()

            return values.iloc[0]

    return np.nan


# ============================================================
# AUTHORITATIVE EMERGENCY TABLES
# ============================================================

EMERGENCY = MODEL["emergency"]

INCIDENT_REPORT = EMERGENCY["incident_report"].copy()
INCIDENT_COMPARISON = EMERGENCY["incident_comparison"].copy()
SITUATION_REPORT = EMERGENCY["situation_report"].copy()
EMERGENCY_KPIS = EMERGENCY["kpis"].copy()

FACILITY_OPTIONS = MODEL["facility_options"].copy()

ROUTES_ALTERNATIVES = MODEL["routes"]["alternatives"].copy()
ROUTES_RECOMMENDED = MODEL["routes"]["recommended"].copy()


# ============================================================
# SCENARIO ACCESSORS
# ============================================================

def get_emergency_kpi(scenario):

    rows = scenario_rows(
        EMERGENCY_KPIS,
        scenario
    )

    if len(rows) == 0:
        return pd.Series(dtype=object)

    return rows.iloc[0]


def get_situation(scenario):

    rows = scenario_rows(
        SITUATION_REPORT,
        scenario
    )

    if len(rows) == 0:
        return pd.Series(dtype=object)

    return rows.iloc[0]


def get_incident_report(scenario, incident_id):

    rows = INCIDENT_REPORT[
        (INCIDENT_REPORT["scenario"] == scenario)
        &
        (INCIDENT_REPORT["incident_id"] == incident_id)
    ].copy()

    if len(rows) == 0:
        return pd.Series(dtype=object)

    # incident_report is the authoritative one-row-per-
    # incident/scenario recommendation table.
    return rows.iloc[0]


def get_facility_options(scenario, incident_id):

    rows = FACILITY_OPTIONS[
        (FACILITY_OPTIONS["scenario"] == scenario)
        &
        (FACILITY_OPTIONS["incident_id"] == incident_id)
    ].copy()

    if len(rows) == 0:
        return rows

    return rows.sort_values(
        ["facility_rank"]
    )


def get_route_options(
    scenario,
    incident_id,
    facility_rank
):

    rows = ROUTES_ALTERNATIVES[
        (ROUTES_ALTERNATIVES["scenario"] == scenario)
        &
        (ROUTES_ALTERNATIVES["incident_id"] == incident_id)
        &
        (ROUTES_ALTERNATIVES["facility_rank"] == facility_rank)
    ].copy()

    if len(rows) == 0:
        return rows

    return rows.sort_values(
        ["route_rank"]
    )


# ============================================================
# SCENARIO DATA
# ============================================================

def get_roads(scenario):

    roads = MODEL["traffic"].get(scenario)

    if roads is None:
        roads = MODEL["roads"].get(scenario)

    if roads is None:
        return gpd.GeoDataFrame()

    return roads.copy()


def get_facilities(scenario):

    facilities = MODEL["facilities"].get(scenario)

    if facilities is None:
        return gpd.GeoDataFrame()

    return facilities.copy()


def get_incidents():

    incidents = MODEL["incidents"]

    if isinstance(incidents, dict):

        points = incidents.get("points")

        if isinstance(
            points,
            (pd.DataFrame, gpd.GeoDataFrame)
        ):
            return points.copy()

    return gpd.GeoDataFrame()


def get_incident_flood(scenario):

    incidents = MODEL["incidents"]

    if isinstance(incidents, dict):

        flood = incidents.get("flood")

        if isinstance(
            flood,
            (pd.DataFrame, gpd.GeoDataFrame)
        ):

            rows = scenario_rows(
                flood,
                scenario
            )

            return rows

    return gpd.GeoDataFrame()


def get_flood_array(scenario):

    flood = MODEL["flood"]

    if isinstance(flood, dict):

        value = flood.get(scenario)

        if isinstance(value, np.ndarray):
            return value

        if isinstance(value, dict):

            for key in [
                "flood_raster",
                "flood",
                "array",
                "impact",
                "flood_impact",
            ]:

                candidate = value.get(key)

                if isinstance(
                    candidate,
                    np.ndarray
                ):
                    return candidate

    return None


# ============================================================
# DASHBOARD KPIs
# ============================================================

def get_dashboard_kpis(scenario):

    kpi = get_emergency_kpi(scenario)

    situation = get_situation(scenario)

    roads = get_roads(scenario)
    facilities = get_facilities(scenario)

    # --------------------------------------------------------
    # Flood impact
    # --------------------------------------------------------

    flood_impact = first_value(
        pd.DataFrame([kpi]),
        [
            "mean_flood_impact",
        ],
    )

    if np.isnan(safe_float(flood_impact)):
        flood_impact = first_value(
            pd.DataFrame([situation]),
            [
                "mean_incident_flood_impact",
            ],
        )

    # --------------------------------------------------------
    # Roads
    # --------------------------------------------------------

    roads_affected = np.nan
    roads_closed = np.nan
    mean_vc = np.nan

    if len(roads):

        if "affected_pct" in roads.columns:
            affected = pd.to_numeric(
                roads["affected_pct"],
                errors="coerce"
            )

            roads_affected = (
                (affected > 0).sum()
            )

        elif "access_status" in roads.columns:
            roads_affected = (
                roads["access_status"]
                .astype(str)
                .str.lower()
                .ne("passable")
                .sum()
            )

        if "passability" in roads.columns:

            roads_closed = (
                pd.to_numeric(
                    roads["passability"],
                    errors="coerce"
                )
                .eq(0)
                .sum()
            )

        if "final_vc_ratio" in roads.columns:

            vc = pd.to_numeric(
                roads["final_vc_ratio"],
                errors="coerce"
            )

            mean_vc = vc.mean()

    # --------------------------------------------------------
    # Facilities affected
    # --------------------------------------------------------

    facilities_affected = np.nan

    if len(facilities):

        if "operationally_affected" in facilities.columns:

            values = facilities[
                "operationally_affected"
            ]

            if values.dtype == bool:
                facilities_affected = int(values.sum())
            else:
                facilities_affected = int(
                    pd.to_numeric(
                        values,
                        errors="coerce"
                    )
                    .fillna(0)
                    .gt(0)
                    .sum()
                )

    # --------------------------------------------------------
    # CBD population metrics
    #
    # These are read from the frozen model rather than
    # recalculated in Streamlit.
    # --------------------------------------------------------

    population_exposed = find_metric_in_model(
        scenario,
        [
            "population_exposed",
            "flood_exposed_population",
            "exposed_population",
        ],
        preferred_sections=[
            "summary",
            "population",
        ],
        aggregation="first",
    )

    access_disrupted = find_metric_in_model(
        scenario,
        [
            "access_disrupted_population",
            "access_disrupted",
            "population_access_disrupted",
        ],
        preferred_sections=[
            "summary",
            "population",
        ],
        aggregation="first",
    )

    priority_population = find_metric_in_model(
        scenario,
        [
            "priority_population",
        ],
        preferred_sections=[
            "summary",
            "population",
        ],
        aggregation="first",
    )

    high_priority = find_metric_in_model(
        scenario,
        [
            "high_priority_population",
        ],
        preferred_sections=[
            "summary",
            "population",
        ],
        aggregation="first",
    )

    # --------------------------------------------------------
    # Emergency response
    # --------------------------------------------------------

    mean_response = first_value(
        pd.DataFrame([kpi]),
        [
            "mean_response_time_min",
        ],
    )

    if np.isnan(safe_float(mean_response)):
        mean_response = first_value(
            pd.DataFrame([situation]),
            [
                "mean_response_time_min",
            ],
        )

    return {
        "flood_impact": flood_impact,
        "roads_affected": roads_affected,
        "roads_closed": roads_closed,
        "mean_vc": mean_vc,
        "facilities_affected": facilities_affected,
        "population_exposed": population_exposed,
        "access_disrupted": access_disrupted,
        "priority_population": priority_population,
        "high_priority": high_priority,
        "mean_response": mean_response,
    }


# ============================================================
# GEO HELPERS
# ============================================================

def prepare_gdf(gdf):

    if gdf is None:
        return gpd.GeoDataFrame()

    if len(gdf) == 0:
        return gdf.copy()

    result = gdf.copy()

    if not isinstance(
        result,
        gpd.GeoDataFrame
    ):
        if "geometry" in result.columns:
            result = gpd.GeoDataFrame(
                result,
                geometry="geometry"
            )
        else:
            return gpd.GeoDataFrame()

    if result.crs is None:
        result = result.set_crs(
            ANALYSIS_CRS,
            allow_override=True
        )

    try:
        result = result.to_crs(
            DISPLAY_CRS
        )
    except Exception:
        pass

    return result


def add_geojson(
    fmap,
    gdf,
    name,
    style_function,
    tooltip=None,
    show=False,
):

    if gdf is None or len(gdf) == 0:
        return

    gdf = prepare_gdf(gdf)

    if len(gdf) == 0:
        return

    folium.GeoJson(
        gdf.to_json(),
        name=name,
        style_function=style_function,
        tooltip=tooltip,
        show=show,
        smooth_factor=0.5,
    ).add_to(fmap)


# ============================================================
# FLOOD MAP
# ============================================================

def add_flood_raster(fmap, scenario):

    array = get_flood_array(scenario)

    if array is None:
        return

    try:

        array = np.asarray(array)

        if array.ndim != 2:
            return

        finite = np.isfinite(array)

        if not finite.any():
            return

        minimum = np.nanmin(array)
        maximum = np.nanmax(array)

        if maximum > minimum:

            normalized = (
                (array - minimum)
                /
                (maximum - minimum)
            )

        else:
            normalized = np.zeros_like(
                array,
                dtype=float
            )

        # Transparent RGBA flood overlay.
        rgba = np.zeros(
            (
                normalized.shape[0],
                normalized.shape[1],
                4
            ),
            dtype=np.uint8
        )

        rgba[:, :, 0] = 220
        rgba[:, :, 1] = 40
        rgba[:, :, 2] = 40

        alpha = (
            normalized * 190
        ).astype(np.uint8)

        alpha[~finite] = 0

        rgba[:, :, 3] = alpha

        from PIL import Image

        image = Image.fromarray(
            rgba,
            mode="RGBA"
        )

        import io
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="PNG"
        )

        import base64

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        folium.raster_layers.ImageOverlay(
            image=(
                "data:image/png;base64,"
                + encoded
            ),
            bounds=FLOOD_BOUNDS,
            opacity=0.55,
            name=f"{scenario} flood impact",
            interactive=True,
            cross_origin=False,
            zindex=1,
            show=True,
        ).add_to(fmap)

    except Exception:
        return


# ============================================================
# ROAD MAP
# ============================================================

def add_roads_layer(fmap, roads):

    if roads is None or len(roads) == 0:
        return

    roads = prepare_gdf(roads)

    if len(roads) == 0:
        return

    for status, color in ROAD_COLORS.items():

        if "passability" in roads.columns:

            if status == "Passable":
                subset = roads[
                    pd.to_numeric(
                        roads["passability"],
                        errors="coerce"
                    ) >= 0.999
                ]

            elif status == "Closed":
                subset = roads[
                    pd.to_numeric(
                        roads["passability"],
                        errors="coerce"
                    ) <= 0.001
                ]

            elif status == "Minor Disruption":

                if "access_status" in roads.columns:
                    subset = roads[
                        roads["access_status"]
                        .astype(str)
                        .eq("Minor Disruption")
                    ]
                else:
                    subset = roads.iloc[0:0]

            elif status == "Severe Disruption":

                if "access_status" in roads.columns:
                    subset = roads[
                        roads["access_status"]
                        .astype(str)
                        .eq("Severe Disruption")
                    ]
                else:
                    subset = roads.iloc[0:0]

            else:
                subset = roads.iloc[0:0]

        else:
            subset = roads.iloc[0:0]

        if len(subset) == 0:
            continue

        folium.GeoJson(
            subset.to_json(),
            name=f"Roads — {status}",
            style_function=lambda feature,
            color=color: {
                "color": color,
                "weight": 2.5,
                "opacity": 0.85,
            },
            show=(status != "Passable"),
        ).add_to(fmap)


# ============================================================
# FACILITY MAP
# ============================================================

def add_facilities_layer(
    fmap,
    facilities,
    recommended_facility_id=None,
):

    if facilities is None or len(facilities) == 0:
        return

    facilities = prepare_gdf(facilities)

    if len(facilities) == 0:
        return

    for _, row in facilities.iterrows():

        geometry = row.geometry

        if geometry is None:
            continue

        try:
            if geometry.geom_type == "Point":
                lat = geometry.y
                lon = geometry.x
            else:
                point = geometry.centroid
                lat = point.y
                lon = point.x

        except Exception:
            continue

        facility_id = str(
            row.get(
                "facility_id",
                ""
            )
        )

        facility_name = str(
            row.get(
                "name",
                row.get(
                    "facility_name",
                    "Facility"
                )
            )
        )

        facility_type = str(
            row.get(
                "facility_type",
                "Facility"
            )
        )

        operationally_affected = row.get(
            "operationally_affected",
            False
        )

        is_recommended = (
            recommended_facility_id is not None
            and facility_id
            == str(recommended_facility_id)
        )

        if is_recommended:

            color = "#00ffff"
            radius = 11
            fill_opacity = 1.0

        elif bool(operationally_affected):

            color = "#e34a33"
            radius = 6
            fill_opacity = 0.85

        else:

            color = "#3388ff"
            radius = 5
            fill_opacity = 0.75

        popup_html = f"""
        <div style="font-family:Arial;min-width:220px;">
            <h4 style="margin-bottom:6px;">
                {facility_name}
            </h4>

            <b>Facility ID:</b> {facility_id}<br>
            <b>Type:</b> {facility_type}<br>
            <b>Status:</b>
            {"Recommended" if is_recommended else
             ("Operationally affected" if bool(operationally_affected)
              else "Operational")}
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=fill_opacity,
            weight=2,
            popup=folium.Popup(
                popup_html,
                max_width=320
            ),
            tooltip=facility_name,
        ).add_to(fmap)


# ============================================================
# INCIDENT MAP
# ============================================================

def add_incidents_layer(
    fmap,
    incidents,
    selected_incident,
):

    if incidents is None or len(incidents) == 0:
        return

    incidents = prepare_gdf(incidents)

    for _, row in incidents.iterrows():

        geometry = row.geometry

        if geometry is None:
            continue

        try:
            lat = geometry.y
            lon = geometry.x
        except Exception:
            continue

        incident_id = str(
            row.get(
                "incident_id",
                ""
            )
        )

        selected = (
            incident_id
            == str(selected_incident)
        )

        color = (
            "#ff0000"
            if selected
            else "#ffcc00"
        )

        radius = (
            10
            if selected
            else 6
        )

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.95,
            weight=2,
            tooltip=incident_id,
            popup=incident_id,
        ).add_to(fmap)


# ============================================================
# ROUTE DRAWING
# ============================================================

def add_route(
    fmap,
    route_row,
    color="#00ffff",
    weight=7,
    name="Recommended emergency route",
    show=True,
):

    if route_row is None:
        return

    if isinstance(
        route_row,
        pd.DataFrame
    ):

        if len(route_row) == 0:
            return

        route_row = route_row.iloc[0]

    geometry = route_row.get(
        "geometry",
        None
    )

    if geometry is None:
        return

    try:

        route_gdf = gpd.GeoDataFrame(
            [route_row],
            geometry="geometry",
            crs=ANALYSIS_CRS,
        )

        route_gdf = prepare_gdf(
            route_gdf
        )

        folium.GeoJson(
            route_gdf.to_json(),
            name=name,
            style_function=lambda feature: {
                "color": color,
                "weight": weight,
                "opacity": 0.95,
            },
            show=show,
            tooltip=folium.GeoJsonTooltip(
                fields=[
                    field
                    for field in [
                        "route_rank",
                        "response_time_min",
                        "route_length_m",
                        "affected_segments",
                        "closed_segments",
                    ]
                    if field in route_gdf.columns
                ],
                aliases=[
                    alias
                    for field, alias in [
                        (
                            "route_rank",
                            "Route"
                        ),
                        (
                            "response_time_min",
                            "Response time (min)"
                        ),
                        (
                            "route_length_m",
                            "Length (m)"
                        ),
                        (
                            "affected_segments",
                            "Affected segments"
                        ),
                        (
                            "closed_segments",
                            "Closed segments"
                        ),
                    ]
                    if field in route_gdf.columns
                ],
                localize=True,
            ),
        ).add_to(fmap)

    except Exception:
        return


# ============================================================
# BUILD MAP
# ============================================================

def build_map(
    scenario,
    map_mode,
    incident_id,
):

    fmap = folium.Map(
        location=MAP_CENTER,
        zoom_start=MAP_ZOOM,
        control_scale=True,
        tiles=None,
    )

    # Base maps
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        control=True,
    ).add_to(fmap)

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri World Imagery",
        name="Satellite",
        control=True,
    ).add_to(fmap)

    # --------------------------------------------------------
    # Flood layer
    # --------------------------------------------------------

    add_flood_raster(
        fmap,
        scenario
    )

    # Flood boundary
    folium.Rectangle(
        bounds=FLOOD_BOUNDS,
        color="#555555",
        weight=1,
        fill=False,
        name="URIP analysis boundary",
    ).add_to(fmap)

    # --------------------------------------------------------
    # Roads
    # --------------------------------------------------------

    roads = get_roads(scenario)

    if map_mode in [
        "Flood Scenario",
        "Emergency Response",
    ]:

        add_roads_layer(
            fmap,
            roads
        )

    # --------------------------------------------------------
    # Emergency information
    # --------------------------------------------------------

    facilities = get_facilities(
        scenario
    )

    incidents = get_incidents()

    report = get_incident_report(
        scenario,
        incident_id
    )

    recommended_facility_id = None

    if len(report):

        recommended_facility_id = report.get(
            "recommended_facility_id",
            None
        )

    # --------------------------------------------------------
    # Facilities
    # --------------------------------------------------------

    add_facilities_layer(
        fmap,
        facilities,
        recommended_facility_id
    )

    # --------------------------------------------------------
    # Incidents
    # --------------------------------------------------------

    add_incidents_layer(
        fmap,
        incidents,
        incident_id
    )

    # --------------------------------------------------------
    # Selected incident flood point
    # --------------------------------------------------------

    incident_flood = get_incident_flood(
        scenario
    )

    if len(incident_flood):

        selected_flood = incident_flood[
            incident_flood["incident_id"]
            == incident_id
        ].copy()

        selected_flood = prepare_gdf(
            selected_flood
        )

        for _, row in selected_flood.iterrows():

            geometry = row.geometry

            if geometry is None:
                continue

            try:

                lat = geometry.y
                lon = geometry.x

                impact = safe_float(
                    row.get(
                        "flood_impact",
                        np.nan
                    )
                )

                flood_class = str(
                    row.get(
                        "flood_class",
                        "Unknown"
                    )
                )

                color = FLOOD_CLASS_COLORS.get(
                    flood_class,
                    "#555555"
                )

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=13,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.25,
                    weight=3,
                    tooltip=(
                        f"{incident_id} — "
                        f"{flood_class}"
                    ),
                    popup=(
                        f"<b>{incident_id}</b><br>"
                        f"Flood impact: "
                        f"{fmt_number(impact, 3)}<br>"
                        f"Class: {flood_class}"
                    ),
                ).add_to(fmap)

            except Exception:
                pass

    # --------------------------------------------------------
    # Emergency route
    # --------------------------------------------------------

    if map_mode == "Emergency Response":

        if len(report):

            facility_rank = safe_int(
                report.get(
                    "facility_rank",
                    1
                ),
                1
            )

            route_rank = safe_int(
                report.get(
                    "route_rank",
                    1
                ),
                1
            )

            route_rows = ROUTES_ALTERNATIVES[
                (ROUTES_ALTERNATIVES["scenario"] == scenario)
                &
                (ROUTES_ALTERNATIVES["incident_id"] == incident_id)
                &
                (
                    ROUTES_ALTERNATIVES["facility_rank"]
                    == facility_rank
                )
                &
                (
                    ROUTES_ALTERNATIVES["route_rank"]
                    == route_rank
                )
            ]

            if len(route_rows):

                add_route(
                    fmap,
                    route_rows.iloc[0],
                    color="#00ffff",
                    weight=8,
                    name=(
                        "Recommended emergency route"
                    ),
                    show=True,
                )

    folium.LayerControl(
        collapsed=False
    ).add_to(fmap)

    return fmap


# ============================================================
# UI STYLES
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0;
        letter-spacing: -0.03em;
    }

    .subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-top: 0.2rem;
        margin-bottom: 1.2rem;
    }

    .status-card {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        background: #f8fafc;
        margin-bottom: 1rem;
    }

    .status-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        font-weight: 700;
    }

    .status-value {
        font-size: 1.15rem;
        font-weight: 750;
        margin-top: 0.2rem;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 750;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    .incident-header {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        background: #0f172a;
        color: white;
        margin-bottom: 0.8rem;
    }

    .facility-card {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.9rem;
        margin-bottom: 0.5rem;
        background: white;
    }

    .facility-rank {
        font-size: 0.75rem;
        font-weight: 800;
        text-transform: uppercase;
        color: #64748b;
    }

    .facility-name {
        font-size: 1rem;
        font-weight: 750;
    }

    .route-row {
        border-left: 3px solid #cbd5e1;
        padding: 0.5rem 0.8rem;
        margin-left: 0.8rem;
        margin-bottom: 0.35rem;
        background: #f8fafc;
        border-radius: 0 7px 7px 0;
        font-size: 0.88rem;
    }

    .recommended-route {
        border-left: 4px solid #06b6d4;
        background: #ecfeff;
    }

    .model-note {
        color: #64748b;
        font-size: 0.8rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 1.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌍 URIP</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Urban Resilience Intelligence Platform —
        Flood, traffic and emergency-response intelligence
        for Nairobi CBD.
    </div>
    """,
    unsafe_allow_html=True,
)

st.success(
    "Complete frozen URIP dashboard package detected."
)


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

with st.sidebar:

    st.header("Scenario Controls")

    scenario = st.selectbox(
        "Rainfall scenario",
        SCENARIOS,
        index=0,
    )

    rainfall = SCENARIO_RAINFALL[
        scenario
    ]

    map_mode = st.radio(
        "Map mode",
        [
            "Flood Scenario",
            "Emergency Response",
        ],
        index=0,
    )

    incident_ids = sorted(
        INCIDENT_REPORT[
            INCIDENT_REPORT["scenario"] == scenario
        ]["incident_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if len(incident_ids) == 0:

        incident_ids = sorted(
            INCIDENT_REPORT[
                "incident_id"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    incident_id = st.selectbox(
        "Emergency incident",
        incident_ids,
        index=0,
    )

    st.divider()

    st.caption(
        f"Rainfall intensity: {rainfall} mm/hr"
    )

    st.caption(
        f"Model file: {MODEL_FILE}"
    )


# ============================================================
# GET CURRENT DATA
# ============================================================

kpis = get_dashboard_kpis(
    scenario
)

kpi_row = get_emergency_kpi(
    scenario
)

situation = get_situation(
    scenario
)

incident_report = get_incident_report(
    scenario,
    incident_id
)

facility_options = get_facility_options(
    scenario,
    incident_id
)


# ============================================================
# KPI STRIP
# ============================================================

st.markdown(
    '<div class="section-title">Scenario Intelligence</div>',
    unsafe_allow_html=True,
)

kpi_cols = st.columns(5)

with kpi_cols[0]:
    st.metric(
        "Flood Impact",
        fmt_number(
            kpis["flood_impact"],
            3
        ),
    )

with kpi_cols[1]:
    st.metric(
        "Roads Affected",
        fmt_number(
            kpis["roads_affected"]
        ),
    )

with kpi_cols[2]:
    st.metric(
        "Roads Closed",
        fmt_number(
            kpis["roads_closed"]
        ),
    )

with kpi_cols[3]:
    st.metric(
        "Mean V/C",
        fmt_number(
            kpis["mean_vc"],
            3
        ),
    )

with kpi_cols[4]:
    st.metric(
        "Facilities Affected",
        fmt_number(
            kpis["facilities_affected"]
        ),
    )


kpi_cols2 = st.columns(5)

with kpi_cols2[0]:
    st.metric(
        "Population Exposed",
        fmt_number(
            kpis["population_exposed"]
        ),
    )

with kpi_cols2[1]:
    st.metric(
        "Access Disrupted",
        fmt_number(
            kpis["access_disrupted"]
        ),
    )

with kpi_cols2[2]:
    st.metric(
        "Priority Population",
        fmt_number(
            kpis["priority_population"]
        ),
    )

with kpi_cols2[3]:
    st.metric(
        "High Priority",
        fmt_number(
            kpis["high_priority"]
        ),
    )

with kpi_cols2[4]:
    st.metric(
        "Mean Response Time",
        fmt_minutes(
            kpis["mean_response"]
        ),
    )


# ============================================================
# EMERGENCY STATUS
# ============================================================

st.markdown(
    '<div class="section-title">Emergency Situation</div>',
    unsafe_allow_html=True,
)

status_cols = st.columns(4)

with status_cols[0]:

    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-title">
                Situation status
            </div>
            <div class="status-value">
                {situation.get("situation_status", "—")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with status_cols[1]:

    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-title">
                Response network
            </div>
            <div class="status-value">
                {situation.get("response_network_status", "—")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with status_cols[2]:

    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-title">
                Facility status
            </div>
            <div class="status-value">
                {situation.get("facility_status", "—")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with status_cols[3]:

    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-title">
                Route availability
            </div>
            <div class="status-value">
                {fmt_pct(
                    kpi_row.get(
                        "route_availability_pct",
                        np.nan
                    )
                )}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if len(kpi_row):

    alert = kpi_row.get(
        "emergency_alert",
        None
    )

    if (
        alert is not None
        and not pd.isna(alert)
    ):
        if scenario == "Normal":
            st.info(str(alert))
        else:
            st.warning(str(alert))


# ============================================================
# MAIN MAP
# ============================================================

st.markdown(
    '<div class="section-title">Spatial Intelligence</div>',
    unsafe_allow_html=True,
)

fmap = build_map(
    scenario,
    map_mode,
    incident_id,
)

st_folium(
    fmap,
    width=None,
    height=650,
    returned_objects=[],
    use_container_width=True,
)


# ============================================================
# INCIDENT INTELLIGENCE
# ============================================================

st.markdown(
    '<div class="section-title">Incident Intelligence</div>',
    unsafe_allow_html=True,
)

if len(incident_report):

    incident_cols = st.columns(4)

    with incident_cols[0]:
        st.metric(
            "Incident flood impact",
            fmt_number(
                incident_report.get(
                    "incident_flood_impact",
                    np.nan
                ),
                3,
            ),
        )

    with incident_cols[1]:
        st.metric(
            "Flood class",
            str(
                incident_report.get(
                    "incident_flood_class",
                    "—"
                )
            ),
        )

    with incident_cols[2]:
        st.metric(
            "Catchment population",
            fmt_number(
                incident_report.get(
                    "population_in_catchment",
                    np.nan
                )
            ),
        )

    with incident_cols[3]:
        st.metric(
            "Priority population",
            fmt_number(
                incident_report.get(
                    "priority_population",
                    np.nan
                )
            ),
        )

    # --------------------------------------------------------
    # Recommended facility
    # --------------------------------------------------------

    recommended_facility = incident_report.get(
        "recommended_facility_name",
        None
    )

    recommended_facility_id = incident_report.get(
        "recommended_facility_id",
        None
    )

    recommended_response = incident_report.get(
        "recommended_facility_response_time_min",
        np.nan
    )

    recommended_route_time = incident_report.get(
        "recommended_route_response_time_min",
        np.nan
    )

    st.markdown(
        f"""
        <div class="incident-header">
            <b>{incident_id}</b>
            &nbsp; • &nbsp;
            {scenario}
            &nbsp; • &nbsp;
            Recommended facility:
            <b>{recommended_facility or "—"}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rec_cols = st.columns(4)

    with rec_cols[0]:
        st.metric(
            "Facility",
            str(
                recommended_facility
                or "—"
            ),
        )

    with rec_cols[1]:
        st.metric(
            "Facility response",
            fmt_minutes(
                recommended_response
            ),
        )

    with rec_cols[2]:
        st.metric(
            "Route response",
            fmt_minutes(
                recommended_route_time
            ),
        )

    with rec_cols[3]:
        st.metric(
            "Route length",
            (
                fmt_number(
                    incident_report.get(
                        "recommended_route_length_m",
                        np.nan
                    )
                )
                + " m"
                if not np.isnan(
                    safe_float(
                        incident_report.get(
                            "recommended_route_length_m",
                            np.nan
                        )
                    )
                )
                else "—"
            ),
        )


# ============================================================
# FACILITY OPTIONS + ROUTE ALTERNATIVES
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <div class="section-title">Facility options and route alternatives</div>
        <div class="section-subtitle">
            Ranked emergency facilities and available response routes for the selected incident
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Selected incident report
# ------------------------------------------------------------

incident_row = get_incident_report(
    selected_scenario,
    selected_incident
)

if incident_row is None or incident_row.empty:

    st.info("No emergency intelligence is available for this incident.")

else:

    incident = incident_row.iloc[0]

    # --------------------------------------------------------
    # Recommended facility / route
    # --------------------------------------------------------

    recommended_facility_id = incident.get("recommended_facility_id")
    recommended_facility_rank = incident.get("facility_rank", 1)
    recommended_route_rank = incident.get("route_rank", 1)

    # --------------------------------------------------------
    # Facility options
    # --------------------------------------------------------

    facility_options = get_facility_options(
        selected_scenario,
        selected_incident
    )

    if facility_options is None or facility_options.empty:

        st.info("No facility options are available for this incident.")

    else:

        facility_options = facility_options.copy()

        # ----------------------------------------------------
        # Sort by authoritative frozen facility rank
        # ----------------------------------------------------

        if "facility_rank" in facility_options.columns:
            facility_options = facility_options.sort_values(
                "facility_rank"
            )

        # ----------------------------------------------------
        # Display each facility
        # ----------------------------------------------------

        for _, facility in facility_options.iterrows():

            facility_rank = int(
                facility.get("facility_rank", 0)
            )

            facility_id = str(
                facility.get("facility_id", "")
            )

            facility_name = str(
                facility.get("facility_name", "Unknown facility")
            )

            facility_type = str(
                facility.get("facility_type", "Facility")
            )

            response_time = float(
                facility.get("travel_time_min", 0)
            )

            normal_time = float(
                facility.get("normal_travel_time_min", 0)
            )

            response_delay = float(
                facility.get("response_delay_min", 0)
            )

            response_delay_pct = float(
                facility.get("response_delay_pct", 0)
            )

            reachable = facility.get(
                "reachable",
                True
            )

            is_recommended = (
                facility_id == recommended_facility_id
            )

            # ------------------------------------------------
            # Facility card styling
            # ------------------------------------------------

            if is_recommended:

                card_border = "#16a34a"
                card_background = "#f0fdf4"
                badge_background = "#16a34a"
                badge_text = "RECOMMENDED"

            else:

                card_border = "#cbd5e1"
                card_background = "#ffffff"
                badge_background = "#64748b"
                badge_text = f"OPTION {facility_rank}"

            reachable_text = (
                "Yes"
                if bool(reachable)
                else "No"
            )

            reachable_color = (
                "#16a34a"
                if bool(reachable)
                else "#dc2626"
            )

            # ------------------------------------------------
            # Facility card
            # ------------------------------------------------

            st.markdown(
                f"""
                <div style="
                    border:1px solid {card_border};
                    border-radius:12px;
                    background:{card_background};
                    padding:16px;
                    margin-top:14px;
                    margin-bottom:8px;
                ">

                    <div style="
                        display:flex;
                        justify-content:space-between;
                        align-items:center;
                        margin-bottom:8px;
                    ">

                        <div style="
                            font-size:12px;
                            font-weight:700;
                            color:white;
                            background:{badge_background};
                            padding:4px 9px;
                            border-radius:999px;
                            letter-spacing:0.3px;
                        ">
                            {badge_text}
                        </div>

                        <div style="
                            font-size:12px;
                            color:#64748b;
                            font-weight:600;
                        ">
                            Facility rank {facility_rank}
                        </div>

                    </div>

                    <div style="
                        font-size:18px;
                        font-weight:700;
                        color:#0f172a;
                        margin-bottom:3px;
                    ">
                        {facility_name}
                    </div>

                    <div style="
                        color:#64748b;
                        font-size:13px;
                    ">
                        {facility_type}
                        &nbsp; • &nbsp;
                        {facility_id}
                    </div>

                    <div style="
                        display:flex;
                        flex-wrap:wrap;
                        gap:22px;
                        margin-top:13px;
                        padding-top:10px;
                        border-top:1px solid #e2e8f0;
                    ">

                        <div>
                            <div style="
                                font-size:11px;
                                color:#64748b;
                            ">
                                RESPONSE
                            </div>
                            <div style="
                                font-size:17px;
                                font-weight:700;
                                color:#0f172a;
                            ">
                                {response_time:.2f} min
                            </div>
                        </div>

                        <div>
                            <div style="
                                font-size:11px;
                                color:#64748b;
                            ">
                                NORMAL
                            </div>
                            <div style="
                                font-size:17px;
                                font-weight:700;
                                color:#0f172a;
                            ">
                                {normal_time:.2f} min
                            </div>
                        </div>

                        <div>
                            <div style="
                                font-size:11px;
                                color:#64748b;
                            ">
                                DELAY
                            </div>
                            <div style="
                                font-size:17px;
                                font-weight:700;
                                color:#dc2626;
                            ">
                                +{response_delay:.2f} min
                            </div>
                        </div>

                        <div>
                            <div style="
                                font-size:11px;
                                color:#64748b;
                            ">
                                REACHABLE
                            </div>
                            <div style="
                                font-size:17px;
                                font-weight:700;
                                color:{reachable_color};
                            ">
                                {reachable_text}
                            </div>
                        </div>

                    </div>

                    <div style="
                        margin-top:8px;
                        font-size:12px;
                        color:#64748b;
                    ">
                        Delay relative to normal:
                        <b>{response_delay_pct:,.1f}%</b>
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            # ------------------------------------------------
            # Route alternatives for this facility
            # ------------------------------------------------

            routes_for_facility = get_route_options(
                selected_scenario,
                selected_incident,
                facility_rank=facility_rank
            )

            if (
                routes_for_facility is None
                or routes_for_facility.empty
            ):
                st.caption(
                    "No route alternatives available for this facility."
                )
                continue

            routes_for_facility = routes_for_facility.copy()

            if "route_rank" in routes_for_facility.columns:
                routes_for_facility = routes_for_facility.sort_values(
                    "route_rank"
                )

            st.markdown(
                """
                <div style="
                    margin-left:20px;
                    margin-top:8px;
                    margin-bottom:4px;
                    font-size:13px;
                    font-weight:700;
                    color:#334155;
                ">
                    Available response routes
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ------------------------------------------------
            # Route cards
            # ------------------------------------------------

            for _, route in routes_for_facility.iterrows():

                route_rank = int(
                    route.get("route_rank", 0)
                )

                route_response = float(
                    route.get("response_time_min", 0)
                )

                route_normal = float(
                    route.get("normal_travel_time_min", 0)
                )

                route_delay = float(
                    route.get("response_delay_min", 0)
                )

                route_delay_pct = float(
                    route.get("response_delay_pct", 0)
                )

                route_length = float(
                    route.get("route_length_m", 0)
                )

                affected_segments = int(
                    route.get("affected_segments", 0)
                )

                closed_segments = int(
                    route.get("closed_segments", 0)
                )

                route_is_recommended = (
                    is_recommended
                    and route_rank == recommended_route_rank
                )

                # --------------------------------------------
                # Route styling
                # --------------------------------------------

                if route_is_recommended:

                    route_border = "#22c55e"
                    route_background = "#f7fee7"
                    route_badge = "RECOMMENDED"
                    route_badge_color = "#16a34a"

                else:

                    route_border = "#e2e8f0"
                    route_background = "#ffffff"
                    route_badge = f"ROUTE {route_rank}"
                    route_badge_color = "#64748b"

                # --------------------------------------------
                # Route card
                # --------------------------------------------

                st.markdown(
                    f"""
                    <div style="
                        margin-left:20px;
                        border-left:3px solid {route_border};
                        border-top:1px solid #e2e8f0;
                        border-right:1px solid #e2e8f0;
                        border-bottom:1px solid #e2e8f0;
                        border-radius:8px;
                        background:{route_background};
                        padding:12px 15px;
                        margin-bottom:8px;
                    ">

                        <div style="
                            display:flex;
                            justify-content:space-between;
                            align-items:center;
                            margin-bottom:8px;
                        ">

                            <div style="
                                font-size:12px;
                                font-weight:700;
                                color:white;
                                background:{route_badge_color};
                                padding:3px 8px;
                                border-radius:999px;
                            ">
                                {route_badge}
                            </div>

                            <div style="
                                font-size:12px;
                                color:#64748b;
                            ">
                                Route {route_rank}
                            </div>

                        </div>

                        <div style="
                            display:flex;
                            flex-wrap:wrap;
                            gap:18px;
                            font-size:13px;
                            color:#334155;
                        ">

                            <div>
                                <span style="color:#64748b;">
                                    Response
                                </span><br>
                                <b>{route_response:.2f} min</b>
                            </div>

                            <div>
                                <span style="color:#64748b;">
                                    Length
                                </span><br>
                                <b>{route_length:,.0f} m</b>
                            </div>

                            <div>
                                <span style="color:#64748b;">
                                    Affected segments
                                </span><br>
                                <b>{affected_segments}</b>
                            </div>

                            <div>
                                <span style="color:#64748b;">
                                    Closed segments
                                </span><br>
                                <b>{closed_segments}</b>
                            </div>

                            <div>
                                <span style="color:#64748b;">
                                    Delay
                                </span><br>
                                <b>+{route_delay:.2f} min</b>
                            </div>

                        </div>

                        <div style="
                            margin-top:8px;
                            font-size:12px;
                            color:#64748b;
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

else:

    st.info(
        "Select an incident to view facility and route intelligence."
    )


# ============================================================
# SCENARIO SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">Scenario status</div>',
    unsafe_allow_html=True,
)

summary_cols = st.columns(4)

for index, current_scenario in enumerate(
    SCENARIOS
):

    row = get_emergency_kpi(
        current_scenario
    )

    with summary_cols[index]:

        st.markdown(
            f"""
            <div class="status-card">
                <div class="status-title">
                    {current_scenario}
                </div>

                <div class="status-value">
                    {row.get(
                        "situation_status",
                        "—"
                    )}
                </div>

                <div style="margin-top:8px;color:#64748b;">
                    Flood:
                    {fmt_number(
                        row.get(
                            "mean_flood_impact",
                            np.nan
                        ),
                        3
                    )}
                    <br>

                    Incidents:
                    {safe_int(
                        row.get(
                            "high_very_high_incidents",
                            np.nan
                        ),
                        0
                    )}
                    high / very high
                    <br>

                    Response:
                    {fmt_minutes(
                        row.get(
                            "mean_response_time_min",
                            np.nan
                        )
                    )}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# MODEL NOTE
# ============================================================

st.markdown(
    """
    <div class="model-note">
        <b>URIP frozen-model dashboard.</b>
        All flood, road disruption, passability, traffic,
        rerouting, facility, population and emergency-response
        outputs shown here are read from the frozen analytical
        package. The Streamlit interface does not recalculate
        the analytical model.
        <br><br>
        Emergency routing represents modelled network routing.
        Route alternatives shown are the frozen alternatives
        generated by the URIP routing model. Affected route
        segments indicate flood-affected segments and do not
        necessarily mean that the segment is closed.
    </div>
    """,
    unsafe_allow_html=True,
)
        
