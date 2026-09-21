

import base64
import gzip
import io
import pickle
import textwrap
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_folium import st_folium

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="URIP | Urban Resilience Intelligence Platform",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
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

SCENARIOS = ["Normal", "Moderate Rainfall", "Heavy Rainfall", "Severe Rainfall"]

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
    "metadata", "scenarios", "flood", "roads", "facilities", "population",
    "incidents", "facility_options", "routes", "traffic", "emergency", "summary",
]

ROUTE_TOOLTIP_FIELDS = [
    ("route_rank", "Route"),
    ("response_time_min", "Response time (min)"),
    ("route_length_m", "Length (m)"),
    ("affected_segments", "Affected segments"),
    ("closed_segments", "Closed segments"),
]


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():
    errors = []

    for path, opener in [(MODEL_GZ, gzip.open), (MODEL_PKL, open)]:
        if not path.exists():
            continue
        try:
            with opener(path, "rb") as f:
                return pickle.load(f), path.name
        except Exception as e:
            errors.append(f"{path.name}: {e}")

    raise FileNotFoundError(
        "No usable URIP model package found.\n\n" + "\n".join(errors)
    )


try:
    MODEL, MODEL_FILE = load_model()
except Exception as e:
    st.error("Unable to load the frozen URIP model.")
    st.exception(e)
    st.stop()

# ============================================================
# FROZEN MODEL SCENARIO RESOLVER
# ============================================================

def resolve_model_scenario(scenario):
    """
    Convert the dashboard's display scenario name to the exact
    scenario key used by the frozen analytical model.

    Dashboard labels:
        Normal
        Moderate Rainfall
        Heavy Rainfall
        Severe Rainfall

    Frozen model keys may be:
        Normal
        Moderate
        Heavy
        Severe
    """

    # Collect scenario keys actually present in the frozen model.
    model_keys = set()

    for section in ["traffic", "facilities", "flood", "roads"]:
        obj = MODEL.get(section)

        if isinstance(obj, dict):
            model_keys.update(str(k) for k in obj.keys())

    # Also inspect authoritative emergency tables.
    for table in [
        MODEL.get("emergency", {}).get("kpis"),
        MODEL.get("emergency", {}).get("situation_report"),
        MODEL.get("emergency", {}).get("incident_report"),
    ]:
        if isinstance(table, (pd.DataFrame, gpd.GeoDataFrame)):
            if "scenario" in table.columns:
                model_keys.update(
                    table["scenario"]
                    .dropna()
                    .astype(str)
                    .unique()
                )

    scenario = str(scenario)

    # Exact match first.
    if scenario in model_keys:
        return scenario

    # Display-label -> frozen-model-key mapping.
    aliases = {
        "Moderate Rainfall": "Moderate",
        "Heavy Rainfall": "Heavy",
        "Severe Rainfall": "Severe",
    }

    candidate = aliases.get(scenario)

    if candidate in model_keys:
        return candidate

    # Fallback: remove the display suffix.
    candidate = scenario.replace(" Rainfall", "").strip()

    if candidate in model_keys:
        return candidate

    # Last resort: return original value.
    # This preserves compatibility if the frozen model actually
    # uses the longer display names.
    return scenario


# ============================================================
# GENERIC HELPERS
# ============================================================

def safe_float(value, default=np.nan):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if value is None or pd.isna(value):
            return default
        return int(round(float(value)))
    except Exception:
        return default


def fmt_number(value, decimals=0):
    value = safe_float(value)
    return "—" if np.isnan(value) else f"{value:,.{decimals}f}"


def fmt_minutes(value):
    value = safe_float(value)
    return "—" if np.isnan(value) else f"{value:,.2f} min"


def fmt_pct(value):
    value = safe_float(value)
    return "—" if np.isnan(value) else f"{value:,.1f}%"


def render_html(html, target=None):
    """
    st.markdown(unsafe_allow_html=True) still runs input through a
    Markdown/CommonMark parser first. Two things break raw HTML built
    from indented f-strings:
      1. Leading whitespace on lines (4+ spaces) can be read as an
         indented code block.
      2. Any line that ends up blank or whitespace-only (e.g. a
         conditional placeholder like {"" if not x else "badge"} sitting
         alone on its own line) terminates the HTML block early, so
         everything after it gets re-parsed as plain text/code.
    Dedent to fix (1) and drop any resulting blank lines to fix (2).
    """
    dedented = textwrap.dedent(html).strip()
    collapsed = "\n".join(line for line in dedented.splitlines() if line.strip() != "")
    (target or st).markdown(collapsed, unsafe_allow_html=True)


def get_df(obj):
    """Return a DataFrame / GeoDataFrame if obj is one, otherwise None."""
    return obj if isinstance(obj, (pd.DataFrame, gpd.GeoDataFrame)) else None


def scenario_rows(df, scenario):
    """Return rows belonging to the exact frozen-model scenario."""

    if df is None or len(df) == 0:
        return df

    if "scenario" not in df.columns:
        return df.iloc[0:0].copy()

    model_scenario = resolve_model_scenario(scenario)

    return df[
        df["scenario"].astype(str) == str(model_scenario)
    ].copy()


def first_value(df, columns, default=np.nan):
    """Return the first non-null value from a list of possible columns."""
    if df is None or len(df) == 0:
        return default
    for column in columns:
        if column in df.columns:
            values = df[column].dropna()
            if len(values):
                return values.iloc[0]
    return default


def sum_value(df, columns, default=np.nan):
    """Sum the first matching numeric column."""
    if df is None or len(df) == 0:
        return default
    for column in columns:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            if values.notna().any():
                return values.sum()
    return default


def recursive_dataframes(obj, prefix=""):
    """Recursively discover DataFrames and GeoDataFrames (read-only)."""
    if isinstance(obj, (pd.DataFrame, gpd.GeoDataFrame)):
        return [(prefix, obj)]

    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            found.extend(recursive_dataframes(value, child_prefix))
    return found


def find_metric_in_model(scenario, preferred_columns, preferred_sections=None, aggregation="first"):
    """
    Search frozen tables for a scenario-specific metric.
    This is a read-only extraction helper; it does not create or
    recalculate analytical outputs.
    """
    preferred_sections = preferred_sections or []
    candidates = []

    # Preferred sections first, then everything else.
    for section in preferred_sections:
        if section in MODEL:
            candidates.extend(recursive_dataframes(MODEL[section], section))

    for section, obj in MODEL.items():
        if section not in preferred_sections:
            candidates.extend(recursive_dataframes(obj, section))

    for _, df in candidates:
        if "scenario" not in df.columns:
            continue

        sdf = scenario_rows(df, scenario)
        if len(sdf) == 0:
            continue

        for column in preferred_columns:
            if column not in sdf.columns:
                continue

            values = pd.to_numeric(sdf[column], errors="coerce").dropna()
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
    rows = scenario_rows(EMERGENCY_KPIS, scenario)
    return rows.iloc[0] if len(rows) else pd.Series(dtype=object)


def get_situation(scenario):
    rows = scenario_rows(SITUATION_REPORT, scenario)
    return rows.iloc[0] if len(rows) else pd.Series(dtype=object)


def get_incident_report(scenario, incident_id):
    rows = INCIDENT_REPORT[
        (INCIDENT_REPORT["scenario"].astype(str)
         == str(resolve_model_scenario(scenario)))
        & (
            INCIDENT_REPORT["incident_id"].astype(str)
            == str(incident_id)
        )
    ]

    return (
        rows.iloc[0]
        if len(rows)
        else pd.Series(dtype=object)
    )


def get_facility_options(scenario, incident_id):

    model_scenario = resolve_model_scenario(
        scenario
    )

    rows = FACILITY_OPTIONS[
        (
            FACILITY_OPTIONS["scenario"].astype(str)
            == str(model_scenario)
        )
        & (
            FACILITY_OPTIONS["incident_id"].astype(str)
            == str(incident_id)
        )
    ]

    return (
        rows.sort_values(["facility_rank"])
        if len(rows)
        else rows
    )


def get_route_options(
    scenario,
    incident_id,
    facility_rank
):

    model_scenario = resolve_model_scenario(
        scenario
    )

    rows = ROUTES_ALTERNATIVES[
        (
            ROUTES_ALTERNATIVES["scenario"].astype(str)
            == str(model_scenario)
        )
        & (
            ROUTES_ALTERNATIVES["incident_id"].astype(str)
            == str(incident_id)
        )
        & (
            ROUTES_ALTERNATIVES["facility_rank"]
            == facility_rank
        )
    ]

    return (
        rows.sort_values(["route_rank"])
        if len(rows)
        else rows
    )


# ============================================================
# SCENARIO DATA
# ============================================================

def get_roads(scenario):

    model_scenario = resolve_model_scenario(scenario)

    roads = MODEL["traffic"].get(model_scenario)

    if roads is None:
        roads = MODEL["roads"].get(model_scenario)

    return (
        roads.copy()
        if roads is not None
        else gpd.GeoDataFrame()
    )


def get_facilities(scenario):

    model_scenario = resolve_model_scenario(scenario)

    facilities = MODEL["facilities"].get(model_scenario)

    return (
        facilities.copy()
        if facilities is not None
        else gpd.GeoDataFrame()
    )


def get_incidents():
    incidents = MODEL["incidents"]
    if isinstance(incidents, dict):
        points = incidents.get("points")
        if isinstance(points, (pd.DataFrame, gpd.GeoDataFrame)):
            return points.copy()
    return gpd.GeoDataFrame()


def get_incident_flood(scenario):
    incidents = MODEL["incidents"]
    if isinstance(incidents, dict):
        flood = incidents.get("flood")
        if isinstance(flood, (pd.DataFrame, gpd.GeoDataFrame)):
            return scenario_rows(flood, scenario)
    return gpd.GeoDataFrame()


def get_flood_array(scenario):

    flood = MODEL["flood"]

    if not isinstance(flood, dict):
        return None

    model_scenario = resolve_model_scenario(scenario)

    value = flood.get(model_scenario)
    if isinstance(value, np.ndarray):
        return value

    if isinstance(value, dict):
        for key in ["flood_raster", "flood", "array", "impact", "flood_impact"]:
            candidate = value.get(key)
            if isinstance(candidate, np.ndarray):
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
        ["mean_flood_impact"]
    )

    if np.isnan(safe_float(flood_impact)):
        flood_impact = first_value(
            pd.DataFrame([situation]),
            ["mean_incident_flood_impact"]
        )

    # --------------------------------------------------------
    # Roads
    # --------------------------------------------------------

    roads_affected = np.nan
    roads_closed = np.nan
    mean_vc = np.nan

    if roads is not None and not roads.empty:

        # Roads affected
        if "affected_pct" in roads.columns:

            affected = pd.to_numeric(
                roads["affected_pct"],
                errors="coerce"
            )

            roads_affected = int(
                (affected.fillna(0) > 0).sum()
            )

        elif "access_status" in roads.columns:

            access = (
                roads["access_status"]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            roads_affected = int(
                (~access.eq("passable")).sum()
            )

        # Roads closed
        if "passability" in roads.columns:

            passability = pd.to_numeric(
                roads["passability"],
                errors="coerce"
            )

            roads_closed = int(
                passability.eq(0).sum()
            )

        # Mean final V/C
        if "final_vc_ratio" in roads.columns:

            vc_values = pd.to_numeric(
                roads["final_vc_ratio"],
                errors="coerce"
            )

            vc_values = (
                vc_values
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
                .dropna()
            )

            if not vc_values.empty:
                mean_vc = float(
                    vc_values.mean()
                )

       # --------------------------------------------------------
    # Facilities affected
    # --------------------------------------------------------

    facilities_affected = np.nan

    if facilities is not None and not facilities.empty:

        if "operationally_affected" in facilities.columns:

            affected_flags = facilities[
                "operationally_affected"
            ].apply(is_true_value)

            facilities_affected = int(
                affected_flags.sum()
            )

        elif "response_disrupted" in facilities.columns:

            affected_flags = facilities[
                "response_disrupted"
            ].apply(is_true_value)

            facilities_affected = int(
                affected_flags.sum()
            )
    # --------------------------------------------------------
    # Emergency population metrics
    # --------------------------------------------------------
    # These values are authoritative in the frozen emergency KPI
    # table. Do not recalculate them from population layers.

    population_exposed = first_value(
        pd.DataFrame([kpi]),
        ["flood_exposed_incident_population"]
    )

    access_disrupted = first_value(
        pd.DataFrame([kpi]),
        ["access_disrupted_incident_population"]
    )

    priority_population = first_value(
        pd.DataFrame([kpi]),
        ["priority_incident_population"]
    )

    high_priority = first_value(
        pd.DataFrame([kpi]),
        ["high_priority_incident_population"]
    )

    # --------------------------------------------------------
    # Emergency response
    # --------------------------------------------------------

    mean_response = first_value(
        pd.DataFrame([kpi]),
        ["mean_response_time_min"]
    )

    if np.isnan(safe_float(mean_response)):

        mean_response = first_value(
            pd.DataFrame([situation]),
            ["mean_response_time_min"]
        )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

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
    if not isinstance(result, gpd.GeoDataFrame):
        if "geometry" not in result.columns:
            return gpd.GeoDataFrame()
        result = gpd.GeoDataFrame(result, geometry="geometry")

    if result.crs is None:
        result = result.set_crs(ANALYSIS_CRS, allow_override=True)

    try:
        result = result.to_crs(DISPLAY_CRS)
    except Exception:
        pass

    return result


def add_geojson(fmap, gdf, name, style_function, tooltip=None, show=False):
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
    """
    Display the frozen-model flood impact as a classified
    continuous raster.

    The analytical values are not recalculated. Only the
    visualization is classified for readability.
    """

    array = get_flood_array(scenario)

    if array is None:
        return

    try:
        array = np.asarray(array, dtype=float)

        if array.ndim != 2:
            return

        finite = np.isfinite(array)

        if not finite.any():
            return

        valid = array[finite]

        minimum = float(np.nanmin(valid))
        maximum = float(np.nanmax(valid))

        # ----------------------------------------------------
        # Create normalized flood-impact surface
        # ----------------------------------------------------

        if maximum > minimum:
            normalized = (array - minimum) / (maximum - minimum)
        else:
            normalized = np.zeros_like(array, dtype=float)

        normalized = np.clip(normalized, 0, 1)

        # ----------------------------------------------------
        # URIP flood-risk colour ramp
        #
        # 0.00 - 0.25 = Low
        # 0.25 - 0.50 = Moderate
        # 0.50 - 0.75 = High
        # 0.75 - 1.00 = Very High
        # ----------------------------------------------------

        rgba = np.zeros((*normalized.shape, 4), dtype=np.uint8)

        # Transparent background
        rgba[:, :, 3] = 0

        low = finite & (normalized < 0.25)
        moderate = finite & (normalized >= 0.25) & (normalized < 0.50)
        high = finite & (normalized >= 0.50) & (normalized < 0.75)
        very_high = finite & (normalized >= 0.75)

        # Low — green
        rgba[low, 0] = 34
        rgba[low, 1] = 197
        rgba[low, 2] = 139
        rgba[low, 3] = 85

        # Moderate — yellow/orange
        rgba[moderate, 0] = 242
        rgba[moderate, 1] = 169
        rgba[moderate, 2] = 59
        rgba[moderate, 3] = 105

        # High — red/orange
        rgba[high, 0] = 229
        rgba[high, 1] = 72
        rgba[high, 2] = 77
        rgba[high, 3] = 135

        # Very High — dark red
        rgba[very_high, 0] = 127
        rgba[very_high, 1] = 0
        rgba[very_high, 2] = 0
        rgba[very_high, 3] = 165

        image = Image.fromarray(rgba, mode="RGBA")

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        folium.raster_layers.ImageOverlay(
            image="data:image/png;base64," + encoded,
            bounds=FLOOD_BOUNDS,
            opacity=0.75,
            name=f"Flood Risk — {scenario}",
            interactive=True,
            cross_origin=False,
            zindex=1,
            show=True,
        ).add_to(fmap)

    except Exception:
        return


# ============================================================
# ROAD / TRAFFIC MAP
# ============================================================

def add_roads_layer(m, roads):

    roads = prepare_gdf(roads)

    if roads.empty:
        return None

    if "final_vc_ratio" not in roads.columns:
        return None

    traffic_group = folium.FeatureGroup(
        name="Traffic / V-C Ratio",
        show=True,
    )

    roads = roads.copy()

    roads["vc"] = pd.to_numeric(
        roads["final_vc_ratio"],
        errors="coerce"
    )

    roads["passability_num"] = pd.to_numeric(
        roads.get("passability"),
        errors="coerce"
    )

    def road_style(feature):

        props = feature.get("properties", {})

        vc = props.get("vc")
        passability = props.get("passability_num")

        try:
            vc = float(vc)
        except:
            vc = np.nan

        try:
            passability = float(passability)
        except:
            passability = np.nan

        # CLOSED
        if (
            np.isfinite(passability)
            and passability <= 0.001
        ):
            return {
                "color": "#7f0000",
                "weight": 5,
                "opacity": 0.95,
            }

        # NO V/C DATA
        if not np.isfinite(vc):
            return {
                "color": "#808080",
                "weight": 2,
                "opacity": 0.45,
            }

        # FREE FLOW
        if vc < 0.60:
            return {
                "color": "#16a34a",
                "weight": 3,
                "opacity": 0.85,
            }

        # MODERATE
        if vc < 0.80:
            return {
                "color": "#facc15",
                "weight": 4,
                "opacity": 0.90,
            }

        # HEAVY
        if vc <= 1.00:
            return {
                "color": "#f97316",
                "weight": 5,
                "opacity": 0.92,
            }

        # OVERSATURATED
        return {
            "color": "#dc2626",
            "weight": 6,
            "opacity": 0.95,
        }

    def road_popup(feature):

        p = feature.get("properties", {})

        road_id = p.get("road_id", "—")
        vc = p.get("vc")
        volume = p.get("final_volume_vph")
        capacity = p.get("capacity_vph")
        passability = p.get("passability_num")

        try:
            vc_text = f"{float(vc):.2f}"
        except:
            vc_text = "N/A"

        try:
            volume_text = f"{float(volume):,.0f}"
        except:
            volume_text = "N/A"

        try:
            capacity_text = f"{float(capacity):,.0f}"
        except:
            capacity_text = "N/A"

        try:
            passability_text = f"{float(passability):.2f}"
        except:
            passability_text = "N/A"

        html = f"""
        <div style="
            font-family:Arial;
            min-width:210px;
            font-size:13px;
        ">

            <div style="
                font-size:15px;
                font-weight:700;
                margin-bottom:8px;
            ">
                Road Traffic
            </div>

            <b>Road:</b> {road_id}<br>
            <b>V/C ratio:</b> {vc_text}<br>
            <b>Final volume:</b> {volume_text} veh/hr<br>
            <b>Capacity:</b> {capacity_text} veh/hr<br>
            <b>Passability:</b> {passability_text}

        </div>
        """

        return folium.Popup(
            html,
            max_width=280
        )

    folium.GeoJson(
        roads.to_json(),
        name="Traffic",
        style_function=road_style,
        popup_function=road_popup,
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "road_id",
                "vc",
                "final_volume_vph",
                "capacity_vph",
            ],
            aliases=[
                "Road",
                "V/C",
                "Final Volume",
                "Capacity",
            ],
            localize=True,
            sticky=False,
        ),
    ).add_to(traffic_group)

    traffic_group.add_to(m)

    return traffic_group

    # --------------------------------------------------------
    # Ensure numeric traffic fields
    # --------------------------------------------------------

    if "final_vc_ratio" in roads.columns:
        roads["__vc"] = pd.to_numeric(
            roads["final_vc_ratio"],
            errors="coerce"
        )
    else:
        roads["__vc"] = np.nan

    if "passability" in roads.columns:
        roads["__passability"] = pd.to_numeric(
            roads["passability"],
            errors="coerce"
        )
    else:
        roads["__passability"] = np.nan

    # --------------------------------------------------------
    # V/C categories
    # --------------------------------------------------------

    categories = [
        (
            "Free Flow — V/C < 0.60",
            roads[
                (roads["__vc"] < 0.60)
                & roads["__vc"].notna()
                & (roads["__passability"] > 0)
            ],
            "#22c58b",
        ),
        (
            "Moderate — V/C 0.60–0.80",
            roads[
                (roads["__vc"] >= 0.60)
                & (roads["__vc"] < 0.80)
                & roads["__vc"].notna()
                & (roads["__passability"] > 0)
            ],
            "#f2c94c",
        ),
        (
            "Heavy — V/C 0.80–1.00",
            roads[
                (roads["__vc"] >= 0.80)
                & (roads["__vc"] <= 1.00)
                & roads["__vc"].notna()
                & (roads["__passability"] > 0)
            ],
            "#f2994a",
        ),
        (
            "Oversaturated — V/C > 1.00",
            roads[
                (roads["__vc"] > 1.00)
                & roads["__vc"].notna()
                & (roads["__passability"] > 0)
            ],
            "#e5484d",
        ),
        (
            "Closed / Impassable",
            roads[
                roads["__passability"].notna()
                & (roads["__passability"] <= 0.001)
            ],
            "#7f0000",
        ),
        (
            "Traffic Data Unavailable",
            roads[
                roads["__vc"].isna()
                & (
                    roads["__passability"].isna()
                    | (roads["__passability"] > 0)
                )
            ],
            "#64748b",
        ),
    ]

    # --------------------------------------------------------
    # Add each traffic class as its own map layer
    # --------------------------------------------------------

    for name, subset, color in categories:

        if subset.empty:
            continue

        tooltip_fields = []

        if "road_id" in subset.columns:
            tooltip_fields.append("road_id")

        if "final_vc_ratio" in subset.columns:
            tooltip_fields.append("final_vc_ratio")

        if "final_volume_vph" in subset.columns:
            tooltip_fields.append("final_volume_vph")

        if "capacity_vph" in subset.columns:
            tooltip_fields.append("capacity_vph")

        if "passability" in subset.columns:
            tooltip_fields.append("passability")

        aliases = {
            "road_id": "Road",
            "final_vc_ratio": "Final V/C",
            "final_volume_vph": "Final volume (veh/hr)",
            "capacity_vph": "Capacity (veh/hr)",
            "passability": "Passability",
        }

        folium.GeoJson(
            subset.to_json(),
            name=f"Traffic — {name}",
            style_function=lambda feature, c=color: {
                "color": c,
                "weight": 4,
                "opacity": 0.9,
            },
            tooltip=(
                folium.GeoJsonTooltip(
                    fields=tooltip_fields,
                    aliases=[
                        aliases.get(field, field)
                        for field in tooltip_fields
                    ],
                    localize=True,
                    sticky=False,
                )
                if tooltip_fields
                else None
            ),
            show=True,
            smooth_factor=0.5,
        ).add_to(fmap)

def is_true_value(value):
    """
    Safely interpret boolean-like values from the frozen model.

    Prevents:
        bool("False") == True
    """

    if isinstance(value, bool):
        return value

    if value is None or pd.isna(value):
        return False

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) > 0

    text = str(value).strip().lower()

    return text in {
        "true",
        "1",
        "yes",
        "y",
        "affected",
        "operationally affected",
        "disrupted",
    }
# ============================================================
# FACILITY MAP
# ============================================================

def add_facilities_layer(
    fmap,
    facilities,
    recommended_facility_id=None
):
    if facilities is None or len(facilities) == 0:
        return

    facilities = prepare_gdf(facilities)

    if len(facilities) == 0:
        return

    # --------------------------------------------------------
    # Independent layer
    # --------------------------------------------------------

    facility_group = folium.FeatureGroup(
        name="Emergency Facilities",
        show=True
    )

    for _, row in facilities.iterrows():

        geometry = row.geometry

        if geometry is None:
            continue

        try:
            if geometry.geom_type == "Point":
                lat, lon = geometry.y, geometry.x
            else:
                point = geometry.centroid
                lat, lon = point.y, point.x
        except Exception:
            continue

        facility_id = str(
            row.get("facility_id", "")
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

        affected = is_true_value(
            row.get("operationally_affected", False)
        )

        response_disrupted = is_true_value(
            row.get("response_disrupted", False)
        )

        physically_affected = is_true_value(
            row.get("physically_affected", False)
        )

        is_recommended = (
            recommended_facility_id is not None
            and facility_id == str(
                recommended_facility_id
            )
        )

        # ----------------------------------------------------
        # Determine display state
        # ----------------------------------------------------

        if is_recommended:
            color = "#00ffff"
            radius = 10
            fill_opacity = 1.0
            status_text = "Recommended"

        elif affected or response_disrupted:
            color = "#e5484d"
            radius = 7
            fill_opacity = 0.95

            if physically_affected:
                status_text = "Physically affected"
            else:
                status_text = "Response disrupted"

        else:
            color = "#3aa0ff"
            radius = 5
            fill_opacity = 0.85
            status_text = "Operational"

        popup_html = f"""
        <div style="font-family:Arial;min-width:240px;">
            <h4 style="margin:0 0 8px 0;">
                {facility_name}
            </h4>

            <b>Facility ID:</b> {facility_id}<br>
            <b>Type:</b> {facility_type}<br>
            <b>Status:</b> {status_text}<br>
            <b>Flood impact:</b>
                {fmt_number(row.get("flood_impact", np.nan), 3)}<br>
            <b>Response time:</b>
                {fmt_minutes(row.get("response_time", np.nan))}
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
                max_width=340
            ),
            tooltip=(
                f"{facility_name} — {status_text}"
            ),
        ).add_to(facility_group)

    facility_group.add_to(fmap)


# ============================================================
# INCIDENT MAP
# ============================================================

def add_incidents_layer(
    fmap,
    incidents,
    selected_incident
):
    if incidents is None or len(incidents) == 0:
        return

    incidents = prepare_gdf(incidents)

    if len(incidents) == 0:
        return

    incident_group = folium.FeatureGroup(
        name="Emergency Incidents",
        show=True
    )

    for _, row in incidents.iterrows():

        geometry = row.geometry

        if geometry is None:
            continue

        try:
            if geometry.geom_type == "Point":
                lat, lon = geometry.y, geometry.x
            else:
                point = geometry.centroid
                lat, lon = point.y, point.x
        except Exception:
            continue

        incident_id = str(
            row.get("incident_id", "")
        )

        selected = (
            incident_id == str(selected_incident)
        )

        # Selected incident is visually prominent.
        if selected:
            color = "#ffffff"
            fill_color = "#e5484d"
            radius = 10
            weight = 4
        else:
            color = "#ffd166"
            fill_color = "#f2994a"
            radius = 6
            weight = 2

        population = row.get(
            "population",
            row.get(
                "population_in_catchment",
                np.nan
            )
        )

        flood_impact = row.get(
            "flood_impact",
            np.nan
        )

        popup_html = f"""
        <div style="font-family:Arial;min-width:220px;">
            <h4 style="margin:0 0 8px 0;">
                Emergency Incident
            </h4>

            <b>Incident:</b> {incident_id}<br>
            <b>Population:</b> {fmt_number(population)}<br>
            <b>Flood impact:</b>
                {fmt_number(flood_impact, 3)}
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.95,
            weight=weight,
            popup=folium.Popup(
                popup_html,
                max_width=320
            ),
            tooltip=(
                f"{incident_id}"
                + (" — SELECTED" if selected else "")
            ),
        ).add_to(incident_group)

    incident_group.add_to(fmap)


# ============================================================
# ROUTE DRAWING
# ============================================================

def add_route(fmap, route_row, color="#00ffff", weight=7, name="Recommended emergency route", show=True):
    if route_row is None:
        return

    if isinstance(route_row, pd.DataFrame):
        if len(route_row) == 0:
            return
        route_row = route_row.iloc[0]

    if route_row.get("geometry", None) is None:
        return

    try:
        route_gdf = gpd.GeoDataFrame([route_row], geometry="geometry", crs=ANALYSIS_CRS)
        route_gdf = prepare_gdf(route_gdf)

        fields = [f for f, _ in ROUTE_TOOLTIP_FIELDS if f in route_gdf.columns]
        aliases = [a for f, a in ROUTE_TOOLTIP_FIELDS if f in route_gdf.columns]

        folium.GeoJson(
            route_gdf.to_json(),
            name=name,
            style_function=lambda feature: {"color": color, "weight": weight, "opacity": 0.95},
            show=show,
            tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases, localize=True),
        ).add_to(fmap)

    except Exception:
        return


# ============================================================
# BUILD MAP
# ============================================================

def build_map(scenario, map_mode, incident_id):
    fmap = folium.Map(
        location=MAP_CENTER, zoom_start=MAP_ZOOM, control_scale=True, tiles=None,
    )

    # Base maps
    folium.TileLayer(tiles="OpenStreetMap", name="OpenStreetMap", control=True).add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
        control=True,
    ).add_to(fmap)

        # --------------------------------------------------------
    # Flood risk
    # --------------------------------------------------------

    add_flood_raster(fmap, scenario)

    folium.Rectangle(
        bounds=FLOOD_BOUNDS,
        color="#555555",
        weight=1,
        fill=False,
        name="URIP analysis boundary",
    ).add_to(fmap)

    # --------------------------------------------------------
    # Traffic / congestion
    # --------------------------------------------------------

    roads = get_roads(scenario)

    if map_mode in (
        "Flood Scenario",
        "Emergency Response"
    ):
        add_roads_layer(
            fmap,
            roads
        )

    # Roads
    roads = get_roads(scenario)
    if map_mode in ("Flood Scenario", "Emergency Response"):
        add_roads_layer(fmap, roads)

    # Emergency information
    facilities = get_facilities(scenario)
    incidents = get_incidents()
    report = get_incident_report(scenario, incident_id)

    recommended_facility_id = report.get("recommended_facility_id", None) if len(report) else None

    add_facilities_layer(fmap, facilities, recommended_facility_id)
    add_incidents_layer(fmap, incidents, incident_id)

        # --------------------------------------------------------
    # Selected incident flood impact
    # --------------------------------------------------------

    incident_flood = get_incident_flood(scenario)

    if len(incident_flood):

        selected_flood = prepare_gdf(
            incident_flood[
                incident_flood["incident_id"].astype(str)
                == str(incident_id)
            ].copy()
        )

        if len(selected_flood):

            flood_incident_group = folium.FeatureGroup(
                name="Selected Incident Flood Impact",
                show=True
            )

            for _, row in selected_flood.iterrows():

                geometry = row.geometry

                if geometry is None:
                    continue

                try:
                    lat, lon = geometry.y, geometry.x

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
                        fill_opacity=0.22,
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
                    ).add_to(
                        flood_incident_group
                    )

                except Exception:
                    pass

            flood_incident_group.add_to(fmap)

    # Emergency route
    if map_mode == "Emergency Response" and len(report):
        facility_rank = safe_int(report.get("facility_rank", 1), 1)
        route_rank = safe_int(report.get("route_rank", 1), 1)

        route_rows = ROUTES_ALTERNATIVES[
            (ROUTES_ALTERNATIVES["scenario"] == scenario)
            & (ROUTES_ALTERNATIVES["incident_id"] == incident_id)
            & (ROUTES_ALTERNATIVES["facility_rank"] == facility_rank)
            & (ROUTES_ALTERNATIVES["route_rank"] == route_rank)
        ]

        if len(route_rows):
            add_route(
                fmap,
                route_rows.iloc[0],
                color="#00ffff",
                weight=8,
                name="Recommended emergency route",
                show=True,
            )

    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


# ============================================================
# UI HELPERS — status pills, badges, small display helpers
# ============================================================

def pill(text, kind="neutral"):
    """A small colored status badge. kind: good | warn | bad | info | neutral."""
    return f'<span class="pill pill-{kind}">{text}</span>'


def route_status(delay_pct, closed_segments, is_recommended):
    if is_recommended:
        return "Recommended", "info"
    if safe_int(closed_segments, 0) > 0:
        return "Closed", "bad"
    delay_pct = safe_float(delay_pct)
    if not np.isnan(delay_pct) and delay_pct > 100:
        return "At Risk", "bad"
    if not np.isnan(delay_pct) and delay_pct > 15:
        return "Risky", "warn"
    return "Available", "good"


def facility_status(reachable, delay_min):
    if not reachable:
        return "Unreachable", "bad"
    delay_min = safe_float(delay_min)
    if not np.isnan(delay_min) and delay_min > 5:
        return "At Risk", "warn"
    return "Accessible", "good"


def get_incident_coords(incident_id):
    incidents = prepare_gdf(get_incidents())
    if len(incidents) == 0 or "incident_id" not in incidents.columns:
        return None, None

    rows = incidents[incidents["incident_id"].astype(str) == str(incident_id)]
    if len(rows) == 0:
        return None, None

    geometry = rows.iloc[0].geometry
    if geometry is None:
        return None, None

    try:
        return geometry.y, geometry.x
    except Exception:
        return None, None


# ============================================================
# THEME
# ============================================================

render_html(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 15% 0%, #0f2036 0%, #0a1420 45%) fixed;
    }

    [data-testid="stSidebar"] {
        background: #0c1728;
        border-right: 1px solid #1c2c42;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* ---------- header ---------- */
    .urip-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 1.4rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #0e2136 0%, #0b1a2c 100%);
        border: 1px solid #1c2c42;
        margin-bottom: 1.1rem;
    }
    .urip-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .urip-logo {
        width: 42px;
        height: 42px;
        border-radius: 11px;
        background: linear-gradient(135deg, #22c58b, #1a8dd8);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        flex-shrink: 0;
    }
    .urip-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #f2f6fb;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .urip-subtitle {
        font-size: 0.82rem;
        color: #22c58b;
        font-weight: 500;
        margin-top: 0.1rem;
    }
    .urip-meta {
        text-align: right;
        color: #9fb2c9;
        font-size: 0.82rem;
        line-height: 1.5;
    }
    .urip-meta b { color: #e7edf5; }
    .urip-badge {
        display: inline-block;
        margin-top: 0.35rem;
        padding: 0.28rem 0.7rem;
        border-radius: 20px;
        background: rgba(34, 197, 139, 0.12);
        border: 1px solid rgba(34, 197, 139, 0.4);
        color: #22c58b;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }

    /* ---------- generic panel ---------- */
    .panel {
        background: #101d30;
        border: 1px solid #1c2c42;
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        margin-bottom: 1.1rem;
    }
    .panel-title {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: #7e92ac;
        margin-bottom: 0.85rem;
    }
    .panel-title span.count {
        color: #4d6280;
        font-weight: 500;
        text-transform: none;
        letter-spacing: 0;
    }

    /* ---------- KPI cards ---------- */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 0.8rem;
        margin-bottom: 1.1rem;
    }
    @media (max-width: 1100px) {
        .kpi-grid { grid-template-columns: repeat(3, 1fr); }
    }
    .kpi-card {
        background: #101d30;
        border: 1px solid #1c2c42;
        border-radius: 12px;
        padding: 0.85rem 1rem;
    }
    .kpi-label {
        font-size: 0.72rem;
        color: #7e92ac;
        font-weight: 500;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 1.35rem;
        font-weight: 800;
        color: #f2f6fb;
        letter-spacing: -0.01em;
    }
    .kpi-sub {
        font-size: 0.72rem;
        color: #4d6280;
        margin-top: 0.15rem;
    }

    /* ---------- pills ---------- */
    .pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        white-space: nowrap;
    }
    .pill-good { background: rgba(34, 197, 139, 0.14); color: #22c58b; border: 1px solid rgba(34,197,139,0.35); }
    .pill-warn { background: rgba(242, 169, 59, 0.14); color: #f2a93b; border: 1px solid rgba(242,169,59,0.35); }
    .pill-bad  { background: rgba(229, 72, 77, 0.14);  color: #f0787c; border: 1px solid rgba(229,72,77,0.35); }
    .pill-info { background: rgba(58, 160, 255, 0.14); color: #5eb3ff; border: 1px solid rgba(58,160,255,0.35); }
    .pill-neutral { background: rgba(126, 146, 172, 0.14); color: #9fb2c9; border: 1px solid rgba(126,146,172,0.3); }

    /* ---------- callouts ---------- */
    .callout {
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 1.1rem;
        border: 1px solid;
    }
    .callout-title {
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .callout-body {
        font-size: 0.88rem;
        color: #cddaea;
        line-height: 1.45;
    }
    .callout-info  { background: rgba(58, 160, 255, 0.08);  border-color: rgba(58, 160, 255, 0.3); }
    .callout-info  .callout-title { color: #5eb3ff; }
    .callout-good  { background: rgba(34, 197, 139, 0.08);  border-color: rgba(34, 197, 139, 0.3); }
    .callout-good  .callout-title { color: #22c58b; }
    .callout-warn  { background: rgba(242, 169, 59, 0.08);  border-color: rgba(242, 169, 59, 0.3); }
    .callout-warn  .callout-title { color: #f2a93b; }

    /* ---------- data table ---------- */
    table.urip-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.84rem;
    }
    table.urip-table th {
        text-align: left;
        color: #7e92ac;
        font-weight: 600;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        padding: 0.4rem 0.6rem;
        border-bottom: 1px solid #1c2c42;
    }
    table.urip-table td {
        padding: 0.5rem 0.6rem;
        border-bottom: 1px solid #16233a;
        color: #d7e1ee;
    }
    table.urip-table tr:last-child td { border-bottom: none; }
    table.urip-table td.num { text-align: right; font-variant-numeric: tabular-nums; }
    table.urip-table tr.is-recommended td { background: rgba(34, 197, 139, 0.06); }

    /* ---------- facility card ---------- */
    .facility-block {
        margin-bottom: 1rem;
        padding-bottom: 0.9rem;
        border-bottom: 1px solid #16233a;
    }
    .facility-block:last-child {
        border-bottom: none;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .facility-name-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.2rem;
    }
    .facility-name {
        font-size: 0.95rem;
        font-weight: 700;
        color: #f2f6fb;
    }
    .facility-sub {
        font-size: 0.76rem;
        color: #7e92ac;
        margin-bottom: 0.55rem;
    }

    /* ---------- insight list ---------- */
    .insight-row {
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
        padding: 0.5rem 0;
        border-bottom: 1px solid #16233a;
        font-size: 0.86rem;
        color: #cddaea;
    }
    .insight-row:last-child { border-bottom: none; }
    .insight-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-top: 0.35rem;
        flex-shrink: 0;
    }
    .dot-good { background: #22c58b; }
    .dot-warn { background: #f2a93b; }
    .dot-bad  { background: #e5484d; }
    .dot-info { background: #3aa0ff; }

    /* ---------- legend ---------- */
    .legend-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.8rem;
        color: #cddaea;
        margin-bottom: 0.35rem;
    }
    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    /* ---------- misc ---------- */
    .muted { color: #7e92ac; }
    .model-note {
        color: #55688a;
        font-size: 0.78rem;
        padding-top: 1rem;
        border-top: 1px solid #1c2c42;
        margin-top: 1.5rem;
        line-height: 1.5;
    }

    /* tighten native widget look */
    div[data-testid="stSelectbox"] label, div[data-testid="stRadio"] label {
        color: #9fb2c9 !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
    }
    </style>
    """
)

# ============================================================
# HEADER
# ============================================================

now = pd.Timestamp.now()

render_html(
    f"""
    <div class="urip-header">
        <div class="urip-brand">
            <div class="urip-logo">🌍</div>
            <div>
                <div class="urip-title">URIP</div>
                <div class="urip-subtitle">Urban Resilience Intelligence Platform</div>
            </div>
        </div>
        <div class="urip-meta">
            <div>📍 <b>Nairobi CBD</b> &nbsp; • &nbsp; {now.strftime("%-d %b %Y, %H:%M")}</div>
            <div class="urip-badge">FROZEN MODEL · {MODEL_FILE}</div>
        </div>
    </div>
    """
)

# ============================================================
# CONTROL BAR
# ============================================================

with st.container():
    ctrl_cols = st.columns([1.3, 1.3, 1.6, 1.8])

    with ctrl_cols[0]:
        scenario = st.selectbox("Rainfall scenario", SCENARIOS, index=0)

    rainfall = SCENARIO_RAINFALL[scenario]

    with ctrl_cols[1]:
        map_mode = st.radio(
            "Map mode", ["Flood Scenario", "Emergency Response"], index=0, horizontal=True
        )

    incident_ids = sorted(
        INCIDENT_REPORT[INCIDENT_REPORT["scenario"] == scenario]["incident_id"]
        .dropna().astype(str).unique().tolist()
    )
    if not incident_ids:
        incident_ids = sorted(
            INCIDENT_REPORT["incident_id"].dropna().astype(str).unique().tolist()
        )

    with ctrl_cols[2]:
        incident_id = st.selectbox("Emergency incident", incident_ids, index=0)

    with ctrl_cols[3]:
        render_html(
            f"""
            <div style="padding-top: 1.7rem; font-size: 0.82rem;">
                <span class="muted">Rainfall intensity</span>
                <b style="color:#e7edf5;"> {rainfall} mm/hr</b>
                &nbsp;·&nbsp;
                <span class="muted">Map mode</span>
                <b style="color:#e7edf5;"> {map_mode}</b>
            </div>
            """
        )

# ============================================================
# GET CURRENT DATA
# ============================================================

kpis = get_dashboard_kpis(scenario)
kpi_row = get_emergency_kpi(scenario)
situation = get_situation(scenario)
incident_report = get_incident_report(scenario, incident_id)
facility_options = get_facility_options(scenario, incident_id)

# ============================================================
# KPI STRIP
# ============================================================

kpi_defs = [
    ("Flood Impact", fmt_number(kpis["flood_impact"], 3), "mean index"),
    ("Roads Affected", fmt_number(kpis["roads_affected"]), f"of {len(get_roads(scenario))} segments"),
    ("Roads Closed", fmt_number(kpis["roads_closed"]), "fully impassable"),
    ("Facilities Affected", fmt_number(kpis["facilities_affected"]), "operationally"),
    ("Population Exposed", fmt_number(kpis["population_exposed"]), "residents"),
    ("Mean Response Time", fmt_minutes(kpis["mean_response"]), "citywide"),
]

kpi_html = '<div class="kpi-grid">'
for label, value, sub in kpi_defs:
    kpi_html += (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f"</div>"
    )
kpi_html += "</div>"
render_html(kpi_html)

# ============================================================
# SITUATION + INCIDENT SUMMARY ROW
# ============================================================

left_col, right_col = st.columns([1.35, 1])

with left_col:
    alert = kpi_row.get("emergency_alert", None) if len(kpi_row) else None
    if alert is not None and not pd.isna(alert):
        callout_kind = "callout-info" if scenario == "Normal" else "callout-warn"
        render_html(
            f"""
            <div class="callout {callout_kind}">
                <div class="callout-title">Current Situation</div>
                <div class="callout-body">{alert}</div>
            </div>
            """
        )

    status_fields = [
        ("Situation", situation.get("situation_status", "—")),
        ("Response network", situation.get("response_network_status", "—")),
        ("Facility status", situation.get("facility_status", "—")),
        ("Route availability", fmt_pct(kpi_row.get("route_availability_pct", np.nan))),
    ]

    status_html = '<div class="panel"><div class="panel-title">Emergency Situation</div>'
    status_html += '<div class="kpi-grid" style="grid-template-columns:repeat(4,1fr);">'
    for title, value in status_fields:
        status_html += (
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{title}</div>'
            f'<div class="kpi-value" style="font-size:1.05rem;">{value}</div>'
            f"</div>"
        )
    status_html += "</div></div>"
    render_html(status_html)

    secondary_defs = [
        ("Access Disrupted", fmt_number(kpis["access_disrupted"])),
        ("Priority Population", fmt_number(kpis["priority_population"])),
        ("High Priority", fmt_number(kpis["high_priority"])),
        ("Mean V/C Ratio", fmt_number(kpis["mean_vc"], 3)),
    ]
    secondary_html = '<div class="panel"><div class="panel-title">Population &amp; Network Load</div>'
    secondary_html += '<div class="kpi-grid" style="grid-template-columns:repeat(4,1fr);">'
    for title, value in secondary_defs:
        secondary_html += (
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{title}</div>'
            f'<div class="kpi-value" style="font-size:1.05rem;">{value}</div>'
            f"</div>"
        )
    secondary_html += "</div></div>"
    render_html(secondary_html)

with right_col:
    incident_lat, incident_lon = get_incident_coords(incident_id)
    coord_text = (
        f"{incident_lat:.4f}, {incident_lon:.4f}"
        if incident_lat is not None and incident_lon is not None
        else "—"
    )

    incident_html = (
        '<div class="panel">'
        '<div class="panel-title">Selected Incident</div>'
        f'<div class="facility-name" style="margin-bottom:0.2rem;">{incident_id}</div>'
        f'<div class="facility-sub">Nairobi CBD &nbsp;·&nbsp; {scenario}</div>'
    )

    if len(incident_report):
        flood_class = str(incident_report.get("incident_flood_class", "—"))
        flood_class_kind = {
            "Low": "good", "Moderate": "warn", "High": "bad", "Very High": "bad",
        }.get(flood_class, "neutral")

        incident_html += (
            '<div style="margin: 0.6rem 0;">'
            f'{pill(flood_class + " flood risk", flood_class_kind)}'
            "</div>"
            '<div style="font-size:0.82rem; color:#9fb2c9; line-height:1.9;">'
            f'Flood impact &nbsp;<b style="color:#e7edf5;">{fmt_number(incident_report.get("incident_flood_impact", np.nan), 3)}</b><br>'
            f'Catchment population &nbsp;<b style="color:#e7edf5;">{fmt_number(incident_report.get("population_in_catchment", np.nan))}</b><br>'
            f'Priority population &nbsp;<b style="color:#e7edf5;">{fmt_number(incident_report.get("priority_population", np.nan))}</b><br>'
            f'Coordinates &nbsp;<b style="color:#e7edf5;">{coord_text}</b>'
            "</div>"
        )

    incident_html += "</div>"
    render_html(incident_html)

    if len(incident_report):
        recommended_facility = incident_report.get("recommended_facility_name", None)
        recommended_facility_id = incident_report.get("recommended_facility_id", None)
        recommended_response = incident_report.get("recommended_facility_response_time_min", np.nan)
        recommended_route_time = incident_report.get("recommended_route_response_time_min", np.nan)
        route_length = safe_float(incident_report.get("recommended_route_length_m", np.nan))
        route_length_text = "—" if np.isnan(route_length) else fmt_number(route_length) + " m"

        render_html(
            f"""
            <div class="callout callout-good">
                <div class="callout-title">Recommended Route</div>
                <div class="callout-body">
                    Dispatch via <b style="color:#e7edf5;">{recommended_facility or "—"}</b>
                    for minimal delay and flood exposure.
                    <br><br>
                    <span class="muted">Facility response</span>
                    <b style="color:#e7edf5;"> {fmt_minutes(recommended_response)}</b>
                    &nbsp;·&nbsp;
                    <span class="muted">Route response</span>
                    <b style="color:#e7edf5;"> {fmt_minutes(recommended_route_time)}</b>
                    &nbsp;·&nbsp;
                    <span class="muted">Route length</span>
                    <b style="color:#e7edf5;"> {route_length_text}</b>
                </div>
            </div>
            """
        )
    else:
        recommended_facility_id = None

# ============================================================
# MAP + INSIGHTS
# ============================================================

map_col, insight_col = st.columns([2.3, 1])

with map_col:
    render_html('<div class="panel-title" style="margin-top:0.4rem;">Spatial Intelligence</div>')
    fmap = build_map(scenario, map_mode, incident_id)
    st_folium(fmap, width=None, height=640, returned_objects=[], use_container_width=True)

with insight_col:
    insights = []

    roads_affected = safe_int(kpis["roads_affected"], -1)
    if roads_affected >= 0:
        total_roads = len(get_roads(scenario))
        pct = (roads_affected / total_roads * 100) if total_roads else 0
        insights.append((
            f"{roads_affected} of {total_roads} road segments are affected ({pct:.0f}% of the network).",
            "warn" if roads_affected else "good",
        ))

    roads_closed = safe_int(kpis["roads_closed"], -1)
    if roads_closed > 0:
        insights.append((f"{roads_closed} road segments are fully closed to traffic.", "bad"))

    facilities_affected = safe_int(kpis["facilities_affected"], -1)
    if facilities_affected >= 0:
        insights.append((
            f"{facilities_affected} emergency facilities are operationally affected.",
            "warn" if facilities_affected else "good",
        ))

    pop_exposed = safe_int(kpis["population_exposed"], -1)
    if pop_exposed > 0:
        insights.append((f"{pop_exposed:,} people are in flood-exposed areas within the CBD.", "info"))

    mean_response = safe_float(kpis["mean_response"])
    if not np.isnan(mean_response):
        insights.append((f"Mean citywide emergency response time is {fmt_minutes(mean_response)}.", "info"))

    if not insights:
        insights.append(("No frozen scenario insights are available for this selection.", "neutral"))

    insight_html = '<div class="panel"><div class="panel-title">Key Insights</div>'
    for text, kind in insights:
        insight_html += (
            f'<div class="insight-row"><div class="insight-dot dot-{kind}"></div><div>{text}</div></div>'
        )
    insight_html += "</div>"
    render_html(insight_html)

    legend_html = '<div class="panel"><div class="panel-title">Road Passability Legend</div>'
    for status, color in ROAD_COLORS.items():
        legend_html += (
            f'<div class="legend-row"><div class="legend-dot" style="background:{color};"></div>{status}</div>'
        )
    legend_html += "</div>"
    render_html(legend_html)

# ============================================================
# FACILITY OPTIONS + ROUTES
# ============================================================

render_html(
    f'<div class="panel-title" style="margin-top:0.3rem;">'
    f'Facility &amp; Route Comparison <span class="count">— {incident_id}, {scenario}</span></div>'
)

if len(facility_options) == 0:
    render_html(
        '<div class="panel"><span class="muted">'
        "No frozen facility options are available for this incident and scenario."
        "</span></div>"
    )
else:
    blocks_html = '<div class="panel">'

    for _, facility in facility_options.iterrows():
        facility_rank = safe_int(facility.get("facility_rank", np.nan), 0)
        facility_id = str(facility.get("facility_id", ""))
        facility_name = str(facility.get("facility_name", "Unnamed facility"))
        facility_type = str(facility.get("facility_type", "Facility"))

        travel_time = safe_float(facility.get("travel_time_min", np.nan))
        normal_time = safe_float(facility.get("normal_travel_time_min", np.nan))
        delay = safe_float(facility.get("response_delay_min", np.nan))
        reachable = bool(facility.get("reachable", False))

        is_recommended = str(recommended_facility_id) == facility_id
        f_label, f_kind = facility_status(reachable, delay)
        if is_recommended:
            f_label, f_kind = "Recommended", "info"

        blocks_html += (
            '<div class="facility-block">'
            '<div class="facility-name-row">'
            f'<div class="facility-name">#{facility_rank} &nbsp;{facility_name}</div>'
            f'{pill(f_label, f_kind)}'
            "</div>"
            f'<div class="facility-sub">{facility_type} &nbsp;·&nbsp; {facility_id} '
            f"&nbsp;·&nbsp; response {fmt_minutes(travel_time)} "
            f"(normal {fmt_minutes(normal_time)}, delay {fmt_minutes(delay)})</div>"
        )

        route_options = get_route_options(scenario, incident_id, facility_rank)

        if len(route_options) == 0:
            blocks_html += '<span class="muted">No frozen route alternative is available for this facility.</span>'
        else:
            blocks_html += (
                '<table class="urip-table"><thead><tr>'
                "<th>Route</th><th class=\"num\">Response</th><th class=\"num\">Length</th>"
                "<th class=\"num\">Affected</th><th class=\"num\">Closed</th>"
                "<th class=\"num\">Delay</th><th>Status</th>"
                "</tr></thead><tbody>"
            )

            for _, route in route_options.iterrows():
                route_rank = safe_int(route.get("route_rank", np.nan), 0)
                response_time = safe_float(route.get("response_time_min", np.nan))
                route_length = safe_float(route.get("route_length_m", np.nan))
                affected_segments = safe_int(route.get("affected_segments", np.nan), 0)
                closed_segments = safe_int(route.get("closed_segments", np.nan), 0)
                delay_pct = safe_float(route.get("response_delay_pct", np.nan))

                route_is_recommended = is_recommended and route_rank == safe_int(
                    incident_report.get("route_rank", 1), 1
                )
                r_label, r_kind = route_status(delay_pct, closed_segments, route_is_recommended)
                row_class = "is-recommended" if route_is_recommended else ""

                blocks_html += (
                    f'<tr class="{row_class}">'
                    f"<td>Route {route_rank}</td>"
                    f'<td class="num">{fmt_minutes(response_time)}</td>'
                    f'<td class="num">{fmt_number(route_length)} m</td>'
                    f'<td class="num">{affected_segments}</td>'
                    f'<td class="num">{closed_segments}</td>'
                    f'<td class="num">{fmt_pct(delay_pct)}</td>'
                    f"<td>{pill(r_label, r_kind)}</td>"
                    "</tr>"
                )

            blocks_html += "</tbody></table>"

        blocks_html += "</div>"

    blocks_html += "</div>"
    render_html(blocks_html)

# ============================================================
# SCENARIO COMPARISON
# ============================================================

render_html('<div class="panel-title" style="margin-top:0.3rem;">Scenario Comparison</div>')

scenario_rows_html = [
    ("Situation", "situation_status", "text"),
    ("Mean flood impact", "mean_flood_impact", "num3"),
    ("High / very high incidents", "high_very_high_incidents", "int"),
    ("Mean response time", "mean_response_time_min", "min"),
    ("Route availability", "route_availability_pct", "pct"),
]

table_html = (
    '<div class="panel"><table class="urip-table"><thead><tr><th>Metric</th>'
)
for s in SCENARIOS:
    table_html += f"<th>{s}</th>"
table_html += "</tr></thead><tbody>"

for label, field, kind in scenario_rows_html:
    table_html += f"<tr><td>{label}</td>"
    for s in SCENARIOS:
        row = get_emergency_kpi(s)
        raw = row.get(field, np.nan) if len(row) else np.nan

        if kind == "text":
            cell = str(raw) if raw is not None and not pd.isna(raw) else "—"
            table_html += f"<td>{cell}</td>"
        elif kind == "num3":
            table_html += f'<td class="num">{fmt_number(raw, 3)}</td>'
        elif kind == "int":
            table_html += f'<td class="num">{safe_int(raw, 0)}</td>'
        elif kind == "min":
            table_html += f'<td class="num">{fmt_minutes(raw)}</td>'
        elif kind == "pct":
            table_html += f'<td class="num">{fmt_pct(raw)}</td>'
    table_html += "</tr>"

table_html += "</tbody></table></div>"
render_html(table_html)

# ============================================================
# MODEL NOTE
# ============================================================

render_html(
    """
    <div class="model-note">
        <b>URIP frozen-model dashboard.</b>
        All flood, road disruption, passability, traffic, rerouting, facility,
        population and emergency-response outputs shown here are read from the
        frozen analytical package. The Streamlit interface does not recalculate
        the analytical model.
        <br><br>
        Emergency routing represents modelled network routing. Route alternatives
        shown are the frozen alternatives generated by the URIP routing model.
        Affected route segments indicate flood-affected segments and do not
        necessarily mean that the segment is closed.
    </div>
    """
)
