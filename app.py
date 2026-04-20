# Action Map — Clean (Actions + xT) — v4
# xT: Options C + D + Base Angle Correction
# New: Distance-Difficulty Bonus (log-scaled, capped, robust)
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from streamlit_image_coordinates import streamlit_image_coordinates
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.path import Path
import math

# ==========================
# Page Configuration
# ==========================
st.set_page_config(layout="wide", page_title="Action Map — Clean (Actions + xT)")

st.markdown(
    """
    <style>
    .small-metric { padding: 6px 8px; }
    .small-metric .label  { font-size: 12px; color: #ffffff; margin-bottom: 3px; opacity: 0.95; }
    .small-metric .value  { font-size: 18px; font-weight: 600; color: #ffffff; }
    .small-metric .delta  { font-size: 11px; color: #e6e6e6; margin-top: 4px; }
    .stats-section-title  { font-size: 14px; font-weight: 600; margin-bottom: 6px; color: #ffffff; }
    .streamlit-expanderHeader { color: #ffffff !important; }
    .streamlit-expander        { background: rgba(255,255,255,0.02); }
    .filter-panel {
      background: linear-gradient(168deg, rgba(30,39,56,0.92) 0%, rgba(22,28,40,0.97) 100%);
      border: 1px solid rgba(255,255,255,0.08); border-radius: 14px;
      padding: 24px 18px 20px 18px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.25), 0 1px 4px rgba(0,0,0,0.12);
      backdrop-filter: blur(6px);
    }
    .filter-panel h3 { font-size: 15px; color: #c8d6e5; letter-spacing: 0.5px; margin-bottom: 8px; }
    .filter-panel .filter-divider { border:none; border-top:1px solid rgba(255,255,255,0.07); margin:14px 0; }
    .stSubheader { color: #ffffff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

def small_metric(label: str, value: str, delta: str | None = None):
    html = f'<div class="small-metric"><div class="label">{label}</div><div class="value">{value}</div>'
    if delta is not None:
        html += f'<div class="delta">{delta}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ==========================
# Constants
# ==========================
st.title("Action Map — Clean (Actions + xT)")

FIELD_X, FIELD_Y   = 120.0, 80.0
HALF_LINE_X        = FIELD_X / 2
FINAL_THIRD_LINE_X = 80
LANE_LEFT_MIN      = 53.33
LANE_RIGHT_MAX     = 26.67
NX, NY             = 16, 12
LATERAL_MIN_DIST   = 12.0

COLOR_TOP   = "#2F80ED"
COLOR_OTHER = "#bfc4ca"
COLOR_FAIL  = "#FF6B6B"
COLOR_ADJ   = "#F4A261"   # orange — highlight for top adjusted ΔxT

START_DOT_SIZE = 28
FIG_W, FIG_H   = 7.9, 5.3
FIG_DPI        = 110

# ==========================
# Distance-Difficulty Bonus
# ==========================
# D_REF    : below this distance (m), no bonus is awarded
# D_SCALE  : log scaling factor — lower = steeper initial curve
# BONUS_CAP: maximum bonus multiplier (0.60 = up to +60% on ΔxT)
#
# Curve shape (D_REF=10, D_SCALE=20, CAP=0.60):
#   5 m  → 0.000  (shorter than ref, no bonus)
#  15 m  → 0.182  (+18%)
#  25 m  → 0.336  (+34%)
#  40 m  → 0.470  (+47%)
#  60 m+ → 0.600  (+60%, capped)
D_REF     = 10.0
D_SCALE   = 20.0
BONUS_CAP = 0.60

def distance_bonus(distance: float | np.ndarray) -> float | np.ndarray:
    """
    Log-scaled, capped distance-difficulty bonus multiplier.
    Returns a value in [0, BONUS_CAP].
    Statistically robust properties:
      - Logarithmic: diminishing marginal returns (40→50m adds less than 10→20m)
      - Capped: single outlier long action cannot dominate rankings
      - Reference threshold: noise-level short actions get zero bonus
    """
    excess = np.maximum(0.0, np.asarray(distance, dtype=float) - D_REF)
    raw    = np.log1p(excess / D_SCALE)          # log(1 + excess/scale)
    return float(np.minimum(BONUS_CAP, raw)) if np.ndim(distance) == 0 else np.minimum(BONUS_CAP, raw)

def adjusted_delta_xt(delta_xt: float | np.ndarray,
                      distance:  float | np.ndarray) -> float | np.ndarray:
    """
    ΔxT_adj = ΔxT × (1 + distance_bonus)
    Only applied to successful actions (ΔxT > 0).
    """
    bonus = distance_bonus(distance)
    return np.asarray(delta_xt) * (1.0 + bonus)

# ==========================
# xT Grid — Options C + D + Base Angle Correction
# ==========================
@st.cache_data(show_spinner=False)
def compute_xt_grid(
    NX=16, NY=12, sub=24,
    goal_width=11.0, penalty_depth=18.5, penalty_width=45.32,
    prox_w=0.50, central_w=0.50,
    internal_prox_power=2.8, internal_central_power=2.4, center_boost=0.20,
    FUNNEL_INFLUENCE_RANGE=35.0, FUNNEL_POWER=1.3, BASE_BOOST_WEIGHT=0.15,
    band_width_m=180.0, blur_window_m=60.0, final_blur_m=12.0,
    ANGLE_WEIGHT=0.50, ANGLE_POWER=1.4,
    BASE_ANGLE_WEIGHT=0.40,
):
    ncols_hr = NX * sub
    nrows_hr = NY * sub
    x_edges_hr   = np.linspace(0.0, FIELD_X, ncols_hr + 1)
    y_edges_hr   = np.linspace(0.0, FIELD_Y, nrows_hr + 1)
    x_centers_hr = (x_edges_hr[:-1] + x_edges_hr[1:]) / 2.0
    y_centers_hr = (y_edges_hr[:-1] + y_edges_hr[1:]) / 2.0
    Xc_hr, Yc_hr = np.meshgrid(x_centers_hr, y_centers_hr)

    xp         = 0.01 + (Xc_hr / FIELD_X) * 0.99
    yc         = 1.0 - np.abs((Yc_hr / FIELD_Y) - 0.5) * 2.0
    XT_BASE_hr = xp * (0.8 + 0.2 * yc)
    XT_BASE_hr = (XT_BASE_hr - XT_BASE_hr.min()) / (XT_BASE_hr.max() - XT_BASE_hr.min() + 1e-12)

    center_y          = FIELD_Y / 2.0
    left_goal_post    = (FIELD_X, center_y - goal_width / 2.0)
    right_goal_post   = (FIELD_X, center_y + goal_width / 2.0)
    x_big             = FIELD_X - penalty_depth
    big_top_corner    = (x_big, center_y + penalty_width / 2.0)
    big_bottom_corner = (x_big, center_y - penalty_width / 2.0)
    funnel_vertices   = [left_goal_post, big_bottom_corner, big_top_corner, right_goal_post]
    inside_flags      = Path(funnel_vertices).contains_points(
        np.column_stack([Xc_hr.ravel(), Yc_hr.ravel()])
    ).reshape(Xc_hr.shape)

    boundary_pts = []
    for i in range(len(funnel_vertices)):
        a, b   = funnel_vertices[i], funnel_vertices[(i + 1) % len(funnel_vertices)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        num    = max(2, int(round(math.hypot(dx, dy) / 0.5)))
        for t in np.linspace(0.0, 1.0, num, endpoint=False):
            boundary_pts.append((a[0] + dx * t, a[1] + dy * t))
    boundary_pts = np.array(boundary_pts)

    flat_X = Xc_hr.ravel(); flat_Y = Yc_hr.ravel()
    min_d2 = np.full(flat_X.size, np.inf)
    for bp in boundary_pts:
        dx = flat_X - bp[0]; dy = flat_Y - bp[1]
        np.minimum(min_d2, dx*dx + dy*dy, out=min_d2)
    abs_dist = np.sqrt(min_d2).reshape(Xc_hr.shape)

    influence_curve = np.clip((1.0 - np.clip(abs_dist / FUNNEL_INFLUENCE_RANGE, 0, 1)) ** FUNNEL_POWER, 0, 1)

    D_hr       = np.hypot(FIELD_X - Xc_hr, center_y - Yc_hr)
    prox_hr    = 1.0 - np.clip(D_hr / np.hypot(FIELD_X, FIELD_Y / 2.0), 0, 1)
    central_hr = 1.0 - np.clip(np.abs((Yc_hr - center_y) / center_y), 0, 1)
    unit_bonus = np.clip(
        (prox_w * np.clip(prox_hr ** internal_prox_power, 0, 1)
         + central_w * np.clip(central_hr ** internal_central_power, 0, 1))
        * (1.0 + center_boost * prox_hr), 0, 1)

    v1x = FIELD_X - Xc_hr; v1y = (center_y + goal_width/2.0) - Yc_hr
    v2x = FIELD_X - Xc_hr; v2y = (center_y - goal_width/2.0) - Yc_hr
    cos_a        = np.clip((v1x*v2x + v1y*v2y) / (np.hypot(v1x,v1y)*np.hypot(v2x,v2y) + 1e-12), -1, 1)
    angle_factor = np.clip((np.arccos(cos_a) / (np.arccos(cos_a).max() + 1e-12)) ** ANGLE_POWER, 0, 1)

    unit_bonus = np.clip(unit_bonus * ((1 - ANGLE_WEIGHT) + ANGLE_WEIGHT * angle_factor), 0, 1)

    XT_BASE_c = XT_BASE_hr * ((1 - BASE_ANGLE_WEIGHT) + BASE_ANGLE_WEIGHT * angle_factor)
    XT_BASE_c = (XT_BASE_c - XT_BASE_c.min()) / (XT_BASE_c.max() - XT_BASE_c.min() + 1e-12)

    XT_BOOST = XT_BASE_c + influence_curve * BASE_BOOST_WEIGHT * unit_bonus

    px_w = FIELD_X / ncols_hr; px_h = FIELD_Y / nrows_hr
    rx   = max(1, int(round((blur_window_m / px_w) / 2.0)))
    ry   = max(1, int(round((blur_window_m / px_h) / 2.0)))

    def box_blur(arr, rx, ry):
        H, W = arr.shape
        a    = np.pad(arr, ((ry,ry),(rx,rx)), mode="edge").astype(np.float64)
        ii   = a.cumsum(0).cumsum(1)
        s    = ii[2*ry:2*ry+H, 2*rx:2*rx+W].copy()
        s   += ii[:H, :W]; s -= ii[:H, 2*rx:2*rx+W]; s -= ii[2*ry:2*ry+H, :W]
        return s / ((2*ry+1)*(2*rx+1))

    w          = 0.5 * (1.0 - np.cos(np.pi * np.clip(abs_dist / band_width_m, 0, 1)))
    XT_BLENDED = w * XT_BOOST + (1 - w) * box_blur(XT_BOOST, rx, ry)

    rx_f = max(1, int(round((final_blur_m / px_w) / 2.0)))
    ry_f = max(1, int(round((final_blur_m / px_h) / 2.0)))
    XT   = 0.85 * XT_BLENDED + 0.15 * box_blur(XT_BLENDED, rx_f, ry_f)
    XT   = (XT - XT.min()) / (XT.max() - XT.min() + 1e-12)

    XT_coarse = np.zeros((NY, NX))
    for iy in range(NY):
        for ix in range(NX):
            XT_coarse[iy, ix] = XT[iy*sub:(iy+1)*sub, ix*sub:(ix+1)*sub].mean()
    XT_coarse = (XT_coarse - XT_coarse.min()) / (XT_coarse.max() - XT_coarse.min() + 1e-12)
    return XT_coarse, XT, inside_flags

XT_GRID, _, _ = compute_xt_grid()

def xt_value(x, y):
    ix = int(np.clip((x / FIELD_X) * NX, 0, NX - 1))
    iy = int(np.clip((y / FIELD_Y) * NY, 0, NY - 1))
    return float(XT_GRID[iy, ix])

# ==========================
# DATA
# ==========================
matches_data = {
    "Ali vs Vancouver": [
        ("ACTION WON", 50.03, 5.76, 48.86, 14.07, None),
        ("ACTION WON", 42.05, 4.26, 65.82, 19.39, None),
        ("ACTION WON", 53.68, 12.57, 39.72, 28.20, None),
        ("ACTION WON", 43.88, 37.17, 44.54, 44.65, None),
        ("ACTION WON", 76.29, 23.21, 65.65, 22.38, None),
        ("ACTION WON", 78.62, 25.54, 87.26, 26.37, None),
        ("ACTION WON", 67.48, 5.76, 76.96, 6.42, None),
        ("ACTION WON", 61.83, 3.43, 111.20, 9.75, None),
        ("ACTION WON", 83.27, 2.93, 118.51, 19.89, None),
        ("ACTION WON", 97.90, 6.75, 111.53, 9.91, None),
        ("ACTION WON", 114.03, 1.93, 107.71, 12.57, None),
        ("ACTION WON", 98.23, 5.59, 90.09, 7.58, None),
        ("ACTION WON", 96.57, 5.92, 91.92, 14.73, None),
        ("ACTION WON", 87.43, 12.24, 78.78, 9.41, None),
        ("ACTION WON", 77.62, 1.93, 72.30, 3.93, None),
        ("ACTION WON", 79.28, 5.59, 70.81, 2.26, None),
        ("ACTION WON", 62.83, 3.43, 79.62, 7.25, None),
        ("ACTION WON", 53.18, 9.41, 68.98, 13.74, None),
        ("ACTION WON", 51.69, 4.76, 40.38, 8.58, None),
        ("ACTION LOST", 116.35, 2.93, 118.68, 11.74, None),
        ("ACTION LOST", 107.88, 10.58, 109.54, 39.83, None),
        ("ACTION LOST", 86.10, 3.43, 87.59, 4.09, None),
        ("ACTION LOST", 73.46, 2.43, 75.13, 3.43, None),
        ("ACTION LOST", 53.18, 2.60, 70.47, 8.58, None),
        ("ACTION LOST", 50.19, 6.09, 67.15, 10.91, None),
        ("ACTION LOST", 47.70, 6.09, 55.01, 14.90, None),
        ("ACTION LOST", 45.87, 35.84, 79.28, 50.14, None),
        ("ACTION LOST", 54.51, 4.43, 54.35, 15.56, None),
        ("ACTION LOST", 64.99, 0.94, 70.97, 1.93, None),
        ("ACTION LOST", 87.43, 7.25, 87.43, 20.88, None),
        ("ACTION LOST", 93.25, 7.92, 119.18, 39.67, None),
        ("ACTION LOST", 99.90, 13.57, 98.90, 23.21, None),
    ],
    "Vs Dallas": [
        ("ACTION WON", 56.01, 3.43, 45.21, 8.08, None),
        ("ACTION WON", 44.04, 2.10, 38.22, 7.25, None),
        ("ACTION WON", 46.54, 11.24, 35.56, 9.91, None),
        ("ACTION WON", 41.22, 10.91, 50.03, 15.23, None),
        ("ACTION WON", 96.57, 2.26, 104.05, 28.86, None),
        ("ACTION WON", 82.28, 22.55, 106.55, 1.43, None),
        ("ACTION WON", 78.78, 21.05, 84.94, 20.72, None),
        ("ACTION WON", 75.79, 18.89, 86.60, 55.63, None),
        ("ACTION WON", 96.07, 39.00, 101.39, 39.00, None),
        ("ACTION LOST", 88.09, 12.24, 87.43, 4.26, None),
        ("ACTION LOST", 78.62, 4.76, 87.59, 1.60, None),
        ("ACTION LOST", 53.85, 1.60, 52.69, 1.10, None),
        ("ACTION LOST", 52.85, 2.93, 62.49, 13.07, None),
        ("ACTION LOST", 40.22, 22.55, 91.09, 25.54, None),
    ],
    "vs Sagoya": [
        ("ACTION WON", 116.19, 14.40, 109.54, 29.36, None),
        ("ACTION WON", 91.92, 3.43, 85.27, 7.75, None),
        ("ACTION WON", 57.51, 6.09, 56.01, 26.70, None),
        ("ACTION WON", 118.35, 1.43, 108.87, 46.82, None),
        ("ACTION WON", 103.72, 40.83, 105.05, 42.49, None),
        ("ACTION WON", 86.93, 4.76, 107.88, 31.36, None),
        ("ACTION WON", 65.82, 40.50, 79.95, 30.86, None),
        ("ACTION WON", 75.79, 8.08, 74.79, 27.53, None),
        ("ACTION WON", 74.46, 5.09, 71.64, 14.07, None),
        ("ACTION WON", 67.31, 2.10, 61.83, 10.91, None),
        ("ACTION WON", 67.65, 5.92, 51.52, 8.08, None),
        ("ACTION WON", 62.49, 2.60, 66.65, 9.41, None),
        ("ACTION WON", 47.03, 2.43, 50.03, 15.73, None),
        ("ACTION WON", 37.23, 10.24, 53.35, 12.57, None),
        ("ACTION WON", 23.59, 2.76, 32.07, 4.92, None),
        ("ACTION WON", 20.94, 14.23, 33.24, 7.25, None),
        ("ACTION WON", 14.62, 18.22, 6.64, 37.01, None),
        ("ACTION LOST", 51.19, 3.59, 117.68, 14.07, None),
        ("ACTION LOST", 65.15, 6.59, 113.86, 20.05, None),
        ("ACTION LOST", 90.92, 2.76, 94.24, 4.76, None),
        ("ACTION LOST", 97.74, 7.09, 101.56, 20.05, None),
        ("ACTION LOST", 84.44, 6.59, 91.09, 13.90, None),
    ],
    "Vs Busan Park": [
        ("ACTION WON", 114.52, 19.05, 103.72, 21.22, None),
        ("ACTION WON", 92.25, 21.88, 112.20, 24.21, None),
        ("ACTION WON", 99.90, 23.21, 90.59, 24.87, None),
        ("ACTION WON", 86.93, 2.10, 82.61, 10.74, None),
        ("ACTION WON", 85.93, 4.92, 94.41, 32.69, None),
        ("ACTION WON", 89.59, 3.26, 80.95, 26.87, None),
        ("ACTION WON", 84.27, 10.74, 76.12, 3.59, None),
        ("ACTION WON", 54.51, 2.76, 52.85, 17.56, None),
        ("ACTION WON", 56.01, 9.08, 46.04, 8.75, None),
        ("ACTION WON", 20.94, 2.43, 2.15, 7.58, None),
        ("ACTION WON", 96.90, 10.41, 111.03, 35.68, None),
        ("ACTION WON", 88.26, 33.35, 97.74, 8.08, None),
        ("ACTION WON", 51.02, 18.39, 66.48, 15.23, None),
        ("ACTION WON", 34.57, 56.12, 69.31, 5.92, None),
        ("ACTION WON", 53.52, 33.35, 65.15, 45.98, None),
        ("ACTION WON", 46.37, 51.47, 85.60, 46.48, None),
        ("ACTION WON", 88.26, 47.98, 107.21, 56.29, None),
        ("ACTION WON", 89.42, 50.31, 100.89, 65.10, None),
        ("ACTION LOST", 113.53, 9.25, 119.01, 38.67, None),
        ("ACTION LOST", 63.16, 37.34, 80.95, 37.67, None),
        ("ACTION LOST", 58.34, 16.56, 67.65, 26.04, None),
        ("ACTION LOST", 67.81, 6.59, 75.96, 31.52, None),
        ("ACTION LOST", 34.57, 57.95, 49.03, 55.63, None),
    ],
    "Vs Atlanta": [
        ("ACTION WON", 95.08, 11.57, 94.41, 0.44, None),
        ("ACTION WON", 54.68, 14.23, 49.36, 19.22, None),
        ("ACTION WON", 33.74, 0.60, 28.42, 7.42, None),
        ("ACTION WON", 38.06, 10.24, 20.27, 24.04, None),
        ("ACTION WON", 15.28, 11.41, 3.48, 30.52, None),
        ("ACTION WON", 26.25, 35.68, 32.07, 41.83, None),
        ("ACTION WON", 53.85, 44.65, 80.95, 51.30, None),
        ("ACTION WON", 72.97, 36.18, 98.23, 65.60, None),
        ("ACTION WON", 102.56, 66.76, 93.08, 47.31, None),
        ("ACTION LOST", 67.81, 69.59, 70.97, 78.73, None),
        ("ACTION LOST", 31.91, 2.43, 41.05, 13.40, None),
    ],
}

# ==========================
# Helpers
# ==========================
def has_video_value(v) -> bool:
    return pd.notna(v) and str(v).strip() != ""

def classify_action_direction(x_start, y_start, x_end, y_end) -> str:
    dx, dy    = x_end - x_start, y_end - y_start
    dist      = np.sqrt(dx**2 + dy**2)
    angle_deg = np.degrees(np.arctan2(abs(dy), dx))
    if angle_deg <= 45.0:   return "forward"
    if angle_deg >= 135.0:  return "backward"
    return "lateral" if dist > LATERAL_MIN_DIST else ("forward" if dx >= 0 else "backward")

# ==========================
# Build DataFrames
# ==========================
dfs_by_match = {}
for match_name, events in matches_data.items():
    dfm = pd.DataFrame(events, columns=["type", "x_start", "y_start", "x_end", "y_end", "video"])
    dfm["match"]           = match_name
    dfm["number"]          = np.arange(1, len(dfm) + 1)
    dfm["is_won"]          = dfm["type"].str.contains("WON", case=False)
    dfm["outcome"]         = np.where(dfm["is_won"], "successful", "failed")
    dfm["direction"]       = dfm.apply(lambda r: classify_action_direction(r.x_start, r.y_start, r.x_end, r.y_end), axis=1)
    dfm["is_forward"]      = dfm["direction"] == "forward"
    dfm["is_backward"]     = dfm["direction"] == "backward"
    dfm["is_lateral"]      = dfm["direction"] == "lateral"
    dfm["xt_start"]        = dfm.apply(lambda r: xt_value(r.x_start, r.y_start), axis=1)
    dfm["xt_end"]          = dfm.apply(lambda r: xt_value(r.x_end,   r.y_end),   axis=1)
    dfm["delta_xt"]        = np.where(dfm["outcome"] == "successful", dfm["xt_end"] - dfm["xt_start"], 0.0)
    dfm["action_distance"] = np.sqrt((dfm.x_end - dfm.x_start)**2 + (dfm.y_end - dfm.y_start)**2)
    # Distance-Difficulty bonus
    dfm["dist_bonus"]      = dfm["action_distance"].apply(distance_bonus)
    dfm["delta_xt_adj"]    = np.where(
        dfm["outcome"] == "successful",
        dfm["delta_xt"] * (1.0 + dfm["dist_bonus"]),
        0.0,
    )
    dfs_by_match[match_name] = dfm

df_all    = pd.concat(dfs_by_match.values(), ignore_index=True)
full_data = {"All Matches": df_all}
full_data.update(dfs_by_match)

# ==========================
# Stats
# ==========================
def compute_stats(df: pd.DataFrame) -> dict:
    total      = len(df)
    successful = int(df["is_won"].sum())
    accuracy   = (successful / total * 100.0) if total else 0.0

    # Raw ΔxT stats
    pos_mask  = (df["outcome"] == "successful") & (df["delta_xt"] > 0)
    pos_count = int(pos_mask.sum())
    pos_sum   = float(df.loc[pos_mask, "delta_xt"].sum())  if pos_count else 0.0
    pos_mean  = float(df.loc[pos_mask, "delta_xt"].mean()) if pos_count else 0.0
    pos_pct   = (pos_count / total * 100.0) if total else 0.0

    # Adjusted ΔxT stats
    adj_mask   = (df["outcome"] == "successful") & (df["delta_xt_adj"] > 0)
    adj_count  = int(adj_mask.sum())
    adj_sum    = float(df.loc[adj_mask, "delta_xt_adj"].sum())  if adj_count else 0.0
    adj_mean   = float(df.loc[adj_mask, "delta_xt_adj"].mean()) if adj_count else 0.0

    # Top 5 by RAW ΔxT
    top5_raw = pd.DataFrame()
    if pos_count:
        top5_raw = (
            df.loc[pos_mask]
            .sort_values("delta_xt", ascending=False)
            .head(5)[["number", "type", "action_distance", "dist_bonus",
                       "xt_start", "xt_end", "delta_xt", "delta_xt_adj"]]
            .reset_index(drop=True)
        )

    # Top 5 by ADJUSTED ΔxT
    top5_adj = pd.DataFrame()
    if adj_count:
        top5_adj = (
            df.loc[adj_mask]
            .sort_values("delta_xt_adj", ascending=False)
            .head(5)[["number", "type", "action_distance", "dist_bonus",
                       "xt_start", "xt_end", "delta_xt", "delta_xt_adj"]]
            .reset_index(drop=True)
        )

    failed_mask  = df["outcome"] == "failed"
    failed_count = int(failed_mask.sum())

    return {
        "total_actions":      total,
        "successful_actions": successful,
        "accuracy_pct":       round(accuracy, 2),
        "forward_total":      int(df["is_forward"].sum()),
        "backward_total":     int(df["is_backward"].sum()),
        "lateral_total":      int(df["is_lateral"].sum()),
        # raw
        "positive_xt_count":  pos_count,
        "positive_xt_sum":    round(pos_sum,  4),
        "positive_xt_mean":   round(pos_mean, 4),
        "positive_xt_pct":    round(pos_pct,  2),
        "top5_raw":           top5_raw,
        # adjusted
        "adj_xt_sum":         round(adj_sum,  4),
        "adj_xt_mean":        round(adj_mean, 4),
        "top5_adj":           top5_adj,
        # failed
        "failed_count":        failed_count,
        "failed_xt_lost_sum":  round(float(df.loc[failed_mask, "xt_start"].sum())  if failed_count else 0.0, 4),
        "failed_xt_lost_mean": round(float(df.loc[failed_mask, "xt_start"].mean()) if failed_count else 0.0, 4),
    }

# ==========================
# Draw functions
# ==========================
def draw_action_map(df: pd.DataFrame, title: str, top_n_highlight: int = 20,
                    use_adj: bool = False):
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1a1a2e", line_color="#ffffff", line_alpha=0.95)
    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))
    fig.set_facecolor("#1a1a2e"); fig.set_dpi(FIG_DPI)
    ax.axvline(x=FINAL_THIRD_LINE_X, color="#FFD54F", linewidth=1.0, alpha=0.20)
    ax.axvline(x=HALF_LINE_X,        color="#ffffff", linewidth=0.6, alpha=0.12, linestyle="--")

    sort_col = "delta_xt_adj" if use_adj else "delta_xt"
    top_idxs = set()
    if not df.empty:
        df_s = df[df["outcome"] == "successful"]
        if not df_s.empty:
            top_idxs = set(df_s.sort_values(sort_col, ascending=False).head(top_n_highlight).index)

    for idx, row in df.iterrows():
        if not row["is_won"]:
            color, alpha = COLOR_FAIL, 0.88
        elif idx in top_idxs:
            color, alpha = (COLOR_ADJ if use_adj else COLOR_TOP), 0.95
        else:
            color, alpha = COLOR_OTHER, 0.20

        pitch.arrows(row.x_start, row.y_start, row.x_end, row.y_end,
                     color=color, width=1.6, headwidth=2.5, headlength=2.5,
                     ax=ax, zorder=3, alpha=alpha)
        if has_video_value(row["video"]):
            pitch.scatter(row.x_start, row.y_start, s=68, marker="o", facecolors="none",
                          edgecolors="#FFD54F", linewidths=2.0, ax=ax, zorder=5, alpha=alpha)
        pitch.scatter(row.x_start, row.y_start, s=START_DOT_SIZE, marker="o",
                      color=color, edgecolors="white", linewidths=0.7, ax=ax, zorder=6, alpha=alpha)

    highlight_color = COLOR_ADJ if use_adj else COLOR_TOP
    highlight_label = f"Top ΔxT adj (dist bonus)" if use_adj else "Top ΔxT (raw)"
    legend_elements = [
        Line2D([0],[0], color=highlight_color, lw=2.5, label=highlight_label),
        Line2D([0],[0], color=COLOR_OTHER,     lw=2.5, label="Other successful"),
        Line2D([0],[0], color=COLOR_FAIL,      lw=2.5, label="Failed"),
    ]
    legend = ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(0.01, 0.99),
                       frameon=True, facecolor="#1a1a2e", edgecolor="#444466",
                       fontsize="x-small", labelspacing=0.5, borderpad=0.5)
    for txt in legend.get_texts(): txt.set_color("white")
    legend.get_frame().set_alpha(0.92)

    ax.set_title(title, fontsize=12, color="#ffffff", pad=8)
    fig.patches.append(FancyArrowPatch((0.45,0.05),(0.55,0.05), transform=fig.transFigure,
                        arrowstyle="-|>", mutation_scale=15, linewidth=2, color="#cccccc"))
    fig.text(0.5, 0.02, "Attack Direction", ha="center", va="center", fontsize=9, color="#cccccc")
    fig.tight_layout(); fig.canvas.draw()
    buf = BytesIO(); fig.savefig(buf, format="png", dpi=FIG_DPI, facecolor=fig.get_facecolor()); buf.seek(0)
    return Image.open(buf), ax, fig


def draw_corridor_heatmap(df: pd.DataFrame, title: str = "Zone Heatmap — Completed Actions"):
    df_s   = df[df["is_won"]].copy()
    x_bins = np.linspace(0.0, FIELD_X, 7)
    corridors = {"left":(LANE_LEFT_MIN,FIELD_Y), "center":(LANE_RIGHT_MAX,LANE_LEFT_MIN), "right":(0.0,LANE_RIGHT_MAX)}
    counts = {}
    for cname, (y0, y1) in corridors.items():
        arr = np.zeros(6, dtype=int)
        for i in range(6):
            x0, x1 = x_bins[i], x_bins[i+1]
            arr[i]  = int(((df_s.x_end>=x0)&(df_s.x_end<x1)&(df_s.y_end>=y0)&(df_s.y_end<y1)).sum())
        counts[cname] = arr

    vmax   = max(1, int(np.concatenate(list(counts.values())).max()))
    pitch  = Pitch(pitch_type="statsbomb", pitch_color="#1a1a2e", line_color="#ffffff", line_alpha=0.95)
    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))
    fig.set_facecolor("#1a1a2e"); fig.set_dpi(FIG_DPI)
    cmap_h = LinearSegmentedColormap.from_list("wr", ["#ffffff","#ffecec","#ffbfbf","#ff8080","#ff3b3b","#ff0000"])
    norm_h = Normalize(vmin=0, vmax=vmax); thr = max(1, vmax*0.35)

    for cname, (y0,y1) in corridors.items():
        for i, val in enumerate(counts[cname]):
            x0, x1 = x_bins[i], x_bins[i+1]
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0, facecolor=cmap_h(norm_h(val)),
                                    edgecolor=(1,1,1,0.12), linewidth=0.6, alpha=0.95, zorder=2))
            ax.text((x0+x1)/2,(y0+y1)/2,str(val),ha="center",va="center",zorder=4,fontsize=11,
                    color="#000000" if val<=thr else "#ffffff",
                    fontweight="700" if val>=vmax*0.5 else "600")

    ax.set_title(title, fontsize=12, color="#ffffff", pad=8)
    ax.axhline(y=LANE_LEFT_MIN,  color="#ffffff", lw=0.5, alpha=0.15, linestyle="--", zorder=3)
    ax.axhline(y=LANE_RIGHT_MAX, color="#ffffff", lw=0.5, alpha=0.15, linestyle="--", zorder=3)
    fig.patches.append(FancyArrowPatch((0.45,0.05),(0.55,0.05), transform=fig.transFigure,
                        arrowstyle="-|>", mutation_scale=15, linewidth=2, color="#cccccc"))
    fig.text(0.5,0.02,"Attack Direction",ha="center",va="center",fontsize=9,color="#cccccc")
    fig.tight_layout(); fig.canvas.draw()
    buf = BytesIO(); fig.savefig(buf, format="png", dpi=FIG_DPI, facecolor=fig.get_facecolor()); buf.seek(0)
    return Image.open(buf), ax, fig

# ==========================
# Layout
# ==========================
col_filters, col_field, col_stats = st.columns([0.9, 2, 1], gap="large")

with col_filters:
    st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
    st.markdown("### 🏟️ Match Selection")
    selected_match = st.selectbox("Choose the match", list(full_data.keys()), index=0)
    st.markdown('<hr class="filter-divider">', unsafe_allow_html=True)
    st.markdown("### 🎯 Action Filter")
    action_filter = st.radio(
        "Filter actions to display",
        ["All Actions","Top N actions (ΔxT)","Top N actions (ΔxT adj)",
         "Unsuccessful actions","Successful actions",
         "Positive xT only (successful)","High xT only (successful)"],
        index=0,
    )
    st.markdown("<div style='margin-top:8px; color:#e6e6e6;'>Top N / High xT controls</div>", unsafe_allow_html=True)
    top_n        = st.number_input("Top N", min_value=1, max_value=100, value=20, step=1)
    xt_threshold = st.slider("High xT threshold (ΔxT) ≥", min_value=0.0, max_value=0.5, value=0.03, step=0.005)
    st.markdown('<hr class="filter-divider">', unsafe_allow_html=True)
    st.markdown("### 📐 Distance Bonus Parameters")
    d_ref_ui     = st.slider("D_REF — min distance for bonus (m)",  5.0, 30.0, float(D_REF),   0.5)
    d_scale_ui   = st.slider("D_SCALE — log scaling factor",        5.0, 50.0, float(D_SCALE),  1.0)
    bonus_cap_ui = st.slider("BONUS_CAP — max bonus multiplier",    0.1,  1.0, float(BONUS_CAP), 0.05)
    st.markdown('</div>', unsafe_allow_html=True)

for key, default in [("heat_selection", None), ("last_match", selected_match), ("last_filter", action_filter)]:
    if key not in st.session_state:
        st.session_state[key] = default
if st.session_state["last_match"] != selected_match:
    st.session_state["heat_selection"] = None; st.session_state["last_match"] = selected_match
if st.session_state["last_filter"] != action_filter:
    st.session_state["heat_selection"] = None; st.session_state["last_filter"] = action_filter

# Recompute bonus columns with UI parameters
def recompute_bonus(df, d_ref, d_scale, bonus_cap):
    excess = np.maximum(0.0, df["action_distance"].values - d_ref)
    raw    = np.log1p(excess / d_scale)
    df = df.copy()
    df["dist_bonus"]   = np.minimum(bonus_cap, raw)
    df["delta_xt_adj"] = np.where(
        df["outcome"] == "successful",
        df["delta_xt"] * (1.0 + df["dist_bonus"]),
        0.0,
    )
    return df

use_adj_highlight = action_filter == "Top N actions (ΔxT adj)"

with col_field:
    df_base = recompute_bonus(full_data[selected_match].copy(), d_ref_ui, d_scale_ui, bonus_cap_ui)

    if action_filter == "All Actions":
        df_base = df_base.reset_index(drop=True)
    elif action_filter in ("Top N actions (ΔxT)", "Top N actions (ΔxT adj)"):
        sort_col = "delta_xt_adj" if use_adj_highlight else "delta_xt"
        df_s     = df_base[df_base["outcome"] == "successful"]
        df_base  = df_s.sort_values(sort_col, ascending=False).head(int(top_n)).reset_index(drop=True)
    elif action_filter == "Unsuccessful actions":
        df_base = df_base[df_base["outcome"] == "failed"].reset_index(drop=True)
    elif action_filter == "Successful actions":
        df_base = df_base[df_base["outcome"] == "successful"].reset_index(drop=True)
    elif action_filter == "Positive xT only (successful)":
        df_base = df_base[(df_base["outcome"]=="successful") & (df_base["delta_xt"]>0)].reset_index(drop=True)
    elif action_filter == "High xT only (successful)":
        df_base = df_base[(df_base["outcome"]=="successful") & (df_base["delta_xt"]>=float(xt_threshold))].reset_index(drop=True)

    DISPLAY_WIDTH        = 780
    pass_map_placeholder = st.empty()

    st.markdown('<h4 style="color:#ffffff; margin:6px 0 6px 0;">Zone Heatmap</h4>', unsafe_allow_html=True)
    heat_img, hax, hfig = draw_corridor_heatmap(df_base)
    heat_click = streamlit_image_coordinates(heat_img, width=DISPLAY_WIDTH)

    if heat_click is not None:
        real_w, real_h   = heat_img.size
        pixel_x  = heat_click["x"] * (real_w / heat_click["width"])
        pixel_y  = heat_click["y"] * (real_h / heat_click["height"])
        field_x, field_y = hax.transData.inverted().transform((pixel_x, real_h - pixel_y))
        x_bins   = np.linspace(0.0, FIELD_X, 7)
        ix       = max(0, min(5, np.searchsorted(x_bins, field_x, side="right") - 1))
        x0, x1   = x_bins[ix], x_bins[ix+1]
        if   field_y >= LANE_LEFT_MIN:  cname,y0,y1 = "left",   LANE_LEFT_MIN,  FIELD_Y
        elif field_y <  LANE_RIGHT_MAX: cname,y0,y1 = "right",  0.0,            LANE_RIGHT_MAX
        else:                           cname,y0,y1 = "center", LANE_RIGHT_MAX, LANE_LEFT_MIN
        st.session_state["heat_selection"] = {"ix":int(ix),"corridor":cname,
            "x0":float(x0),"x1":float(x1),"y0":float(y0),"y1":float(y1)}
    plt.close(hfig)

    with pass_map_placeholder.container():
        st.markdown('<h4 style="color:#ffffff; margin:0 0 6px 0;">Action Map</h4>', unsafe_allow_html=True)
        if st.button("Limpar filtro do quadrante", key="clear_heat_filter"):
            st.session_state["heat_selection"] = None

        df_to_draw = df_base
        if st.session_state["heat_selection"] is not None:
            sel = st.session_state["heat_selection"]
            df_to_draw = df_base[
                (df_base.x_end>=sel["x0"])&(df_base.x_end<sel["x1"])&
                (df_base.y_end>=sel["y0"])&(df_base.y_end<sel["y1"])
            ].reset_index(drop=True)

        img_obj, ax, fig = draw_action_map(
            df_to_draw, title=f"Action Map — {selected_match}",
            top_n_highlight=int(top_n), use_adj=use_adj_highlight)
        click = streamlit_image_coordinates(img_obj, width=DISPLAY_WIDTH)

    selected_action = None
    if click is not None:
        real_w, real_h   = img_obj.size
        pixel_x  = click["x"] * (real_w / click["width"])
        pixel_y  = click["y"] * (real_h / click["height"])
        field_x, field_y = ax.transData.inverted().transform((pixel_x, real_h - pixel_y))
        df_sel   = df_to_draw.copy()
        df_sel["dist"] = np.sqrt((df_sel.x_start-field_x)**2 + (df_sel.y_start-field_y)**2)
        cands    = df_sel[df_sel["dist"] < 5.0]
        if not cands.empty:
            selected_action = cands.sort_values("dist").iloc[0]
    plt.close(fig)

    if st.session_state["heat_selection"] is not None:
        sel      = st.session_state["heat_selection"]
        sel_mask = ((df_base.x_end>=sel["x0"])&(df_base.x_end<sel["x1"])&
                    (df_base.y_end>=sel["y0"])&(df_base.y_end<sel["y1"]))
        st.markdown(
            f"<div style='color:#ffffff;margin-top:6px;'><strong>Filtro:</strong> "
            f"corredor <code>{sel['corridor']}</code>, col X #{sel['ix']+1} — {int(sel_mask.sum())} ações</div>",
            unsafe_allow_html=True)

    st.divider()
    st.subheader("Selected Action")
    if selected_action is None:
        st.info("Click the start dot to inspect the action details.")
    else:
        st.success(f"Selected action: #{int(selected_action['number'])} ({selected_action['type']})")
        c1, c2 = st.columns(2)
        c1.write(f"**Start:** ({selected_action.x_start:.2f}, {selected_action.y_start:.2f})")
        c2.write(f"**End:**   ({selected_action.x_end:.2f},   {selected_action.y_end:.2f})")

        dir_emoji = {"forward":"⬆️","backward":"⬇️","lateral":"↔️"}
        t1,t2,t3 = st.columns(3)
        t1.write(f"**Direction:** {dir_emoji.get(selected_action['direction'],'')} {selected_action['direction'].capitalize()}")
        t2.write(f"**Successful:** {'✅' if selected_action['is_won'] else '❌'}")
        t3.write("")

        x1,x2,x3,x4,x5,x6 = st.columns(6)
        x1.metric("Distance",   f"{selected_action['action_distance']:.1f}m")
        x2.metric("xT Start",   f"{selected_action['xt_start']:.4f}")
        x3.metric("xT End",     f"{selected_action['xt_end']:.4f}")
        x4.metric("ΔxT",        f"{selected_action['delta_xt']:.4f}")
        x5.metric("Dist Bonus", f"+{selected_action['dist_bonus']*100:.1f}%")
        x6.metric("ΔxT adj",    f"{selected_action['delta_xt_adj']:.4f}",
                  delta=f"{selected_action['delta_xt_adj']:.4f}" if selected_action["delta_xt_adj"] != 0 else None)

        if has_video_value(selected_action["video"]):
            try: st.video(selected_action["video"])
            except Exception: st.error(f"Video not found: {selected_action['video']}")
        else:
            st.warning("No video is attached to this event.")

    with st.expander("📊 Full Actions Data Table"):
        display_cols = ["number","type","outcome","direction",
                        "x_start","y_start","x_end","y_end","action_distance",
                        "xt_start","xt_end","delta_xt","dist_bonus","delta_xt_adj"]
        st.dataframe(
            df_to_draw[display_cols].style.format({
                "x_start":"{:.2f}","y_start":"{:.2f}","x_end":"{:.2f}","y_end":"{:.2f}",
                "action_distance":"{:.1f}","xt_start":"{:.4f}","xt_end":"{:.4f}",
                "delta_xt":"{:.4f}","dist_bonus":"{:.3f}","delta_xt_adj":"{:.4f}",
            }),
            use_container_width=True, height=400,
        )

with col_stats:
    stats_safe = compute_stats(df_to_draw)

    with st.expander("General Statistics", expanded=False):
        st.markdown('<div class="stats-section-title">Overview</div>', unsafe_allow_html=True)
        r1,r2,r3 = st.columns(3)
        with r1: small_metric("Total Actions", f"{stats_safe['total_actions']}")
        with r2: small_metric("Successful",    f"{stats_safe['successful_actions']}")
        with r3: small_metric("Accuracy",      f"{stats_safe['accuracy_pct']:.1f}%")
        st.markdown("<hr style='margin:6px 0 8px 0;'>", unsafe_allow_html=True)
        st.markdown('<div class="stats-section-title">Action Directions</div>', unsafe_allow_html=True)
        d1,d2,d3 = st.columns(3)
        with d1: small_metric("⬆️ Forward",  f"{stats_safe['forward_total']}")
        with d2: small_metric("⬇️ Backward", f"{stats_safe['backward_total']}")
        with d3: small_metric("↔️ Lateral",  f"{stats_safe['lateral_total']}")

    with st.expander("Raw ΔxT Statistics", expanded=False):
        st.markdown('<div class="stats-section-title">Expected Threat — Raw ΔxT</div>', unsafe_allow_html=True)
        xt1,xt2,xt3 = st.columns(3)
        with xt1: small_metric("Σ ΔxT",          f"{stats_safe['positive_xt_sum']:.4f}")
        with xt2: small_metric("Mean ΔxT",        f"{stats_safe['positive_xt_mean']:.4f}")
        with xt3: small_metric("% Actions ΔxT>0", f"{stats_safe['positive_xt_pct']:.1f}%")
        st.markdown("<hr style='margin:6px 0 8px 0;'>", unsafe_allow_html=True)
        st.markdown('<div class="stats-section-title">Top 5 — Raw ΔxT</div>', unsafe_allow_html=True)
        if not stats_safe["top5_raw"].empty:
            st.dataframe(stats_safe["top5_raw"].style.format({
                "action_distance":"{:.1f}","dist_bonus":"{:.3f}",
                "xt_start":"{:.4f}","xt_end":"{:.4f}",
                "delta_xt":"{:.4f}","delta_xt_adj":"{:.4f}",
            }), use_container_width=True, height=220)
        else:
            st.write("No positive ΔxT actions.")

    with st.expander("📐 Adjusted ΔxT Statistics (Distance Bonus)", expanded=True):
        st.markdown('<div class="stats-section-title">Expected Threat — ΔxT adj (distance-weighted)</div>', unsafe_allow_html=True)
        ax1,ax2 = st.columns(2)
        with ax1: small_metric("Σ ΔxT adj",   f"{stats_safe['adj_xt_sum']:.4f}")
        with ax2: small_metric("Mean ΔxT adj", f"{stats_safe['adj_xt_mean']:.4f}")

        # Bonus curve preview
        st.markdown("<hr style='margin:6px 0 8px 0;'>", unsafe_allow_html=True)
        st.markdown('<div class="stats-section-title">Bonus curve preview</div>', unsafe_allow_html=True)
        d_range = np.linspace(0, 70, 200)
        excess  = np.maximum(0.0, d_range - d_ref_ui)
        b_curve = np.minimum(bonus_cap_ui, np.log1p(excess / d_scale_ui))
        fig_c, ax_c = plt.subplots(figsize=(3.2, 1.8))
        fig_c.patch.set_facecolor("#1a1a2e"); ax_c.set_facecolor("#1a1a2e")
        ax_c.plot(d_range, b_curve * 100, color=COLOR_ADJ, lw=2)
        ax_c.axvline(x=d_ref_ui, color="#ffffff", lw=0.7, linestyle="--", alpha=0.4)
        ax_c.set_xlabel("Distance (m)", color="#cccccc", fontsize=8)
        ax_c.set_ylabel("Bonus (%)",    color="#cccccc", fontsize=8)
        ax_c.tick_params(colors="#cccccc", labelsize=7)
        for spine in ax_c.spines.values(): spine.set_edgecolor("#444466")
        fig_c.tight_layout()
        buf_c = BytesIO(); fig_c.savefig(buf_c, format="png", dpi=90, facecolor="#1a1a2e"); buf_c.seek(0)
        st.image(buf_c, use_container_width=True)
        plt.close(fig_c)

        st.markdown("<hr style='margin:6px 0 8px 0;'>", unsafe_allow_html=True)
        st.markdown('<div class="stats-section-title">Top 5 — ΔxT adj</div>', unsafe_allow_html=True)
        if not stats_safe["top5_adj"].empty:
            st.dataframe(stats_safe["top5_adj"].style.format({
                "action_distance":"{:.1f}","dist_bonus":"{:.3f}",
                "xt_start":"{:.4f}","xt_end":"{:.4f}",
                "delta_xt":"{:.4f}","delta_xt_adj":"{:.4f}",
            }), use_container_width=True, height=220)
        else:
            st.write("No adjusted ΔxT actions.")

    with st.expander("Failed actions", expanded=False):
        st.markdown('<div class="stats-section-title">xT contrários</div>', unsafe_allow_html=True)
        fx1,fx2,fx3 = st.columns(3)
        with fx1: small_metric("Failed",              f"{stats_safe['failed_count']}")
        with fx2: small_metric("Σ xT start — failed", f"{stats_safe['failed_xt_lost_sum']:.4f}")
        with fx3: small_metric("Mean xT — failed",    f"{stats_safe['failed_xt_lost_mean']:.4f}")

    st.divider()
    st.caption(
        "ΔxT adj = ΔxT × (1 + bônus de distância). "
        "Bônus = min(CAP, log(1 + max(0, d−D_REF) / D_SCALE)). "
        "Ajuste D_REF, D_SCALE e BONUS_CAP no painel esquerdo."
    )
