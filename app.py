import streamlit as st

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from mplsoccer import Pitch

import pandas as pd

import numpy as np

from PIL import Image

from io import BytesIO

from matplotlib.patches import FancyArrowPatch, Rectangle

from streamlit_image_coordinates import streamlit_image_coordinates

from matplotlib.colors import Normalize, LinearSegmentedColormap

from collections import defaultdict

import math



st.set_page_config(layout='wide', page_title='Action Map - Clean (Actions + xT)')



st.markdown('''

<style>

.small-metric{padding:6px 8px;}

.small-metric .label{font-size:12px;color:#ffffff;margin-bottom:3px;opacity:.95;}

.small-metric .value{font-size:18px;font-weight:600;color:#ffffff;}

.small-metric .delta{font-size:11px;color:#e6e6e6;margin-top:4px;}

.stats-section-title{font-size:14px;font-weight:600;margin-bottom:6px;color:#ffffff;}

.streamlit-expanderHeader{color:#ffffff!important;}

.streamlit-expander{background:rgba(255,255,255,.02);}

.filter-panel{

  background:linear-gradient(168deg,rgba(30,39,56,.92) 0%,rgba(22,28,40,.97) 100%);

  border:1px solid rgba(255,255,255,.08);border-radius:14px;

  padding:24px 18px 20px 18px;

  box-shadow:0 4px 24px rgba(0,0,0,.25),0 1px 4px rgba(0,0,0,.12);

  backdrop-filter:blur(6px);}

.filter-panel h3{font-size:15px;color:#c8d6e5;letter-spacing:.5px;margin-bottom:8px;}

.filter-panel .filter-divider{border:none;border-top:1px solid rgba(255,255,255,.07);margin:14px 0;}

.stSubheader{color:#ffffff!important;}

</style>

''', unsafe_allow_html=True)



def small_metric(label, value, delta=None):

    html = (f'<div class="small-metric">'

            f'<div class="label">{label}</div>'

            f'<div class="value">{value}</div>')

    if delta is not None:

        html += f'<div class="delta">{delta}</div>'

    html += '</div>'

    st.markdown(html, unsafe_allow_html=True)



st.title('Action Map - Clean (Actions + xT)')



FIELD_X, FIELD_Y = 120.0, 80.0

HALF_LINE_X = FIELD_X / 2

FINAL_THIRD_LINE_X = 80

LANE_LEFT_MIN = 53.33

LANE_RIGHT_MAX = 26.67

NX, NY = 16, 12

LATERAL_MIN_DIST = 12.0



CMAP_ACTION = LinearSegmentedColormap.from_list(

    'xt_action',

    ['#ffffcc','#ffeda0','#fed976','#feb24c','#fd8d3c','#f03b20','#bd0026','#67000d']

)

NORM_ACTION = Normalize(vmin=0.0, vmax=1.0)



D_REF = 10.0

D_SCALE = 20.0

BONUS_CAP = 0.60



FIG_W, FIG_H = 7.9, 5.3

FIG_DPI = 110



OFFSET_GRID_X = 10

OFFSET_GRID_Y = 8



def distance_bonus(distance):

    excess = np.maximum(0.0, np.asarray(distance, dtype=float) - D_REF)

    return np.minimum(BONUS_CAP, np.log1p(excess / D_SCALE))



@st.cache_data(show_spinner=False)

def compute_xt_grid(NX=16, NY=12, sub=24,

    goal_width=11.0, penalty_depth=18.5, penalty_width=45.32,

    prox_w=0.50, central_w=0.50,

    internal_prox_power=2.8, internal_central_power=2.4, center_boost=0.20,

    FUNNEL_INFLUENCE_RANGE=35.0, FUNNEL_POWER=1.3, BASE_BOOST_WEIGHT=0.15,

    band_width_m=180.0, blur_window_m=60.0, final_blur_m=12.0,

    ANGLE_WEIGHT=0.50, ANGLE_POWER=1.4, BASE_ANGLE_WEIGHT=0.40):



    ncols_hr = NX * sub

    nrows_hr = NY * sub

    xe = np.linspace(0, FIELD_X, ncols_hr + 1)

    ye = np.linspace(0, FIELD_Y, nrows_hr + 1)

    xc = (xe[:-1] + xe[1:]) / 2

    yc_arr = (ye[:-1] + ye[1:]) / 2

    Xc, Yc = np.meshgrid(xc, yc_arr)



    xp = 0.01 + (Xc / FIELD_X) * 0.99

    yc = 1.0 - np.abs((Yc / FIELD_Y) - 0.5) * 2.0

    BASE = xp * (0.8 + 0.2 * yc)

    BASE = (BASE - BASE.min()) / (BASE.max() - BASE.min() + 1e-12)



    cy = FIELD_Y / 2.0

    fv = [(FIELD_X, cy-goal_width/2), (FIELD_X-penalty_depth, cy-penalty_width/2),

          (FIELD_X-penalty_depth, cy+penalty_width/2), (FIELD_X, cy+goal_width/2)]

    bpts = []

    for i in range(len(fv)):

        a, b = fv[i], fv[(i+1) % len(fv)]

        dx, dy = b[0] - a[0], b[1] - a[1]

        n = max(2, int(round(math.hypot(dx, dy) / 0.5)))

        for t in np.linspace(0, 1, n, endpoint=False):

            bpts.append((a[0] + dx * t, a[1] + dy * t))



    bpts = np.array(bpts)

    fX = Xc.ravel()

    fY = Yc.ravel()

    md2 = np.full(fX.size, np.inf)

    for bp in bpts:

        dx = fX - bp[0]

        dy = fY - bp[1]

        np.minimum(md2, dx*dx + dy*dy, out=md2)

    adist = np.sqrt(md2).reshape(Xc.shape)



    infl = np.clip((1 - np.clip(adist / FUNNEL_INFLUENCE_RANGE, 0, 1))**FUNNEL_POWER, 0, 1)



    D = np.hypot(FIELD_X - Xc, cy - Yc)

    prox = 1 - np.clip(D / np.hypot(FIELD_X, FIELD_Y/2), 0, 1)

    cent = 1 - np.clip(np.abs((Yc - cy) / cy), 0, 1)

    ub = np.clip((prox_w * np.clip(prox**internal_prox_power, 0, 1) +

                  central_w * np.clip(cent**internal_central_power, 0, 1)) *

                 (1 + center_boost * prox), 0, 1)



    v1x = FIELD_X - Xc

    v1y = (cy + goal_width/2) - Yc

    v2x = FIELD_X - Xc

    v2y = (cy - goal_width/2) - Yc

    ca = np.clip((v1x*v2x + v1y*v2y) / (np.hypot(v1x,v1y) * np.hypot(v2x,v2y) + 1e-12), -1, 1)

    ang = np.arccos(ca)

    af = np.clip((ang / (ang.max() + 1e-12))**ANGLE_POWER, 0, 1)



    ub = np.clip(ub * ((1-ANGLE_WEIGHT) + ANGLE_WEIGHT*af), 0, 1)

    Bc = BASE * ((1-BASE_ANGLE_WEIGHT) + BASE_ANGLE_WEIGHT*af)

    Bc = (Bc - Bc.min()) / (Bc.max() - Bc.min() + 1e-12)

    XTB = Bc + infl * BASE_BOOST_WEIGHT * ub



    pw = FIELD_X / ncols_hr

    ph = FIELD_Y / nrows_hr

    rx = max(1, int(round((blur_window_m/pw)/2)))

    ry = max(1, int(round((blur_window_m/ph)/2)))



    def blur(a, rx, ry):

        H, W = a.shape

        p = np.pad(a, ((ry,ry),(rx,rx)), mode='edge').astype(np.float64)

        ii = p.cumsum(0).cumsum(1)

        s = ii[2*ry:2*ry+H, 2*rx:2*rx+W].copy()

        s += ii[:H,:W]

        s -= ii[:H,2*rx:2*rx+W]

        s -= ii[2*ry:2*ry+H,:W]

        return s / ((2*ry+1)*(2*rx+1))



    w = 0.5 * (1 - np.cos(np.pi * np.clip(adist / band_width_m, 0, 1)))

    XTbl = w * XTB + (1-w) * blur(XTB, rx, ry)

    rf = max(1, int(round((final_blur_m/pw)/2)))

    rfy = max(1, int(round((final_blur_m/ph)/2)))

    XT = 0.85 * XTbl + 0.15 * blur(XTbl, rf, rfy)

    XT = (XT - XT.min()) / (XT.max() - XT.min() + 1e-12)



    XTc = np.zeros((NY, NX))

    for iy in range(NY):

        for ix in range(NX):

            XTc[iy, ix] = XT[iy*sub:(iy+1)*sub, ix*sub:(ix+1)*sub].mean()



    XTc = (XTc - XTc.min()) / (XTc.max() - XTc.min() + 1e-12)

    return XTc, XT



XT_GRID, _ = compute_xt_grid()



def xt_value(x, y):

    ix = int(np.clip((x/FIELD_X)*NX, 0, NX-1))

    iy = int(np.clip((y/FIELD_Y)*NY, 0, NY-1))

    return float(XT_GRID[iy, ix])



matches_data = {

    'Ali vs Vancouver': [

        ('ACTION WON',50.03,5.76,48.86,14.07,None),('ACTION WON',42.05,4.26,65.82,19.39,None),

        ('ACTION WON',53.68,12.57,39.72,28.20,None),('ACTION WON',43.88,37.17,44.54,44.65,None),

        ('ACTION WON',76.29,23.21,65.65,22.38,None),('ACTION WON',78.62,25.54,87.26,26.37,None),

        ('ACTION WON',67.48,5.76,76.96,6.42,None),('ACTION WON',61.83,3.43,111.20,9.75,None),

        ('ACTION WON',83.27,2.93,118.51,19.89,None),('ACTION WON',97.90,6.75,111.53,9.91,None),

        ('ACTION WON',114.03,1.93,107.71,12.57,None),('ACTION WON',98.23,5.59,90.09,7.58,None),

        ('ACTION WON',96.57,5.92,91.92,14.73,None),('ACTION WON',87.43,12.24,78.78,9.41,None),

        ('ACTION WON',77.62,1.93,72.30,3.93,None),('ACTION WON',79.28,5.59,70.81,2.26,None),

        ('ACTION WON',62.83,3.43,79.62,7.25,None),('ACTION WON',53.18,9.41,68.98,13.74,None),

        ('ACTION WON',51.69,4.76,40.38,8.58,None),('ACTION LOST',116.35,2.93,118.68,11.74,None),

        ('ACTION LOST',107.88,10.58,109.54,39.83,None),('ACTION LOST',86.10,3.43,87.59,4.09,None),

        ('ACTION LOST',73.46,2.43,75.13,3.43,None),('ACTION LOST',53.18,2.60,70.47,8.58,None),

        ('ACTION LOST',50.19,6.09,67.15,10.91,None),('ACTION LOST',47.70,6.09,55.01,14.90,None),

        ('ACTION LOST',45.87,35.84,79.28,50.14,None),('ACTION LOST',54.51,4.43,54.35,15.56,None),

        ('ACTION LOST',64.99,0.94,70.97,1.93,None),('ACTION LOST',87.43,7.25,87.43,20.88,None),

        ('ACTION LOST',93.25,7.92,119.18,39.67,None),('ACTION LOST',99.90,13.57,98.90,23.21,None),

    ],

    'Vs Dallas': [

        ('ACTION WON',56.01,3.43,45.21,8.08,None),('ACTION WON',44.04,2.10,38.22,7.25,None),

        ('ACTION WON',46.54,11.24,35.56,9.91,None),('ACTION WON',41.22,10.91,50.03,15.23,None),

        ('ACTION WON',96.57,2.26,104.05,28.86,None),('ACTION WON',82.28,22.55,106.55,1.43,None),

        ('ACTION WON',78.78,21.05,84.94,20.72,None),('ACTION WON',75.79,18.89,86.60,55.63,None),

        ('ACTION WON',96.07,39.00,101.39,39.00,None),('ACTION LOST',88.09,12.24,87.43,4.26,None),

        ('ACTION LOST',78.62,4.76,87.59,1.60,None),('ACTION LOST',53.85,1.60,52.69,1.10,None),

        ('ACTION LOST',52.85,2.93,62.49,13.07,None),('ACTION LOST',40.22,22.55,91.09,25.54,None),

    ],

    'vs Sagoya': [

        ('ACTION WON',116.19,14.40,109.54,29.36,None),('ACTION WON',91.92,3.43,85.27,7.75,None),

        ('ACTION WON',57.51,6.09,56.01,26.70,None),('ACTION WON',118.35,1.43,108.87,46.82,None),

        ('ACTION WON',103.72,40.83,105.05,42.49,None),('ACTION WON',86.93,4.76,107.88,31.36,None),

        ('ACTION WON',65.82,40.50,79.95,30.86,None),('ACTION WON',75.79,8.08,74.79,27.53,None),

        ('ACTION WON',74.46,5.09,71.64,14.07,None),('ACTION WON',67.31,2.10,61.83,10.91,None),

        ('ACTION WON',67.65,5.92,51.52,8.08,None),('ACTION WON',62.49,2.60,66.65,9.41,None),

        ('ACTION WON',47.03,2.43,50.03,15.73,None),('ACTION WON',37.23,10.24,53.35,12.57,None),

        ('ACTION WON',23.59,2.76,32.07,4.92,None),('ACTION WON',20.94,14.23,33.24,7.25,None),

        ('ACTION WON',14.62,18.22,6.64,37.01,None),('ACTION LOST',51.19,3.59,117.68,14.07,None),

        ('ACTION LOST',65.15,6.59,113.86,20.05,None),('ACTION LOST',90.92,2.76,94.24,4.76,None),

        ('ACTION LOST',97.74,7.09,101.56,20.05,None),('ACTION LOST',84.44,6.59,91.09,13.90,None),

    ],

    'Vs Busan Park': [

        ('ACTION WON',114.52,19.05,103.72,21.22,None),('ACTION WON',92.25,21.88,112.20,24.21,None),

        ('ACTION WON',99.90,23.21,90.59,24.87,None),('ACTION WON',86.93,2.10,82.61,10.74,None),

        ('ACTION WON',85.93,4.92,94.41,32.69,None),('ACTION WON',89.59,3.26,80.95,26.87,None),

        ('ACTION WON',84.27,10.74,76.12,3.59,None),('ACTION WON',54.51,2.76,52.85,17.56,None),

        ('ACTION WON',56.01,9.08,46.04,8.75,None),('ACTION WON',20.94,2.43,2.15,7.58,None),

        ('ACTION WON',96.90,10.41,111.03,35.68,None),('ACTION WON',88.26,33.35,97.74,8.08,None),

        ('ACTION WON',51.02,18.39,66.48,15.23,None),('ACTION WON',34.57,56.12,69.31,5.92,None),

        ('ACTION WON',53.52,33.35,65.15,45.98,None),('ACTION WON',46.37,51.47,85.60,46.48,None),

        ('ACTION WON',88.26,47.98,107.21,56.29,None),('ACTION WON',89.42,50.31,100.89,65.10,None),

        ('ACTION LOST',113.53,9.25,119.01,38.67,None),('ACTION LOST',63.16,37.34,80.95,37.67,None),

        ('ACTION LOST',58.34,16.56,67.65,26.04,None),('ACTION LOST',67.81,6.59,75.96,31.52,None),

        ('ACTION LOST',34.57,57.95,49.03,55.63,None),

    ],

    'Vs Atlanta': [

        ('ACTION WON',95.08,11.57,94.41,0.44,None),('ACTION WON',54.68,14.23,49.36,19.22,None),

        ('ACTION WON',33.74,0.60,28.42,7.42,None),('ACTION WON',38.06,10.24,20.27,24.04,None),

        ('ACTION WON',15.28,11.41,3.48,30.52,None),('ACTION WON',26.25,35.68,32.07,41.83,None),

        ('ACTION WON',53.85,44.65,80.95,51.30,None),('ACTION WON',72.97,36.18,98.23,65.60,None),

        ('ACTION WON',102.56,66.76,93.08,47.31,None),('ACTION LOST',67.81,69.59,70.97,78.73,None),

        ('ACTION LOST',31.91,2.43,41.05,13.40,None),

    ],

}



def has_video_value(v):

    return pd.notna(v) and str(v).strip() != ''



def classify_action_direction(x0, y0, x1, y1):

    dx, dy = x1 - x0, y1 - y0

    dist = np.sqrt(dx**2 + dy**2)

    ang = np.degrees(np.arctan2(abs(dy), dx))

    if ang <= 45:

        return 'forward'

    if ang >= 135:

        return 'backward'

    return 'lateral' if dist > LATERAL_MIN_DIST else ('forward' if dx >= 0 else 'backward')



def recompute_bonus(df):

    df = df.copy()

    excess = np.maximum(0.0, df['action_distance'].values - D_REF)

    df['dist_bonus'] = np.minimum(BONUS_CAP, np.log1p(excess / D_SCALE))

    df['delta_xt_adj'] = np.where(df['outcome'] == 'successful', df['delta_xt'] * (1.0 + df['dist_bonus']), 0.0)

    return df



dfs_by_match = {}

for match_name, events in matches_data.items():

    dfm = pd.DataFrame(events, columns=['type','x_start','y_start','x_end','y_end','video'])

    dfm['match'] = match_name

    dfm['number'] = np.arange(1, len(dfm) + 1)

    dfm['is_won'] = dfm['type'].str.contains('WON', case=False)

    dfm['outcome'] = np.where(dfm['is_won'], 'successful', 'failed')

    dfm['direction'] = dfm.apply(lambda r: classify_action_direction(r.x_start, r.y_start, r.x_end, r.y_end), axis=1)

    dfm['is_forward'] = dfm['direction'] == 'forward'

    dfm['is_backward'] = dfm['direction'] == 'backward'

    dfm['is_lateral'] = dfm['direction'] == 'lateral'

    dfm['xt_start'] = dfm.apply(lambda r: xt_value(r.x_start, r.y_start), axis=1)

    dfm['xt_end'] = dfm.apply(lambda r: xt_value(r.x_end, r.y_end), axis=1)

    dfm['delta_xt'] = np.where(dfm['outcome']=='successful', dfm['xt_end'] - dfm['xt_start'], 0.0)

    dfm['action_distance'] = np.sqrt((dfm.x_end-dfm.x_start)**2 + (dfm.y_end-dfm.y_start)**2)

    dfm['dist_bonus'] = distance_bonus(dfm['action_distance'].values)

    dfm['delta_xt_adj'] = np.where(dfm['outcome']=='successful', dfm['delta_xt'] * (1 + dfm['dist_bonus']), 0.0)

    dfs_by_match[match_name] = dfm



df_all = pd.concat(dfs_by_match.values(), ignore_index=True)

full_data = {'All Matches': df_all}

full_data.update(dfs_by_match)



def compute_stats(df):

    total = len(df)

    successful = int(df['is_won'].sum())

    accuracy = (successful / total * 100) if total else 0.0



    succ_mask = df['outcome'] == 'successful'

    sum_delta_xt = float(df.loc[succ_mask, 'delta_xt_adj'].sum()) if succ_mask.any() else 0.0



    pos_mask = succ_mask & (df['delta_xt_adj'] > 0)

    pos_count = int(pos_mask.sum())

    pos_sum = float(df.loc[pos_mask, 'delta_xt_adj'].sum()) if pos_count else 0.0

    pos_mean = float(df.loc[pos_mask, 'delta_xt_adj'].mean()) if pos_count else 0.0

    pos_pct = (pos_count / total * 100) if total else 0.0



    top10_df = (df.loc[pos_mask].sort_values('delta_xt_adj', ascending=False).head(10)) if pos_count else pd.DataFrame()

    top10_sum = float(top10_df['delta_xt_adj'].sum()) if not top10_df.empty else 0.0

    top10_mean = float(top10_df['delta_xt_adj'].mean()) if not top10_df.empty else 0.0



    xt_end_mean = float(df.loc[succ_mask, 'xt_end'].mean()) if succ_mask.any() else 0.0

    xt_end_sum = float(df.loc[succ_mask, 'xt_end'].sum()) if succ_mask.any() else 0.0



    failed_mask = df['outcome'] == 'failed'

    failed_count = int(failed_mask.sum())

    failed_xt_sum = float(df.loc[failed_mask, 'xt_start'].sum()) if failed_count else 0.0

    failed_xt_mean = float(df.loc[failed_mask, 'xt_start'].mean()) if failed_count else 0.0



    return {

        'total_actions': total,

        'successful_actions': successful,

        'accuracy_pct': round(accuracy, 2),

        'forward_total': int(df['is_forward'].sum()),

        'backward_total': int(df['is_backward'].sum()),

        'lateral_total': int(df['is_lateral'].sum()),

        'sum_delta_xt': round(sum_delta_xt, 4),

        'positive_xt_count': pos_count,

        'pos_sum': round(pos_sum, 4),

        'pos_mean': round(pos_mean, 4),

        'pos_pct': round(pos_pct, 2),

        'top10_sum': round(top10_sum, 4),

        'top10_mean': round(top10_mean, 4),

        'xt_end_mean': round(xt_end_mean, 4),

        'xt_end_sum': round(xt_end_sum, 4),

        'failed_count': failed_count,

        'failed_xt_sum': round(failed_xt_sum, 4),

        'failed_xt_mean': round(failed_xt_mean, 4),

    }



def compute_parallel_offsets(df, offset_step=1.5):

    n = len(df)

    xs0 = df['x_start'].values.copy().astype(float)

    ys0 = df['y_start'].values.copy().astype(float)

    xs1 = df['x_end'].values.copy().astype(float)

    ys1 = df['y_end'].values.copy().astype(float)



    if n == 0:

        return xs0, ys0, xs1, ys1



    sx = np.clip((xs0 / FIELD_X * OFFSET_GRID_X).astype(int), 0, OFFSET_GRID_X - 1)

    sy = np.clip((ys0 / FIELD_Y * OFFSET_GRID_Y).astype(int), 0, OFFSET_GRID_Y - 1)

    ex = np.clip((xs1 / FIELD_X * OFFSET_GRID_X).astype(int), 0, OFFSET_GRID_X - 1)

    ey = np.clip((ys1 / FIELD_Y * OFFSET_GRID_Y).astype(int), 0, OFFSET_GRID_Y - 1)



    groups = defaultdict(list)

    for i in range(n):

        groups[(sx[i], sy[i], ex[i], ey[i])].append(i)



    for _, idxs in groups.items():

        m = len(idxs)

        if m == 1:

            continue



        mean_dx = float(np.mean(xs1[idxs] - xs0[idxs]))

        mean_dy = float(np.mean(ys1[idxs] - ys0[idxs]))

        length = math.hypot(mean_dx, mean_dy)



        if length < 1e-6:

            perp_x, perp_y = 0.0, 1.0

        else:

            perp_x = -mean_dy / length

            perp_y = mean_dx / length



        half = (m - 1) / 2.0

        offsets = [(j - half) * offset_step for j in range(m)]



        for j, i in enumerate(idxs):

            d = offsets[j]

            xs0[i] += perp_x * d

            ys0[i] += perp_y * d

            xs1[i] += perp_x * d

            ys1[i] += perp_y * d



    return xs0, ys0, xs1, ys1



def _action_style(is_won, is_top):

    if not is_won:

        return 1.0, ':', 0.28, 12, 18

    if is_top:

        return 2.0, '--', 0.95, 22, 62

    return 1.3, '--', 0.42, 14, 32



def draw_action_map(df, title, top_n_highlight=20, offset_step=1.5):

    pitch = Pitch(pitch_type='statsbomb', pitch_color='#1a1a2e', line_color='#ffffff', line_alpha=0.95)

    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))

    fig.set_facecolor('#1a1a2e')

    fig.set_dpi(FIG_DPI)

    ax.axvline(x=FINAL_THIRD_LINE_X, color='#FFD54F', lw=1.0, alpha=0.20)

    ax.axvline(x=HALF_LINE_X, color='#ffffff', lw=0.6, alpha=0.12, linestyle='--')



    top_idxs = set()

    if not df.empty:

        df_s = df[df['outcome'] == 'successful']

        if not df_s.empty:

            top_idxs = set(df_s.sort_values('delta_xt_adj', ascending=False).head(top_n_highlight).index)



    xs0_off, ys0_off, xs1_off, ys1_off = compute_parallel_offsets(df, offset_step=offset_step)



    def draw_row(row, pos):

        color = CMAP_ACTION(NORM_ACTION(float(row['xt_end'])))

        is_top = row.name in top_idxs

        lw, ls, alpha, s_start, s_end = _action_style(row['is_won'], is_top)

        ox0, oy0 = xs0_off[pos], ys0_off[pos]

        ox1, oy1 = xs1_off[pos], ys1_off[pos]



        ax.plot([ox0, ox1], [oy0, oy1], color=color, linestyle=ls, linewidth=lw, alpha=alpha, zorder=3, solid_capstyle='round')

        pitch.scatter(ox0, oy0, s=s_start, marker='o', facecolors='none', edgecolors=color, linewidths=1.2 if is_top else 0.9, ax=ax, zorder=5, alpha=alpha)

        pitch.scatter(ox1, oy1, s=s_end, marker='D', facecolors=color, edgecolors='white', linewidths=0.7 if is_top else 0.4, ax=ax, zorder=6, alpha=alpha)



        if has_video_value(row['video']):

            pitch.scatter(ox0, oy0, s=s_start+40, marker='o', facecolors='none', edgecolors='#FFD54F', linewidths=1.8, ax=ax, zorder=7, alpha=alpha)



    for pos, (idx, row) in enumerate(df.iterrows()):

        if idx not in top_idxs:

            draw_row(row, pos)

    for pos, (idx, row) in enumerate(df.iterrows()):

        if idx in top_idxs:

            draw_row(row, pos)



    ax.set_title(title, fontsize=12, color='#ffffff', pad=8)



    legend_items = [

        ax.scatter([], [], s=18, marker='o', facecolors='none', edgecolors='#fd8d3c', linewidths=1.2, label='Origin'),

        ax.scatter([], [], s=45, marker='D', facecolors='#fd8d3c', edgecolors='white', linewidths=0.7, label='Destination'),

        ax.plot([], [], color='#fd8d3c', linestyle='--', lw=2.0, label=f'Top {top_n_highlight}')[0],

        ax.plot([], [], color='#aaaaaa', linestyle=':', lw=1.0, label='Failed')[0],

    ]

    legend = ax.legend(handles=legend_items, loc='upper left', bbox_to_anchor=(0.01, 0.99), frameon=True, facecolor='#1a1a2e', edgecolor='#444466', fontsize='xx-small', labelspacing=0.35, borderpad=0.5, handletextpad=0.4)

    for t in legend.get_texts():

        t.set_color('white')

    legend.get_frame().set_alpha(0.90)



    sm = plt.cm.ScalarMappable(cmap=CMAP_ACTION, norm=NORM_ACTION)

    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, shrink=0.72)

    cbar.set_label('xT end', color='#cccccc', fontsize=7, labelpad=3)

    cbar.ax.yaxis.set_tick_params(color='#cccccc', labelsize=6)

    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#cccccc')



    fig.patches.append(FancyArrowPatch((0.44, 0.04), (0.54, 0.04), transform=fig.transFigure, arrowstyle='-|>', mutation_scale=13, linewidth=1.8, color='#cccccc'))

    fig.text(0.49, 0.015, 'Attack Direction', ha='center', va='center', fontsize=8, color='#cccccc')



    fig.tight_layout()

    fig.canvas.draw()

    buf = BytesIO()

    fig.savefig(buf, format='png', dpi=FIG_DPI, facecolor=fig.get_facecolor())

    buf.seek(0)

    return Image.open(buf), ax, fig



def draw_corridor_heatmap(df, title='Zone Heatmap - Completed Actions'):

    df_s = df[df['is_won']].copy()

    x_bins = np.linspace(0, FIELD_X, 7)

    corridors = {'left': (LANE_LEFT_MIN, FIELD_Y), 'center': (LANE_RIGHT_MAX, LANE_LEFT_MIN), 'right': (0.0, LANE_RIGHT_MAX)}

    counts = {}

    for cname, (y0, y1) in corridors.items():

        arr = np.zeros(6, dtype=int)

        for i in range(6):

            x0, x1 = x_bins[i], x_bins[i+1]

            arr[i] = int(((df_s.x_end >= x0) & (df_s.x_end < x1) & (df_s.y_end >= y0) & (df_s.y_end < y1)).sum())

        counts[cname] = arr



    vmax = max(1, int(np.concatenate(list(counts.values())).max()))

    pitch = Pitch(pitch_type='statsbomb', pitch_color='#1a1a2e', line_color='#ffffff', line_alpha=0.95)

    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))

    fig.set_facecolor('#1a1a2e')

    fig.set_dpi(FIG_DPI)



    cmap_h = LinearSegmentedColormap.from_list('wr', ['#ffffff','#ffecec','#ffbfbf','#ff8080','#ff3b3b','#ff0000'])

    norm_h = Normalize(vmin=0, vmax=vmax)

    thr = max(1, vmax * 0.35)



    for cname, (y0, y1) in corridors.items():

        for i, val in enumerate(counts[cname]):

            x0, x1 = x_bins[i], x_bins[i+1]

            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, facecolor=cmap_h(norm_h(val)), edgecolor=(1,1,1,0.12), lw=0.6, alpha=0.95, zorder=2))

            ax.text((x0+x1)/2, (y0+y1)/2, str(val), ha='center', va='center', zorder=4, fontsize=11, color='#000000' if val <= thr else '#ffffff', fontweight='700' if val >= vmax*0.5 else '600')



    ax.set_title(title, fontsize=12, color='#ffffff', pad=8)

    ax.axhline(y=LANE_LEFT_MIN, color='#ffffff', lw=0.5, alpha=0.15, linestyle='--', zorder=3)

    ax.axhline(y=LANE_RIGHT_MAX, color='#ffffff', lw=0.5, alpha=0.15, linestyle='--', zorder=3)

    fig.patches.append(FancyArrowPatch((0.44, 0.04), (0.54, 0.04), transform=fig.transFigure, arrowstyle='-|>', mutation_scale=13, linewidth=1.8, color='#cccccc'))

    fig.text(0.49, 0.015, 'Attack Direction', ha='center', va='center', fontsize=8, color='#cccccc')



    fig.tight_layout()

    fig.canvas.draw()

    buf = BytesIO()

    fig.savefig(buf, format='png', dpi=FIG_DPI, facecolor=fig.get_facecolor())

    buf.seek(0)

    return Image.open(buf), ax, fig



def render_top10(df, title='Top 10 - ΔxT (Adj.) and xT End'):

    st.markdown(f'<h4 style="color:#ffffff;margin:0 0 6px 0;">{title}</h4>', unsafe_allow_html=True)

    df_s = df[(df['outcome'] == 'successful') & (df['delta_xt_adj'] > 0)]

    if df_s.empty:

        st.caption('No successful actions with positive ΔxT.')

        return



    top = df_s.sort_values('delta_xt_adj', ascending=False).head(10).reset_index(drop=True)

    rows_html = ''

    for i, row in top.iterrows():

        rank = i + 1

        color = matplotlib.colors.to_hex(CMAP_ACTION(NORM_ACTION(float(row['xt_end']))))

        dot = (f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:{color};vertical-align:middle;margin-right:5px;border:1px solid rgba(255,255,255,.4);"></span>')

        rows_html += (

            f"<tr style='border-bottom:1px solid rgba(255,255,255,.05);'>"

            f"<td style='color:#888;text-align:center;padding:4px 8px;font-size:11px;'>#{rank}</td>"

            f"<td style='color:#ccc;text-align:center;padding:4px 8px;font-size:11px;'>{int(row['number'])}</td>"

            f"<td style='color:#fff;text-align:right;padding:4px 8px;font-weight:700;font-size:12px;'>{row['delta_xt_adj']:.4f}</td>"

            f"<td style='text-align:center;padding:4px 8px;'>{dot}<span style='color:#fff;font-size:12px;'>{row['xt_end']:.4f}</span></td>"

            '</tr>'

        )



    table_html = f"""

    <table style='width:100%;border-collapse:collapse;background:rgba(255,255,255,.03);border-radius:8px;overflow:hidden;margin-bottom:6px;'>

      <thead>

        <tr style='background:rgba(255,255,255,.06);'>

          <th style='color:#aaa;padding:5px 8px;text-align:center;font-weight:500;font-size:11px;'>Rank</th>

          <th style='color:#aaa;padding:5px 8px;text-align:center;font-weight:500;font-size:11px;'>#</th>

          <th style='color:#aaa;padding:5px 8px;text-align:right;font-weight:500;font-size:11px;'>ΔxT (Adj.)</th>

          <th style='color:#aaa;padding:5px 8px;text-align:center;font-weight:500;font-size:11px;'>xT End</th>

        </tr>

      </thead>

      <tbody>{rows_html}</tbody>

    </table>

    """

    st.markdown(table_html, unsafe_allow_html=True)



def build_match_metrics(dfs_by_match):

    rows = []

    for match_name, dfm in dfs_by_match.items():

        dfr = recompute_bonus(dfm.copy())

        s = compute_stats(dfr)

        rows.append({

            'match': match_name,

            'sum_delta_xt': s['sum_delta_xt'],

            'pos_pct': s['pos_pct'],

            'top10_sum': s['top10_sum'],

            'top10_mean': s['top10_mean'],

            'xt_end_sum': s['xt_end_sum'],

            'xt_end_mean': s['xt_end_mean'],

            'failed_xt_sum': s['failed_xt_sum'],

            'failed_xt_mean': s['failed_xt_mean'],

        })

    return pd.DataFrame(rows)



def render_direction_cards(stats):

    cards = [

        ('Forward', '&rarr;', int(stats['forward_total']), '#34d399'),

        ('Backward', '&larr;', int(stats['backward_total']), '#f97316'),

        ('Lateral', '&harr;', int(stats['lateral_total']), '#60a5fa'),

    ]

    html = '<div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;">'

    for label, arrow, value, color in cards:

        html += (

            '<div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);'

            'border-radius:10px;padding:8px 9px;">'

            f'<div style="display:flex;align-items:center;justify-content:space-between;'

            f'font-size:11px;color:#cbd5e1;"><span>{label}</span><span style="color:{color};font-size:14px;">{arrow}</span></div>'

            f'<div style="font-size:18px;font-weight:700;color:#ffffff;line-height:1.1;">{value}</div>'

            '</div>'

        )

    html += '</div>'

    st.markdown(html, unsafe_allow_html=True)


def plot_metric_line(metrics_df, key, label):

    if metrics_df.empty:

        st.info('No data available for this chart.')

        return


    y = metrics_df[key].astype(float).to_numpy()

    x = np.arange(len(y), dtype=float)

    labels = metrics_df['match'].tolist()


    if len(x) > 1:

        x_dense = np.linspace(x.min(), x.max(), max(260, len(x) * 40))

        y_dense = np.interp(x_dense, x, y)

        if len(x) > 2:

            kernel = np.array([1, 2, 3, 2, 1], dtype=float)

            kernel /= kernel.sum()

            y_dense = np.convolve(y_dense, kernel, mode='same')

    else:

        x_dense, y_dense = x, y


    base = min(0.0, float(y.min()))


    fig, ax = plt.subplots(figsize=(8.2, 3.6), dpi=220)

    fig.patch.set_facecolor('#0b1220')

    ax.set_facecolor('#101827')


    ax.fill_between(x_dense, y_dense, base, color='#0ea5e9', alpha=0.18, zorder=1)

    ax.plot(x_dense, y_dense, color='#7dd3fc', linewidth=9, alpha=0.08, solid_capstyle='round', zorder=2)

    ax.plot(x_dense, y_dense, color='#38bdf8', linewidth=3.2, solid_capstyle='round', zorder=3)

    ax.scatter(x, y, s=62, color='#22d3ee', edgecolors='white', linewidths=1.0, zorder=4)


    y_avg = float(np.mean(y))

    ax.axhline(y_avg, color='#fbbf24', linestyle=(0, (4, 4)), linewidth=1.2, alpha=0.75, zorder=2)

    ax.text(

        x.max() + 0.03,

        y_avg,

        f' avg: {y_avg:.2f}',

        color='#fcd34d',

        fontsize=8,

        va='center',

        ha='left',

        bbox=dict(boxstyle='round,pad=0.2', fc='#1f2937', ec='none', alpha=0.8),

    )


    for xi, yi in zip(x, y):

        ax.text(

            xi,

            yi,

            f'{yi:.2f}',

            color='#e2e8f0',

            fontsize=8,

            ha='center',

            va='bottom',

            bbox=dict(boxstyle='round,pad=0.18', fc='#0f172a', ec='none', alpha=0.72),

            zorder=5,

        )


    for spine in ax.spines.values():

        spine.set_visible(False)


    ax.set_xticks(x)

    ax.set_xticklabels(labels, rotation=18, ha='right', color='#cbd5e1', fontsize=9)

    ax.tick_params(axis='y', colors='#cbd5e1', labelsize=9)

    ax.set_ylabel(label, color='#bfdbfe', fontsize=10)

    ax.set_title(f'Match-by-Match Trend - {label}', loc='left', color='#f8fafc', fontsize=14, fontweight='700', pad=12)

    ax.grid(axis='y', color='#94a3b8', alpha=0.24, linestyle='--', linewidth=0.8)

    ax.grid(axis='x', color='#94a3b8', alpha=0.12, linestyle=':', linewidth=0.6)

    ax.margins(x=0.03)

    fig.tight_layout(pad=1.4)


    st.pyplot(fig, use_container_width=False)

    plt.close(fig)



for key, default in [

    ('heat_selection', None),

    ('last_match', 'All Matches'),

    ('last_filter', 'All Actions'),

    ('selected_action', None),

]:

    if key not in st.session_state:

        st.session_state[key] = default



tab_maps, tab_stats = st.tabs(['Maps', 'Stats'])



with tab_maps:

    col_filters, col_field, col_events = st.columns([0.95, 2.15, 1.2], gap='large')



    with col_filters:

        st.markdown('<div class="filter-panel">', unsafe_allow_html=True)

        st.markdown('### Match Selection')

        selected_match = st.selectbox('Choose the match', list(full_data.keys()), index=0)

        st.markdown('<hr class="filter-divider">', unsafe_allow_html=True)



        st.markdown('### Action Filter')

        action_filter = st.radio('Filter actions to display', [

            'All Actions',

            'Top N Actions (ΔxT)',

            'Unsuccessful Actions',

            'Successful Actions',

            'Positive xT only',

        ], index=0)

        top_n = st.number_input('Top N', min_value=1, max_value=100, value=20, step=1)

        st.markdown('</div>', unsafe_allow_html=True)



    if st.session_state['last_match'] != selected_match:

        st.session_state['heat_selection'] = None

        st.session_state['last_match'] = selected_match

    if st.session_state['last_filter'] != action_filter:

        st.session_state['heat_selection'] = None

        st.session_state['last_filter'] = action_filter



    df_base = recompute_bonus(full_data[selected_match].copy())

    if action_filter == 'All Actions':

        df_base = df_base.reset_index(drop=True)

    elif action_filter == 'Top N Actions (ΔxT)':

        df_s = df_base[df_base['outcome'] == 'successful']

        df_base = df_s.sort_values('delta_xt_adj', ascending=False).head(int(top_n)).reset_index(drop=True)

    elif action_filter == 'Unsuccessful Actions':

        df_base = df_base[df_base['outcome'] == 'failed'].reset_index(drop=True)

    elif action_filter == 'Successful Actions':

        df_base = df_base[df_base['outcome'] == 'successful'].reset_index(drop=True)

    elif action_filter == 'Positive xT only':

        df_base = df_base[(df_base['outcome'] == 'successful') & (df_base['delta_xt'] > 0)].reset_index(drop=True)



    with col_field:

        DISPLAY_WIDTH = 760



        if st.button('Clear zone filter', key='clear_heat_filter'):

            st.session_state['heat_selection'] = None



        df_to_draw = df_base

        if st.session_state['heat_selection'] is not None:

            sel = st.session_state['heat_selection']

            df_to_draw = df_base[(df_base.x_end >= sel['x0']) & (df_base.x_end < sel['x1']) & (df_base.y_end >= sel['y0']) & (df_base.y_end < sel['y1'])].reset_index(drop=True)



        st.markdown('<h4 style="color:#ffffff;margin:4px 0 3px 0;">Action Map</h4>', unsafe_allow_html=True)

        img_obj, ax, fig = draw_action_map(df_to_draw, title=f'Action Map - {selected_match}', top_n_highlight=int(top_n), offset_step=1.5)

        click = streamlit_image_coordinates(img_obj, width=DISPLAY_WIDTH)



        if click is not None:

            rw, rh = img_obj.size

            px = click['x'] * (rw / click['width'])

            py = click['y'] * (rh / click['height'])

            fx, fy = ax.transData.inverted().transform((px, rh - py))

            df_sel = df_to_draw.copy()

            df_sel['dist'] = np.sqrt((df_sel.x_start - fx)**2 + (df_sel.y_start - fy)**2)

            cands = df_sel[df_sel['dist'] < 5.0]

            if not cands.empty:

                st.session_state['selected_action'] = cands.sort_values('dist').iloc[0]



        plt.close(fig)



        st.markdown('<h4 style="color:#ffffff;margin:8px 0 4px 0;">Zone Heatmap</h4>', unsafe_allow_html=True)

        heat_img, hax, hfig = draw_corridor_heatmap(df_base)

        heat_click = streamlit_image_coordinates(heat_img, width=DISPLAY_WIDTH)



        if heat_click is not None:

            rw, rh = heat_img.size

            px = heat_click['x'] * (rw / heat_click['width'])

            py = heat_click['y'] * (rh / heat_click['height'])

            fx, fy = hax.transData.inverted().transform((px, rh - py))

            xb = np.linspace(0, FIELD_X, 7)

            ix = max(0, min(5, np.searchsorted(xb, fx, side='right') - 1))

            x0, x1 = xb[ix], xb[ix+1]

            if fy >= LANE_LEFT_MIN:

                cn, y0, y1 = 'left', LANE_LEFT_MIN, FIELD_Y

            elif fy < LANE_RIGHT_MAX:

                cn, y0, y1 = 'right', 0.0, LANE_RIGHT_MAX

            else:

                cn, y0, y1 = 'center', LANE_RIGHT_MAX, LANE_LEFT_MIN

            st.session_state['heat_selection'] = {'ix': int(ix), 'corridor': cn, 'x0': float(x0), 'x1': float(x1), 'y0': float(y0), 'y1': float(y1)}



        plt.close(hfig)



        if st.session_state['heat_selection'] is not None:

            sel = st.session_state['heat_selection']

            smsk = ((df_base.x_end >= sel['x0']) & (df_base.x_end < sel['x1']) & (df_base.y_end >= sel['y0']) & (df_base.y_end < sel['y1']))

            st.markdown(f"<div style='color:#ffffff;margin-top:4px;'><strong>Filter:</strong> corridor <code>{sel['corridor']}</code>, X column #{sel['ix']+1} - {int(smsk.sum())} actions</div>", unsafe_allow_html=True)



        with st.expander('Full Actions Data Table'):

            dcols = ['number','type','outcome','direction','x_start','y_start','x_end','y_end','action_distance','xt_start','xt_end','delta_xt','dist_bonus','delta_xt_adj']

            st.dataframe(df_to_draw[dcols].style.format({

                'x_start':'{:.2f}','y_start':'{:.2f}','x_end':'{:.2f}','y_end':'{:.2f}',

                'action_distance':'{:.1f}','xt_start':'{:.4f}','xt_end':'{:.4f}',

                'delta_xt':'{:.4f}','dist_bonus':'{:.3f}','delta_xt_adj':'{:.4f}'

            }), use_container_width=True, height=380)



    with col_events:

        st.subheader('Event Panel')

        selected_action = st.session_state.get('selected_action', None)

        if selected_action is None:

            st.info('Click the origin ring on the map to open the event.')

        else:

            act_color = matplotlib.colors.to_hex(CMAP_ACTION(NORM_ACTION(float(selected_action['xt_end']))))

            st.markdown(

                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'

                f'<span style="display:inline-block;width:13px;height:13px;border-radius:50%;background:{act_color};border:2px solid #fff;"></span>'

                f'<strong style="color:#fff;">Action #{int(selected_action["number"])} - {selected_action["type"]}</strong></div>',

                unsafe_allow_html=True

            )

            st.write(f'**Start:** ({selected_action.x_start:.2f}, {selected_action.y_start:.2f})')

            st.write(f'**End:** ({selected_action.x_end:.2f}, {selected_action.y_end:.2f})')

            st.write(f'**Direction:** {selected_action["direction"].capitalize()}')

            st.write(f'**Successful:** {"Yes" if selected_action["is_won"] else "No"}')

            st.metric('Distance', f'{selected_action["action_distance"]:.1f}m')

            st.metric('xT Start', f'{selected_action["xt_start"]:.4f}')

            st.metric('xT End', f'{selected_action["xt_end"]:.4f}')

            st.metric('ΔxT', f'{selected_action["delta_xt"]:.4f}')

            st.metric('Dist Bonus', f'+{selected_action["dist_bonus"]*100:.1f}%')

            st.metric('ΔxT (Adj.)', f'{selected_action["delta_xt_adj"]:.4f}')

            if has_video_value(selected_action['video']):

                try:

                    st.video(selected_action['video'])

                except Exception:

                    st.error(f'Video not found: {selected_action["video"]}')

            else:

                st.warning('No video attached to this event.')



with tab_stats:

    st.subheader('Stats (General, Advanced)')



    selected_match_stats = st.selectbox('Match for Stats', list(full_data.keys()), index=0, key='stats_match_select')

    stats_df = recompute_bonus(full_data[selected_match_stats].copy())

    stats = compute_stats(stats_df)



    col_left, col_right = st.columns([1.02, 1.25], gap='large')



    with col_left:

        render_top10(stats_df, title='Top 10 ΔxT (Selected Match)')



    with col_right:

        with st.expander('General Statistics', expanded=True):

            st.markdown('<div class="stats-section-title">Overview</div>', unsafe_allow_html=True)

            r1, r2, r3 = st.columns(3)

            with r1: small_metric('Total Actions', f"{stats['total_actions']}")

            with r2: small_metric('Successful', f"{stats['successful_actions']}")

            with r3: small_metric('Accuracy', f"{stats['accuracy_pct']:.1f}%")

            st.markdown('<hr style="margin:6px 0 8px 0;">', unsafe_allow_html=True)

            st.markdown('<div class="stats-section-title">Directions</div>', unsafe_allow_html=True)

            render_direction_cards(stats)



        with st.expander('Advanced Statistics', expanded=True):

            st.markdown('<div class="stats-section-title">ΔxT</div>', unsafe_allow_html=True)

            a1, a2, a3 = st.columns(3)

            with a1: small_metric('Σ ΔxT', f"{stats['sum_delta_xt']:.2f}")

            with a2: small_metric('% Positive', f"{stats['pos_pct']:.2f}%")

            with a3: small_metric('Avg. Positive', f"{stats['pos_mean']:.2f}")



            st.markdown('<hr style="margin:6px 0 8px 0;">', unsafe_allow_html=True)

            st.markdown('<div class="stats-section-title">Top 10 / End xT / Failed</div>', unsafe_allow_html=True)

            b1, b2, b3, b4 = st.columns(4)

            with b1: small_metric('Σ Top10', f"{stats['top10_sum']:.2f}")

            with b2: small_metric('Avg. Top10', f"{stats['top10_mean']:.2f}")

            with b3: small_metric('Σ End xT', f"{stats['xt_end_sum']:.2f}")

            with b4: small_metric('Avg. End xT', f"{stats['xt_end_mean']:.2f}")



            c1, c2 = st.columns(2)

            with c1: small_metric('Σ xT Start (Failed)', f"{stats['failed_xt_sum']:.2f}")

            with c2: small_metric('Avg. xT (Failed)', f"{stats['failed_xt_mean']:.2f}")



    st.markdown('<h4 style="color:#ffffff;margin:12px 0 6px 0;">Match Trend Line Chart</h4>', unsafe_allow_html=True)

    metric_options = {

        'Σ ΔxT': ('sum_delta_xt', 'Σ ΔxT'),

        '% Positive ΔxT': ('pos_pct', '% Positive ΔxT'),

        'Σ Top 10 ΔxT': ('top10_sum', 'Σ Top 10 ΔxT'),

        'Avg. Top 10 ΔxT': ('top10_mean', 'Avg. Top 10 ΔxT'),

        'Σ End xT': ('xt_end_sum', 'Σ End xT'),

        'Avg. End xT': ('xt_end_mean', 'Avg. End xT'),

        'Σ xT Start (Failed)': ('failed_xt_sum', 'Σ xT Start (Failed)'),

        'Avg. xT (Failed)': ('failed_xt_mean', 'Avg. xT (Failed)'),

    }



    metric_label = st.selectbox('Select a metric for the chart', list(metric_options.keys()), index=0)

    metric_key, metric_name = metric_options[metric_label]

    match_metrics_df = build_match_metrics(dfs_by_match)

    plot_metric_line(match_metrics_df, metric_key, metric_name)



    st.caption('Mapping: origin (ring), destination (diamond), color by xT End, and parallel lines for proximity-based groups.')
