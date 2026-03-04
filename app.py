import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="F1 Race Finish Predictor", page_icon="🏎️", layout="wide")

# ── Inject global CSS ─────────────────────────────────────────────────────────
with open("model.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div style="display:flex;align-items:center;gap:1.2rem;margin-bottom:0.3rem">
        <img
            src="https://www.formula1.com/etc/designs/fom-website/images/f1_logo.svg"
            alt="F1 Logo"
            style="height:30px;width:auto;object-fit:contain;filter:drop-shadow(0 0 4px rgba(225,6,0,0.5))"
        />
        <h1 style="margin:0">RACE FINISH <span>PREDICTOR</span></h1>
    </div>
    <p>Classification · 2019–2024 Season Data · Podium / Points / No Points / DNF</p>
</div>
""", unsafe_allow_html=True)

# ── Embedded CSS for components.html iframes ─────────────────────────────────
EMBED_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Barlow:wght@300;400;600&display=swap');
*{box-sizing:border-box;margin:0;padding:0;}
body{background:transparent;font-family:'Barlow',sans-serif;color:#e8e8e8;}

.cards-wrap{display:flex;gap:10px;}
.card{flex:1;background:#111117;border:1px solid #333;border-radius:8px;padding:1rem 0.7rem;text-align:center;}
.card.podium{border:2px solid #ffd700;}
.card.points{border:2px solid #00c48c;}
.card.nopoints{border:2px solid #e10600;}
.card.glow{box-shadow:0 0 18px rgba(225,6,0,0.3);}

.model-tag{font-size:0.7rem;letter-spacing:1.5px;color:#888;text-transform:uppercase;margin-bottom:0.4rem;}
.outcome-lbl{font-family:'Bebas Neue',sans-serif;font-size:1.45rem;letter-spacing:1px;margin:0.2rem 0;}
.outcome-lbl.podium{color:#ffd700;}
.outcome-lbl.points{color:#00c48c;}
.outcome-lbl.nopoints{color:#e10600;}
.range-lbl{font-size:0.72rem;color:#666;margin-bottom:0.45rem;}
.bar-wrap{
    background:#1e1e28;
    border-radius:20px;
    height:0;
    padding-bottom:3%;   /* percentage height = zoom-proof */
    margin:0.45rem auto;
    width:80%;
    overflow:hidden;
    position:relative;
}
.bar-fill{
    position:absolute;
    top:0;left:0;bottom:0;
    border-radius:20px;
    background:#e10600;
    min-height:4px;      /* absolute floor so it never vanishes */
}
.conf-txt{font-size:0.78rem;color:#aaa;margin-top:0.35rem;}
.badge{font-family:'Bebas Neue',sans-serif;font-size:0.68rem;letter-spacing:1px;color:#e10600;margin-top:0.35rem;min-height:1em;}

/* verdict — base */
.verdict{
    border-radius:10px;
    padding:1.15rem 2rem 1.8rem;
    text-align:center;
    border-width:2px;
    border-style:solid;
    margin:2px 2px 6px 2px;  /* give all edges room so border never clips */
}
/* color variants */
.verdict.podium  { background:linear-gradient(135deg,#1a1400,#0a0a0f); border-color:#ffd700; }
.verdict.points  { background:linear-gradient(135deg,#001a10,#0a0a0f); border-color:#00c48c; }
.verdict.nopoints{ background:linear-gradient(135deg,#1a0000,#0a0a0f); border-color:#e10600; }
.verdict.tied    { background:linear-gradient(135deg,#0d0d1a,#0a0a0f); border-color:#888;    }

.verdict-title{font-family:'Bebas Neue',sans-serif;font-size:0.82rem;letter-spacing:3px;color:#888;}
.verdict-result{font-family:'Bebas Neue',sans-serif;font-size:2.1rem;letter-spacing:2px;margin:0.2rem 0;}
.verdict-result.podium  {color:#ffd700;}
.verdict-result.points  {color:#00c48c;}
.verdict-result.nopoints{color:#e10600;}
.verdict-result.tied    {color:#aaa;}
.verdict-sub{font-size:0.8rem;color:#aaa;letter-spacing:0.5px;}

.metrics-wrap{display:flex;gap:10px;}
.metric-card{flex:1;background:#111117;border:1px solid #222;border-radius:8px;padding:0.85rem;text-align:center;}
.metric-label{font-size:0.66rem;letter-spacing:1.5px;color:#888;text-transform:uppercase;}
.metric-value{font-family:'Bebas Neue',sans-serif;font-size:1.75rem;color:#e10600;}
.metric-sub{font-size:0.68rem;color:#555;margin-top:0.1rem;}

.info-strip{display:flex;gap:2rem;align-items:center;flex-wrap:wrap;
    background:#111117;border:1px solid #222;border-radius:8px;padding:0.75rem 1.2rem;}
.info-label{font-size:0.63rem;color:#888;letter-spacing:1.5px;text-transform:uppercase;}
.info-value{font-family:'Bebas Neue',sans-serif;font-size:1.25rem;color:#e8e8e8;}
.info-value.red{color:#e10600;}

.acc-table{background:#111117;border:1px solid #222;border-radius:8px;overflow:hidden;}
.acc-row{display:flex;justify-content:space-between;align-items:center;
    padding:0.5rem 1rem;border-bottom:1px solid #1e1e28;font-size:0.83rem;}
.acc-row:last-child{border-bottom:none;}
.acc-model{color:#ccc;}
.acc-vals{display:flex;gap:0.8rem;}
.acc-val{font-family:'Bebas Neue',sans-serif;font-size:0.95rem;color:#e10600;}
.acc-val.best{color:#ffd700;}
.acc-footer{padding:0.45rem 1rem;font-size:0.7rem;color:#888;letter-spacing:0.8px;border-top:1px solid #1e1e28;}

.ov-wrap{display:flex;gap:10px;}
.ov-card{flex:1;background:#111117;border-radius:8px;padding:1rem;text-align:center;}
.ov-card.podium{border:2px solid #ffd700;}
.ov-card.points{border:2px solid #00c48c;}
.ov-card.nopoints{border:2px solid #e10600;}
.ov-emoji{font-size:1.4rem;}
.ov-label{font-family:'Bebas Neue',sans-serif;font-size:1.05rem;letter-spacing:1px;margin:0.25rem 0;}
.ov-label.podium{color:#ffd700;}
.ov-label.points{color:#00c48c;}
.ov-label.nopoints{color:#e10600;}
.ov-count{font-family:'Bebas Neue',sans-serif;font-size:1.7rem;color:#e8e8e8;}
.ov-pct{font-size:0.72rem;color:#888;}

.section-label{font-family:'Bebas Neue',sans-serif;font-size:0.95rem;letter-spacing:2px;
    color:#e10600;border-bottom:1px solid #222;padding-bottom:0.3rem;margin-bottom:0.7rem;}

.placeholder{display:flex;flex-direction:column;align-items:center;justify-content:center;
    min-height:260px;background:#111117;border:1px solid #222;border-radius:8px;
    text-align:center;gap:0.5rem;padding:2rem;}
.ph-emoji{font-size:2.8rem;}
.ph-title{font-family:'Bebas Neue',sans-serif;font-size:1.9rem;letter-spacing:3px;color:#e10600;}
.ph-sub{color:#666;font-size:0.86rem;letter-spacing:0.8px;line-height:1.6;}
"""

def iframe(body_html, height):
    full = f"""<!DOCTYPE html>
<html><head><style>
{EMBED_CSS}
html,body{{background:transparent;overflow:hidden;}}
</style></head>
<body>{body_html}</body></html>"""
    components.html(full, height=height)

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return (
        pd.read_csv("results.csv"),
        pd.read_csv("races.csv"),
        pd.read_csv("drivers.csv"),
        pd.read_csv("constructors.csv"),
        pd.read_csv("qualifying.csv"),
        pd.read_csv("driver_standings.csv"),
    )

@st.cache_resource
def train_models():
    results, races, drivers, constructors, qualifying, standings = load_data()

    races_f = races[(races['year'] >= 2019) & (races['year'] <= 2024)][['raceId','year','round']]
    res_f   = results.merge(races_f, on='raceId', how='inner')

    data = res_f[['raceId','grid','driverId','constructorId','positionOrder','statusId']].copy()
    data = data[data['grid'] > 0].dropna()
    data = data.merge(races[['raceId','year','round']], on='raceId', how='left')
    data = data.sort_values(['year','round']).reset_index(drop=True)

    fb = data['positionOrder'].mean()
    data['driver_rolling_avg']      = data.groupby('driverId')['positionOrder'].transform(
        lambda x: x.shift(1).rolling(5,min_periods=1).mean()).fillna(fb)
    data['constructor_rolling_avg'] = data.groupby('constructorId')['positionOrder'].transform(
        lambda x: x.shift(1).rolling(5,min_periods=1).mean()).fillna(fb)
    data['driver_season_avg']       = data.groupby(['driverId','year'])['positionOrder'].transform(
        lambda x: x.shift(1).expanding().mean()).fillna(fb)
    hga = data[data['grid'] >= 15].groupby('driverId')['positionOrder'].mean()
    data['driver_highgrid_avg']     = data['driverId'].map(hga).fillna(fb)

    def best_qt(row):
        for c in ['q3','q2','q1']:
            if pd.notna(row[c]) and str(row[c]) != '\\N': return row[c]
        return np.nan

    def to_sec(t):
        try:
            m,s = str(t).split(':'); return int(m)*60+float(s)
        except: return np.nan

    qualifying['best_q_seconds'] = qualifying.apply(best_qt,axis=1).apply(to_sec)
    pole = qualifying.groupby('raceId')['best_q_seconds'].min().rename('pole_time')
    qj   = qualifying.join(pole, on='raceId')
    qj['q_gap_to_pole'] = qj['best_q_seconds'] - qj['pole_time']
    data = data.merge(qj[['raceId','driverId','q_gap_to_pole']], on=['raceId','driverId'], how='left')
    data['q_gap_to_pole'] = data['q_gap_to_pole'].fillna(data['q_gap_to_pole'].median())

    data = data.merge(
        standings[['raceId','driverId','points','position']].rename(
            columns={'points':'champ_points','position':'champ_position'}),
        on=['raceId','driverId'], how='left')
    data['champ_points']   = data['champ_points'].fillna(0)
    data['champ_position'] = data['champ_position'].fillna(20)
    data = data.dropna()

    def clf(row):
        if row['statusId'] > 13:         return 2
        elif row['positionOrder'] <= 3:  return 0
        elif row['positionOrder'] <= 10: return 1
        else:                            return 2

    y = data.apply(clf, axis=1)

    pre = ColumnTransformer(
        [('ohe', OneHotEncoder(handle_unknown='ignore'), ['driverId','constructorId'])],
        remainder='passthrough')
    X = data[['grid','driverId','constructorId','driver_rolling_avg','constructor_rolling_avg',
               'driver_highgrid_avg','driver_season_avg','q_gap_to_pole','champ_points','champ_position']]
    Xe = pre.fit_transform(X)
    Xtr,Xte,ytr,yte = train_test_split(Xe, y, test_size=0.2, random_state=42)

    lr  = LogisticRegression(max_iter=5000, class_weight='balanced'); lr.fit(Xtr,ytr)
    dt  = DecisionTreeClassifier(random_state=42, max_depth=8, min_samples_leaf=2, class_weight='balanced'); dt.fit(Xtr,ytr)
    rf  = RandomForestClassifier(n_estimators=200, max_depth=15, max_features='sqrt', random_state=42, class_weight='balanced'); rf.fit(Xtr,ytr)
    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42); xgb.fit(Xtr,ytr)

    mdls = [('Logistic Regression',lr),('Decision Tree',dt),('Random Forest',rf),('XGBoost',xgb)]
    evl  = pd.DataFrame([{
        'Model': n,
        'Accuracy': round(accuracy_score(yte, m.predict(Xte)), 3),
        'F1 Score': round(f1_score(yte, m.predict(Xte), average='weighted'), 3),
    } for n,m in mdls])

    return mdls, pre, data, hga, evl, drivers, constructors

# ── Train ─────────────────────────────────────────────────────────────────────
with st.spinner("🏁 Loading data & training models..."):
    models, preprocessor, data, high_grid_avg, eval_df, drivers_df, constructors_df = train_models()

driver_names      = dict(zip(drivers_df['driverId'], drivers_df['forename']+' '+drivers_df['surname']))
constructor_names = dict(zip(constructors_df['constructorId'], constructors_df['name']))

driver_options = {driver_names.get(d, f'Driver {d}'): d for d in sorted(data['driverId'].unique())}
cons_options   = {constructor_names.get(c, f'Cons {c}'): c for c in sorted(data['constructorId'].unique())}

OMAP  = {0:'PODIUM 🏆',     1:'POINTS ✅',    2:'NO POINTS / DNF ❌'}
OCLS  = {0:'podium',         1:'points',       2:'nopoints'}
RMAP  = {0:'P1 – P3',        1:'P4 – P10',     2:'P11+ / DNF'}

def get_feats(did, cid):
    dd   = data[data['driverId']==did].sort_values(['year','round'])
    cd   = data[data['constructorId']==cid].sort_values(['year','round'])
    mp   = data['positionOrder'].mean()
    r    = dd['positionOrder'].tail(5).mean()
    cr   = cd['positionOrder'].tail(5).mean()
    s    = dd['positionOrder'].mean()
    return (
        r  if not np.isnan(r)  else mp,
        cr if not np.isnan(cr) else mp,
        high_grid_avg.get(did, mp),
        s  if not np.isnan(s)  else mp,
        dd['q_gap_to_pole'].iloc[-1]  if len(dd)>0 else data['q_gap_to_pole'].median(),
        dd['champ_points'].iloc[-1]   if len(dd)>0 else 0,
        dd['champ_position'].iloc[-1] if len(dd)>0 else 20,
    )

# ── Layout ────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2], gap="large")

# ────────────────────────── LEFT ─────────────────────────────────────────────
with left_col:
    st.markdown('<p class="section-label">RACE INPUTS</p>', unsafe_allow_html=True)
    sel_drv  = st.selectbox("Driver",              options=list(driver_options.keys()))
    sel_cons = st.selectbox("Constructor / Team",  options=list(cons_options.keys()))
    grid_pos = st.slider("Grid Position", 1, 20, 10)
    st.markdown("<br>", unsafe_allow_html=True)
    predict  = st.button("🏁  PREDICT FINISH")

    # Accuracy table
    st.markdown('<p class="section-label" style="margin-top:1.2rem">MODEL ACCURACY</p>',
                unsafe_allow_html=True)
    best_idx  = eval_df['F1 Score'].idxmax()
    best_row  = eval_df.loc[best_idx]
    rows_html = ""
    for i, row in eval_df.iterrows():
        c = "best" if i == best_idx else ""
        rows_html += f"""<div class="acc-row">
            <span class="acc-model">{row['Model']}</span>
            <span class="acc-vals">
                <span class="acc-val {c}">Acc&nbsp;{row['Accuracy']}</span>
                <span class="acc-val {c}">F1&nbsp;{row['F1 Score']}</span>
            </span>
        </div>"""
    acc_body = f"""
    <div class="acc-table">
        {rows_html}
        <div class="acc-footer">
            ★ BEST: <span style="color:#ffd700">{best_row['Model']}</span>
            &nbsp;·&nbsp; F1 <span style="color:#ffd700">{best_row['F1 Score']}</span>
            &nbsp;·&nbsp; Acc <span style="color:#ffd700">{best_row['Accuracy']}</span>
        </div>
    </div>"""
    iframe(acc_body, 225)

# ────────────────────────── RIGHT ────────────────────────────────────────────
with right_col:
    if predict:
        did = driver_options[sel_drv]
        cid = cons_options[sel_cons]
        ra, cr, hg, sa, qg, cp, cpos = get_feats(did, cid)

        inp = pd.DataFrame([[grid_pos,did,cid,ra,cr,hg,sa,qg,cp,cpos]],
            columns=['grid','driverId','constructorId','driver_rolling_avg','constructor_rolling_avg',
                     'driver_highgrid_avg','driver_season_avg','q_gap_to_pole','champ_points','champ_position'])
        ie = preprocessor.transform(inp)

        rl = []
        for name, mdl in models:
            p = mdl.predict(ie)[0]
            pb = mdl.predict_proba(ie)[0][p]
            rl.append({'Model':name,'pred':p,'prob':pb})

        vc   = Counter(r['pred'] for r in rl)
        tv   = vc.most_common()
        mpred= tv[0][0]
        tied = len(tv)>1 and tv[0][1]==tv[1][1]
        best = max(rl, key=lambda r: r['prob'])

        # Info strip
        iframe(f"""<div class="info-strip">
            <div><div class="info-label">DRIVER</div>
                 <div class="info-value">{sel_drv}</div></div>
            <div><div class="info-label">CONSTRUCTOR</div>
                 <div class="info-value">{sel_cons}</div></div>
            <div><div class="info-label">GRID POSITION</div>
                 <div class="info-value red">P{grid_pos}</div></div>
        </div>""", 80)

        st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)

        # Verdict — color-coded by outcome
        if tied:
            vcls   = "tied"
            vlabel = "TIED — SEE MOST CONFIDENT"
            vrange = ""
        else:
            vcls   = OCLS[mpred]
            vlabel = OMAP[mpred]
            vrange = RMAP[mpred]

        iframe(f"""<div class="verdict {vcls}">
            <div class="verdict-title">MAJORITY VERDICT</div>
            <div class="verdict-result {vcls}">{vlabel}</div>
            <div class="verdict-sub">{vrange} &nbsp;·&nbsp;
                Most Confident: <strong>{best['Model']}</strong> at {best['prob']:.1%}
            </div>
        </div>""", 145)

        # Spacer + Model cards
        st.markdown("<div style='margin-top:1.6rem'></div>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">MODEL PREDICTIONS</p>', unsafe_allow_html=True)
        cards = ""
        for r in rl:
            oc   = OCLS[r['pred']]
            conf = int(r['prob']*100)
            glow = ' glow' if r['Model']==best['Model'] else ''
            badge= '★ MOST CONFIDENT' if r['Model']==best['Model'] else '&nbsp;'
            cards += f"""<div class="card {oc}{glow}">
                <div class="model-tag">{r['Model']}</div>
                <div class="outcome-lbl {oc}">{OMAP[r['pred']]}</div>
                <div class="range-lbl">{RMAP[r['pred']]}</div>
                <div class="bar-wrap"><div class="bar-fill" style="width:{conf}%"></div></div>
                <div class="conf-txt">{conf}% confidence</div>
                <div class="badge">{badge}</div>
            </div>"""
        iframe(f'<div class="cards-wrap">{cards}</div>', 205)

        # Quick stats
        st.markdown('<p class="section-label" style="margin-top:1.8rem">QUICK STATS</p>',
                    unsafe_allow_html=True)
        dd    = data[data['driverId']==did]
        davg  = round(dd['positionOrder'].mean(),1) if len(dd) else "—"
        ta    = eval_df.loc[eval_df['Accuracy'].idxmax()]
        tf    = eval_df.loc[eval_df['F1 Score'].idxmax()]
        stats = [("GRID POS",f"P{grid_pos}","Starting Position"),
                 ("DRIVER AVG",str(davg),"Avg Finish (2019–24)"),
                 ("BEST ACCURACY",str(ta['Accuracy']),ta['Model']),
                 ("BEST F1",str(tf['F1 Score']),tf['Model'])]
        mcards = "".join(f"""<div class="metric-card">
            <div class="metric-label">{l}</div>
            <div class="metric-value">{v}</div>
            <div class="metric-sub">{s}</div>
        </div>""" for l,v,s in stats)
        iframe(f'<div class="metrics-wrap">{mcards}</div>', 105)

    else:
        # Placeholder
        iframe("""<div class="placeholder">
            <div class="ph-emoji">🏎️</div>
            <div class="ph-title">READY TO RACE</div>
            <div class="ph-sub">Select a driver, constructor &amp; grid position,<br>
                then press <strong style="color:#e10600">PREDICT FINISH</strong></div>
        </div>""", 290)

        # Dataset overview
        st.markdown('<p class="section-label" style="margin-top:1rem">DATASET OVERVIEW (2019–2024)</p>',
                    unsafe_allow_html=True)

        def clf_dist(row):
            if row['statusId']>13:         return 'No Points / DNF'
            elif row['positionOrder']<=3:  return 'Podium'
            elif row['positionOrder']<=10: return 'Points'
            else:                          return 'No Points / DNF'

        dist  = data.apply(clf_dist, axis=1).value_counts()
        total = dist.sum()
        ov = ""
        for label, cls, emoji in [("Podium","podium","🏆"),("Points","points","✅"),("No Points / DNF","nopoints","❌")]:
            cnt = dist.get(label, 0)
            pct = round(cnt/total*100,1)
            ov += f"""<div class="ov-card {cls}">
                <div class="ov-emoji">{emoji}</div>
                <div class="ov-label {cls}">{label}</div>
                <div class="ov-count">{cnt}</div>
                <div class="ov-pct">{pct}% of entries</div>
            </div>"""
        iframe(f'<div class="ov-wrap">{ov}</div>', 155)
# 
# ───────────────────────────────────────────────────────────────────────────────
# sidebar disclaimer
with st.sidebar:
    st.markdown('---')
    st.caption('🚀 **Disclaimer**')
    st.caption(
        'This is an unofficial fan project created for educational and portfolio purposes. '
        'All F1 logos, names, and trademarks are the property of their respective owners (Formula One Management/Liberty Media). '
        'Data is sourced from the Kaggle datasets.'
    )
