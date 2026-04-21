# ============================================================
# ML Spark Challenge — Streamlit Dashboard
# Run: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle, os, warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                              f1_score, precision_score, recall_score)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="ML Spark — Smart Procurement",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border: 1px solid #2e3452;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 6px 0;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #4fc3f7; }
    .metric-label { font-size: 0.85rem; color: #9aa0b4; margin-top: 4px; }
    .section-header {
        font-size: 1.3rem; font-weight: 700;
        color: #e0e6f0; margin: 18px 0 10px 0;
        border-left: 4px solid #4fc3f7; padding-left: 12px;
    }
    .insight-box {
        background: #1a1f2e; border: 1px solid #2e3452;
        border-radius: 10px; padding: 14px 18px; margin: 8px 0;
        color: #ccd3e0; font-size: 0.92rem;
    }
    .badge-high   { background:#e74c3c22; color:#e74c3c; border:1px solid #e74c3c55;
                    padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
    .badge-medium { background:#f39c1222; color:#f39c12; border:1px solid #f39c1255;
                    padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
    .badge-low    { background:#2ecc7122; color:#2ecc71; border:1px solid #2ecc7155;
                    padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
    .stSelectbox label, .stSlider label, .stNumberInput label { color: #9aa0b4 !important; }
    div[data-testid="stSidebar"] { background: #13161f; }
</style>
""", unsafe_allow_html=True)


# ── Data & model loading (cached) ──────────────────────────
@st.cache_data
def load_data():
    df   = pd.read_csv('Deliveries.csv')
    proj = pd.read_csv('Projects.csv')
    fact = pd.read_csv('Factories.csv')
    ext  = pd.read_csv('External_Factors.csv')
    return df, proj, fact, ext

@st.cache_data
def engineer_features(df, proj, fact, ext):
    d = df.copy()
    d = d.merge(fact.add_prefix('fac_').rename(columns={'fac_factory_id':'factory_id'}), on='factory_id')
    d = d.merge(proj.add_prefix('prj_').rename(columns={'prj_project_id':'project_id'}), on='project_id')
    d = d.merge(ext, on='date')
    d['priority_encoded']    = d['prj_priority_level'].map({'Low':0,'Medium':1,'High':2})
    d['expected_speed_kmh']  = d['distance_km'] / d['expected_time_hours']
    d['weather_x_traffic']   = d['weather_index'] * d['traffic_index']
    d['dist_x_traffic']      = d['distance_km'] * d['traffic_index']
    d['dist_x_weather']      = d['distance_km'] * d['weather_index']
    d['prod_variability_risk'] = d['fac_production_variability'] * d['distance_km']
    d['date'] = pd.to_datetime(d['date'])
    d['day_of_week']   = d['date'].dt.dayofweek
    d['week_of_month'] = d['date'].dt.day // 7
    d['delay_ratio']   = d['actual_time_hours'] / d['expected_time_hours']
    FEATURES = [
        'distance_km','expected_time_hours',
        'fac_base_production_per_week','fac_production_variability','fac_max_storage',
        'fac_latitude','fac_longitude',
        'prj_demand','priority_encoded','prj_latitude','prj_longitude',
        'weather_index','traffic_index','expected_speed_kmh','weather_x_traffic',
        'dist_x_traffic','dist_x_weather','prod_variability_risk','day_of_week','week_of_month'
    ]
    return d, FEATURES

@st.cache_resource
def train_models(df_eng, FEATURES):
    X = df_eng[FEATURES]
    y = df_eng['delay_flag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    cw = dict(zip([0,1], compute_class_weight('balanced', classes=np.array([0,1]), y=y_train)))
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
        'XGBoost':             xgb.XGBClassifier(scale_pos_weight=cw[0]/cw[1], n_estimators=200,
                                                   learning_rate=0.05, max_depth=5, random_state=42,
                                                   eval_metric='logloss', verbosity=0),
    }
    results = {}
    for name, model in models.items():
        Xtr = X_train_s if 'Logistic' in name else X_train
        Xte = X_test_s  if 'Logistic' in name else X_test
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:,1]
        results[name] = {
            'model': model, 'y_pred': y_pred,
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_prob),
        }
    importances = pd.Series(
        results['XGBoost']['model'].feature_importances_, index=FEATURES
    ).sort_values(ascending=False)
    return results, scaler, X_test, y_test, importances


# ── Load everything ────────────────────────────────────────
df, proj, fact, ext = load_data()
df_eng, FEATURES = engineer_features(df, proj, fact, ext)

with st.spinner("Training models..."):
    results, scaler, X_test, y_test, importances = train_models(df_eng, FEATURES)

xgb_model = results['XGBoost']['model']

# ── Sidebar navigation ─────────────────────────────────────
with st.sidebar:
    st.markdown("## 📦 ML Spark")
    st.markdown("**Smart Procurement Dashboard**")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠  Overview",
        "📊  EDA & Insights",
        "🤖  Model Performance",
        "🔮  Predict Delay",
        "🏆  Reward Optimizer",
    ])
    st.markdown("---")
    st.caption("Dataset: 1,200 deliveries | 5 factories | 200 projects")

page_key = page.split("  ")[1]


# ════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ════════════════════════════════════════════════════════════
if page_key == "Overview":
    st.markdown("# 📦 Smart Procurement Dashboard")
    st.markdown("*Predicting Delivery Delays & Reward-Based Planning Optimization*")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Total Deliveries</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{df['delay_flag'].mean()*100:.1f}%</div>
            <div class="metric-label">Overall Delay Rate</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{results['XGBoost']['roc_auc']:.3f}</div>
            <div class="metric-label">XGBoost ROC-AUC</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        avg_delay = df_eng['delay_ratio'].mean()
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{avg_delay:.2f}×</div>
            <div class="metric-label">Avg Delay Ratio</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Key Findings</div>', unsafe_allow_html=True)
        findings = [
            ("99.7%", "of all deliveries are delayed — systemic, not factory-specific"),
            ("Distance × Traffic", "is the #1 predictor of delay (15.7% feature importance)"),
            ("1.65×", "average actual/expected time — deliveries take 65% longer than planned"),
            ("High-priority projects", "face the same delay rate as Low — urgency is ignored in dispatch"),
        ]
        for val, desc in findings:
            st.markdown(f"""<div class="insight-box">
                <strong style="color:#4fc3f7">{val}</strong> — {desc}
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
        overview = pd.DataFrame({
            'File': ['Deliveries.csv', 'Projects.csv', 'Factories.csv', 'External_Factors.csv'],
            'Records': [1200, 200, 5, 30],
            'Key Column': ['delay_flag (target)', 'priority_level', 'production_variability', 'weather/traffic index'],
        })
        st.dataframe(overview, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header">Reward Framework</div>', unsafe_allow_html=True)
        reward_df = pd.DataFrame({
            'Event': ['On-time delivery', 'Delayed delivery', 'High-priority served', 'Each excess hour'],
            'Score': ['+10', '−15', '+5', '−2'],
        })
        st.dataframe(reward_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════
# PAGE 2: EDA
# ════════════════════════════════════════════════════════════
elif page_key == "EDA & Insights":
    st.markdown("# 📊 Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Delay Patterns", "External Factors", "Distributions"])

    with tab1:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Delay Rate by Factory**")
            fact_delay = df_eng.groupby('factory_id')['delay_flag'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#1a1f2e')
            ax.set_facecolor('#1a1f2e')
            bars = ax.bar(fact_delay['factory_id'], fact_delay['delay_flag']*100,
                          color='#e74c3c', edgecolor='#2e3452', linewidth=0.8, width=0.55)
            ax.set_ylim(90, 102)
            ax.set_ylabel('Delay Rate (%)', color='#9aa0b4')
            ax.tick_params(colors='#9aa0b4')
            for spine in ax.spines.values(): spine.set_edgecolor('#2e3452')
            for bar, val in zip(bars, fact_delay['delay_flag']*100):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8, color='white')
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with c2:
            st.markdown("**Delay Rate by Priority**")
            prio_delay = df_eng.groupby('prj_priority_level')['delay_flag'].mean()
            fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#1a1f2e')
            ax.set_facecolor('#1a1f2e')
            for p, c in zip(['Low','Medium','High'], ['#3498db','#f39c12','#e74c3c']):
                val = prio_delay.get(p, 0) * 100
                bar = ax.bar(p, val, color=c, edgecolor='#2e3452', width=0.5)
                ax.text(p, val+0.1, f'{val:.1f}%', ha='center', va='bottom', fontsize=9, color='white')
            ax.set_ylim(90, 102)
            ax.set_ylabel('Delay Rate (%)', color='#9aa0b4')
            ax.tick_params(colors='#9aa0b4')
            for spine in ax.spines.values(): spine.set_edgecolor('#2e3452')
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.markdown("**Delay Rate by Day of Week**")
        days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        dow = df_eng.groupby('day_of_week')['delay_flag'].mean().reindex(range(7)).fillna(0)
        fig, ax = plt.subplots(figsize=(10, 3), facecolor='#1a1f2e')
        ax.set_facecolor('#1a1f2e')
        ax.bar(days, dow*100, color='#4fc3f7', edgecolor='#2e3452', linewidth=0.8, width=0.6)
        ax.set_ylim(85, 105)
        ax.set_ylabel('Delay Rate (%)', color='#9aa0b4')
        ax.tick_params(colors='#9aa0b4')
        for spine in ax.spines.values(): spine.set_edgecolor('#2e3452')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab2:
        ext_s = ext.copy()
        ext_s['date'] = pd.to_datetime(ext_s['date'])

        st.markdown("**Daily Weather & Traffic Index (April 2026)**")
        fig, ax = plt.subplots(figsize=(12, 4), facecolor='#1a1f2e')
        ax.set_facecolor('#1a1f2e')
        ax.plot(ext_s['date'], ext_s['weather_index'], color='#3498db', label='Weather Index', linewidth=2)
        ax.plot(ext_s['date'], ext_s['traffic_index'], color='#e67e22', label='Traffic Index', linewidth=2)
        ax.set_ylabel('Index (0=mild, 1=severe)', color='#9aa0b4')
        ax.tick_params(colors='#9aa0b4', axis='x', rotation=30)
        ax.tick_params(colors='#9aa0b4', axis='y')
        for spine in ax.spines.values(): spine.set_edgecolor('#2e3452')
        ax.legend(facecolor='#1a1f2e', edgecolor='#2e3452', labelcolor='white')
        st.pyplot(fig, use_container_width=True)
        plt.close()

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Avg Weather Index", f"{ext['weather_index'].mean():.2f}")
            st.metric("Max Weather Index", f"{ext['weather_index'].max():.2f}")
        with c2:
            st.metric("Avg Traffic Index", f"{ext['traffic_index'].mean():.2f}")
            st.metric("Max Traffic Index", f"{ext['traffic_index'].max():.2f}")

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Delay Ratio Distribution**")
            fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#1a1f2e')
            ax.set_facecolor('#1a1f2e')
            ax.hist(df_eng['delay_ratio'], bins=40, color='#4fc3f7', edgecolor='#1a1f2e', alpha=0.85)
            ax.axvline(1.2, color='#e74c3c', linestyle='--', linewidth=2, label='Threshold (1.2×)')
            ax.axvline(df_eng['delay_ratio'].mean(), color='#f39c12', linestyle=':', linewidth=1.5,
                       label=f'Mean ({df_eng["delay_ratio"].mean():.2f}×)')
            ax.set_xlabel('Actual / Expected Time', color='#9aa0b4')
            ax.set_ylabel('Count', color='#9aa0b4')
            ax.tick_params(colors='#9aa0b4')
            for spine in ax.spines.values(): spine.set_edgecolor('#2e3452')
            ax.legend(facecolor='#1a1f2e', edgecolor='#2e3452', labelcolor='white', fontsize=8)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with c2:
            st.markdown("**Distance Distribution by Outcome**")
            fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#1a1f2e')
            ax.set_facecolor('#1a1f2e')
            ax.hist(df_eng[df_eng['delay_flag']==1]['distance_km'], bins=30, alpha=0.7,
                    color='#e74c3c', label='Delayed', density=True)
            ax.hist(df_eng[df_eng['delay_flag']==0]['distance_km'], bins=5, alpha=0.9,
                    color='#2ecc71', label='On-time', density=True)
            ax.set_xlabel('Distance (km)', color='#9aa0b4')
            ax.set_ylabel('Density', color='#9aa0b4')
            ax.tick_params(colors='#9aa0b4')
            for spine in ax.spines.values(): spine.set_edgecolor('#2e3452')
            ax.legend(facecolor='#1a1f2e', edgecolor='#2e3452', labelcolor='white', fontsize=8)
            st.pyplot(fig, use_container_width=True)
            plt.close()


# ════════════════════════════════════════════════════════════
# PAGE 3: MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
elif page_key == "Model Performance":
    st.markdown("# 🤖 Model Training & Evaluation")
    st.markdown("---")

    # Metric cards
    cols = st.columns(3)
    model_colors = {'Logistic Regression': '#3498db', 'Random Forest': '#2ecc71', 'XGBoost': '#e74c3c'}
    for col, (name, res) in zip(cols, results.items()):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div style="font-size:1rem;font-weight:700;color:{model_colors[name]}">{name}</div>
                <div class="metric-value">{res['roc_auc']:.4f}</div>
                <div class="metric-label">ROC-AUC</div>
                <div style="margin-top:8px;font-size:0.85rem;color:#9aa0b4">
                    F1: {res['f1']:.4f} &nbsp;|&nbsp; Precision: {res['precision']:.4f}<br>Recall: {res['recall']:.4f}
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
        metrics = ['f1', 'roc_auc', 'precision', 'recall']
        metric_labels = ['F1', 'ROC-AUC', 'Precision', 'Recall']
        x = np.arange(len(results))
        width = 0.2
        fig, ax = plt.subplots(figsize=(7, 4), facecolor='#1a1f2e')
        ax.set_facecolor('#1a1f2e')
        colors = ['#3498db','#e74c3c','#2ecc71','#f39c12']
        for i, (m, label, c) in enumerate(zip(metrics, metric_labels, colors)):
            vals = [results[n][m] for n in results]
            ax.bar(x + i*width - 1.5*width, vals, width, label=label, color=c, alpha=0.85, edgecolor='#1a1f2e')
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(' ','\n') for n in results], color='#9aa0b4', fontsize=9)
        ax.set_ylim(0.85, 1.02)
        ax.set_ylabel('Score', color='#9aa0b4')
        ax.tick_params(colors='#9aa0b4')
        for spine in ax.spines.values(): spine.set_edgecolor('#2e3452')
        ax.legend(facecolor='#1a1f2e', edgecolor='#2e3452', labelcolor='white', fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c2:
        st.markdown('<div class="section-header">Feature Importance (XGBoost)</div>', unsafe_allow_html=True)
        top12 = importances.head(12)
        fig, ax = plt.subplots(figsize=(7, 4), facecolor='#1a1f2e')
        ax.set_facecolor('#1a1f2e')
        bar_colors = ['#e74c3c' if v>0.10 else '#4fc3f7' if v>0.05 else '#7f8c9a' for v in top12.values]
        ax.barh(range(len(top12)), top12.values, color=bar_colors, edgecolor='#1a1f2e')
        ax.set_yticks(range(len(top12)))
        ax.set_yticklabels([n.replace('_',' ').title() for n in top12.index], fontsize=8, color='#9aa0b4')
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score', color='#9aa0b4')
        ax.tick_params(colors='#9aa0b4')
        for spine in ax.spines.values(): spine.set_edgecolor('#2e3452')
        high_p = mpatches.Patch(color='#e74c3c', label='>10%')
        mid_p  = mpatches.Patch(color='#4fc3f7', label='5–10%')
        low_p  = mpatches.Patch(color='#7f8c9a', label='<5%')
        ax.legend(handles=[high_p,mid_p,low_p], facecolor='#1a1f2e', edgecolor='#2e3452',
                  labelcolor='white', fontsize=8, loc='lower right')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.markdown('<div class="section-header">Classification Report — XGBoost</div>', unsafe_allow_html=True)
    y_pred = results['XGBoost']['y_pred']
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).T.round(4)
    st.dataframe(report_df, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 4: PREDICT DELAY
# ════════════════════════════════════════════════════════════
elif page_key == "Predict Delay":
    st.markdown("# 🔮 Predict Delivery Delay")
    st.markdown("Fill in the delivery details below to get an instant prediction.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Delivery Details**")
        distance_km = st.slider("Distance (km)", 50, 1000, 400, step=10)
        expected_time = st.number_input("Expected Time (hours)", min_value=1.0, max_value=50.0, value=10.0, step=0.5)
        day_of_week = st.selectbox("Day of Week", ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
        week_of_month = st.selectbox("Week of Month", [0, 1, 2, 3, 4])

    with col2:
        st.markdown("**Factory Details**")
        factory_id = st.selectbox("Select Factory", fact['factory_id'].tolist())
        frow = fact[fact['factory_id']==factory_id].iloc[0]
        st.info(f"📍 Lat: {frow['latitude']:.4f}, Lon: {frow['longitude']:.4f}  \n"
                f"⚙️ Production/week: {frow['base_production_per_week']}  \n"
                f"📉 Variability: {frow['production_variability']}  \n"
                f"🏗️ Max Storage: {frow['max_storage']}")

    with col3:
        st.markdown("**Project & External Conditions**")
        priority = st.selectbox("Project Priority", ['Low', 'Medium', 'High'])
        demand = st.number_input("Project Demand (units)", min_value=1, max_value=20, value=3)
        weather_index = st.slider("Weather Index (0=clear → 1=severe)", 0.0, 1.0, 0.3, step=0.01)
        traffic_index = st.slider("Traffic Index (0=free → 1=gridlock)", 0.0, 1.0, 0.4, step=0.01)
        prj_lat = st.number_input("Project Latitude", value=14.5, format="%.4f")
        prj_lon = st.number_input("Project Longitude", value=77.5, format="%.4f")

    st.markdown("---")
    if st.button("🔮 Predict", use_container_width=True):
        dow_map = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}
        priority_map = {'Low':0,'Medium':1,'High':2}

        expected_speed = distance_km / expected_time
        weather_x_traffic = weather_index * traffic_index
        dist_x_traffic    = distance_km * traffic_index
        dist_x_weather    = distance_km * weather_index
        prod_var_risk     = frow['production_variability'] * distance_km

        input_data = pd.DataFrame([{
            'distance_km':                distance_km,
            'expected_time_hours':        expected_time,
            'fac_base_production_per_week': frow['base_production_per_week'],
            'fac_production_variability': frow['production_variability'],
            'fac_max_storage':            frow['max_storage'],
            'fac_latitude':               frow['latitude'],
            'fac_longitude':              frow['longitude'],
            'prj_demand':                 demand,
            'priority_encoded':           priority_map[priority],
            'prj_latitude':               prj_lat,
            'prj_longitude':              prj_lon,
            'weather_index':              weather_index,
            'traffic_index':              traffic_index,
            'expected_speed_kmh':         expected_speed,
            'weather_x_traffic':          weather_x_traffic,
            'dist_x_traffic':             dist_x_traffic,
            'dist_x_weather':             dist_x_weather,
            'prod_variability_risk':      prod_var_risk,
            'day_of_week':                dow_map[day_of_week],
            'week_of_month':              week_of_month,
        }])

        delay_prob = xgb_model.predict_proba(input_data)[0][1]
        prediction = xgb_model.predict(input_data)[0]

        r1, r2, r3 = st.columns(3)
        with r1:
            color = "#e74c3c" if prediction == 1 else "#2ecc71"
            label = "DELAYED" if prediction == 1 else "ON TIME"
            st.markdown(f"""<div class="metric-card" style="border-color:{color}55">
                <div class="metric-value" style="color:{color}">{label}</div>
                <div class="metric-label">Prediction</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:#f39c12">{delay_prob*100:.1f}%</div>
                <div class="metric-label">Delay Probability</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            risk = "High Risk" if delay_prob > 0.8 else "Medium Risk" if delay_prob > 0.5 else "Low Risk"
            risk_color = "#e74c3c" if delay_prob > 0.8 else "#f39c12" if delay_prob > 0.5 else "#2ecc71"
            st.markdown(f"""<div class="metric-card" style="border-color:{risk_color}55">
                <div class="metric-value" style="color:{risk_color}">{risk}</div>
                <div class="metric-label">Risk Level</div>
            </div>""", unsafe_allow_html=True)

        # Gauge chart
        fig, ax = plt.subplots(figsize=(5, 2.5), facecolor='#1a1f2e')
        ax.set_facecolor('#1a1f2e')
        ax.barh(0, 1, color='#2e3452', height=0.4)
        ax.barh(0, delay_prob, color=('#e74c3c' if delay_prob>0.7 else '#f39c12' if delay_prob>0.4 else '#2ecc71'), height=0.4)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel('Delay Probability', color='#9aa0b4')
        ax.tick_params(colors='#9aa0b4')
        for spine in ax.spines.values(): spine.set_edgecolor('#2e3452')
        ax.text(delay_prob, 0.25, f'{delay_prob*100:.1f}%', color='white', fontsize=12, fontweight='bold', va='bottom')
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Recommendation
        st.markdown("---")
        st.markdown("**Recommendation**")
        if delay_prob > 0.8 and priority == 'High':
            st.error("🔴 High-priority delivery with very high delay risk — pre-position from a closer factory or reschedule to a low-traffic day.")
        elif delay_prob > 0.8:
            st.warning("🟡 High delay probability — consider rerouting or scheduling on a lower-traffic day.")
        elif priority == 'High':
            st.success("🟢 High-priority delivery with manageable delay risk — dispatch as planned.")
        else:
            st.success("🟢 Low risk — safe to dispatch.")


# ════════════════════════════════════════════════════════════
# PAGE 5: REWARD OPTIMIZER
# ════════════════════════════════════════════════════════════
elif page_key == "Reward Optimizer":
    st.markdown("# 🏆 Reward-Based Dispatch Optimizer")
    st.markdown("---")

    df_test = df_eng.iloc[X_test.index].copy()
    df_test['delay_prob']      = xgb_model.predict_proba(X_test)[:,1]
    df_test['predicted_delay'] = xgb_model.predict(X_test)

    def compute_reward(row):
        reward = 10 if row['delay_flag'] == 0 else -15
        if row['prj_priority_level'] == 'High': reward += 5
        excess = max(0, row['actual_time_hours'] - row['expected_time_hours'])
        reward -= 2 * int(excess)
        return reward

    df_test['reward'] = df_test.apply(compute_reward, axis=1)
    df_test['priority_score'] = (
        df_test['priority_encoded'] * 3 +
        (1 - df_test['delay_prob']) * 5 +
        (1 / (df_test['distance_km'] / df_test['distance_km'].max() + 0.01)) * 2
    )

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{df_test['reward'].mean():.1f}</div>
            <div class="metric-label">Avg Reward Score</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#e74c3c">{df_test['reward'].min():.0f}</div>
            <div class="metric-label">Worst Delivery</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#2ecc71">{df_test['reward'].max():.0f}</div>
            <div class="metric-label">Best Delivery</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        high_prio_count = (df_test['prj_priority_level']=='High').sum()
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{high_prio_count}</div>
            <div class="metric-label">High-Priority Deliveries</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Reward Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='#1a1f2e')
        ax.set_facecolor('#1a1f2e')
        ax.hist(df_test['reward'], bins=25, color='#4fc3f7', edgecolor='#1a1f2e', alpha=0.85)
        ax.axvline(df_test['reward'].mean(), color='#e74c3c', linestyle='--', linewidth=2,
                   label=f"Mean: {df_test['reward'].mean():.1f}")
        ax.set_xlabel('Reward Score', color='#9aa0b4')
        ax.set_ylabel('Count', color='#9aa0b4')
        ax.tick_params(colors='#9aa0b4')
        for spine in ax.spines.values(): spine.set_edgecolor('#2e3452')
        ax.legend(facecolor='#1a1f2e', edgecolor='#2e3452', labelcolor='white', fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown('<div class="section-header">Avg Reward by Priority</div>', unsafe_allow_html=True)
        reward_prio = df_test.groupby('prj_priority_level')['reward'].mean()
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='#1a1f2e')
        ax.set_facecolor('#1a1f2e')
        for p, c in zip(['Low','Medium','High'], ['#3498db','#f39c12','#e74c3c']):
            if p in reward_prio.index:
                ax.bar(p, reward_prio[p], color=c, edgecolor='#1a1f2e', width=0.5)
                ax.text(p, reward_prio[p]+0.3, f'{reward_prio[p]:.1f}', ha='center',
                        fontweight='bold', color='white')
        ax.set_ylabel('Avg Reward', color='#9aa0b4')
        ax.axhline(0, color='#2e3452', linewidth=1)
        ax.tick_params(colors='#9aa0b4')
        for spine in ax.spines.values(): spine.set_edgecolor('#2e3452')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.markdown('<div class="section-header">Top Deliveries to Dispatch First</div>', unsafe_allow_html=True)

    n_top = st.slider("Show top N deliveries", 5, 30, 10)
    top_del = df_test.nlargest(n_top, 'priority_score')[[
        'delivery_id','factory_id','project_id','distance_km',
        'prj_priority_level','delay_prob','priority_score','reward'
    ]].reset_index(drop=True)
    top_del.columns = ['Delivery','Factory','Project','Distance (km)','Priority','Delay Prob','Priority Score','Reward']
    top_del['Delay Prob'] = top_del['Delay Prob'].map(lambda x: f"{x*100:.1f}%")
    top_del['Priority Score'] = top_del['Priority Score'].map(lambda x: f"{x:.2f}")
    top_del['Distance (km)'] = top_del['Distance (km)'].map(lambda x: f"{x:.1f}")
    st.dataframe(top_del, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Dispatch Strategy Guide</div>', unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown("""<div class="insight-box">
            <span class="badge-high">HIGH PRIORITY</span><br><br>
            Low delay probability → <strong>Dispatch immediately</strong><br>
            High delay probability → Pre-position from nearest factory
        </div>""", unsafe_allow_html=True)
    with sc2:
        st.markdown("""<div class="insight-box">
            <span class="badge-medium">MEDIUM PRIORITY</span><br><br>
            Low delay probability → Dispatch in next batch<br>
            High delay probability → Reschedule to low-traffic day
        </div>""", unsafe_allow_html=True)
    with sc3:
        st.markdown("""<div class="insight-box">
            <span class="badge-low">LOW PRIORITY</span><br><br>
            High delay probability → <strong>Defer or reroute</strong><br>
            Low delay probability → Queue for next cycle
        </div>""", unsafe_allow_html=True)
