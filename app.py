import streamlit as st
import numpy as np
import pandas as pd
import random
from deap import base, creator, tools, algorithms
import plotly.express as px

# =========================
# 0. AYARLAR VE KÜTÜPHANE
# =========================
RNG = np.random.default_rng(42)
TOP_K = 10
LIB_SIZE = 3000

st.set_page_config(page_title="Civil-AI Lab v3.3", layout="wide")
st.title("🏗️ Civil-AI: Profesyonel Malzeme Reçete Sentezleyici")

@st.cache_data
def build_library(n=LIB_SIZE):
    rows = []
    categories = {
        "Bağlayıcı (Çimento vb.)": {"min": 0.15, "max": 0.40, "density": 3150},
        "Agrega (Kum/Çakıl)": {"min": 0.55, "max": 0.80, "density": 2700},
        "Polimer Katkı": {"min": 0.0, "max": 0.05, "density": 1100},
        "Nano Malzeme": {"min": 0.0, "max": 0.03, "density": 2200},
        "Su": {"min": 0.08, "max": 0.20, "density": 1000},
    }
    for i in range(n):
        cat = random.choice(list(categories.keys()))
        cfg = categories[cat]
        s_val = (40 + 100 * RNG.uniform(0.2, 0.8) - 50 * RNG.uniform(0.1, 0.4)) / 10
        f_val = (5 + 50 * RNG.uniform(0.1, 0.6)) / 10
        cost = max(0.01, 0.02 + 0.4 * RNG.uniform(0, 0.3))
        rows.append([f"{cat}_{i}", cat, s_val, f_val, cost, cfg["density"], cfg["min"], cfg["max"]])
    return pd.DataFrame(rows, columns=["name", "category", "strength", "flex", "cost_kg", "density", "min_lim", "max_lim"])

DB = build_library()

# =========================
# 1. GENETİK ALGORİTMA MANTIĞI
# =========================
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
def create_ind():
    idx = random.sample(range(len(DB)), TOP_K)
    ratios = [random.random() for _ in range(TOP_K)]
    return creator.Individual(idx + ratios)

toolbox.register("individual", create_ind)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(ind):
    idx = np.clip(np.array(ind[:TOP_K], dtype=int), 0, LIB_SIZE - 1).tolist()
    ratios = np.array(ind[TOP_K:], dtype=float)
    ratios /= np.sum(ratios)
    sel = DB.iloc[idx]
    
    # Mühendislik Kısıtları (Ceza Puanları)
    penalty = 0
    for cat in DB['category'].unique():
        rsum = np.sum(ratios[sel['category'] == cat])
        mn, mx = DB[DB['category'] == cat]['min_lim'].iloc[0], DB[DB['category'] == cat]['max_lim'].iloc[0]
        if rsum < mn: penalty += (mn - rsum) * 5000
        if rsum > mx: penalty += (rsum - mx) * 5000

    s_eff = np.sum(ratios * sel['strength']) * 10
    f_eff = np.sum(ratios * sel['flex']) * 10
    cost = np.sum(ratios * sel['density'] * sel['cost_kg'])
    
    score = (s_eff * 2 + f_eff * 1.5) - (cost / 10) - penalty
    return (max(1.0, float(score)),)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=25, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# =========================
# 2. ARAYÜZ VE REÇETE HESABI
# =========================
col_set, col_res = st.columns([1, 2])

with col_set:
    st.subheader("⚙️ Optimizasyon Ayarları")
    pop = st.slider("Popülasyon", 200, 1000, 400)
    gen = st.slider("Nesil", 50, 500, 150)
    btn = st.button("🚀 Reçeteyi Oluştur")

if btn:
    with st.spinner("En dayanıklı ve ekonomik karışım hesaplanıyor..."):
        pop_list = toolbox.population(n=pop)
        hof = tools.HallOfFame(1)
        algorithms.eaSimple(pop_list, toolbox, 0.7, 0.3, gen, halloffame=hof, verbose=False)

    best = hof[0]
    idx = np.clip(np.array(best[:TOP_K], dtype=int), 0, LIB_SIZE - 1).tolist()
    ratios = np.array(best[TOP_K:], dtype=float)
    ratios /= np.sum(ratios)
    
    res = DB.iloc[idx].copy()
    res['Hacim Oranı (%)'] = ratios * 100
    
    # --- KRİTİK HESAPLAMA: kg/m³ ---
    # Formül: Hacim Oranı * Malzeme Yoğunluğu
    res['Miktar (kg/m³)'] = (ratios * res['density']).astype(int)
    
    st.success("✅ Karışım Sentezi Tamamlandı!")
    
    m1, m2, m3 = st.columns(3)
    s_tot = np.sum(ratios * res['strength']) * 10
    m1.metric("Tahmini Dayanım", f"{s_tot:.1f} MPa")
    m2.metric("Toplam Ağırlık", f"{int(res['Miktar (kg/m³)'].sum())} kg/m³")
    m3.metric("Maliyet", f"${int(np.sum(ratios * res['density'] * res['cost_kg']))}/m³")

    st.subheader("📋 Uygulama Reçetesi (1 m³ İçin)")
    # Kullanıcıya ne kullanacağını net şekilde gösteren tablo
    st.table(res[['category', 'name', 'Miktar (kg/m³)', 'Hacim Oranı (%)']])
    
    fig = px.pie(res, values='Miktar (kg/m³)', names='category', title="Kütlesel Dağılım Grafiği")
    st.plotly_chart(fig)
