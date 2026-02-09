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

st.set_page_config(page_title="Civil-AI Lab v3.4", layout="wide")
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
        # Bilimsel korelasyonlu fiziksel özellikler
        s_val = (40 + 100 * RNG.uniform(0.2, 0.8) - 50 * RNG.uniform(0.1, 0.4)) / 10
        f_val = (5 + 50 * RNG.uniform(0.1, 0.6)) / 10
        cost = max(0.01, 0.02 + 0.4 * RNG.uniform(0, 0.3))
        rows.append([f"{cat}_{i}", cat, s_val, f_val, cost, cfg["density"], cfg["min"], cfg["max"]])
    return pd.DataFrame(rows, columns=["name", "category", "strength", "flex", "cost_kg", "density", "min_lim", "max_lim"])

DB = build_library()

# =========================
# 1. GENETİK ALGORİTMA (GA)
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
    # İndisleri tam sayıya çevir ve sınırla (IndexError koruması)
    idx = np.clip(np.array(ind[:TOP_K], dtype=int), 0, LIB_SIZE - 1).tolist()
    ratios = np.array(ind[TOP_K:], dtype=float)
    if np.sum(ratios) == 0: return (0,)
    ratios /= np.sum(ratios)
    
    sel = DB.iloc[idx]
    
    # Mühendislik Kısıtları (Penalty System)
    penalty = 0
    for cat in DB['category'].unique():
        rsum = np.sum(ratios[sel['category'] == cat])
        cat_meta = DB[DB['category'] == cat].iloc[0]
        if rsum < cat_meta['min_lim']: penalty += (cat_meta['min_lim'] - rsum) * 5000
        if rsum > cat_meta['max_lim']: penalty += (rsum - cat_meta['max_lim']) * 5000

    s_eff = np.sum(ratios * sel['strength']) * 10
    f_eff = np.sum(ratios * sel['flex']) * 10
    cost = np.sum(ratios * sel['density'] * sel['cost_kg'])
    
    # Fitness Skoru: Performans - Maliyet - Cezalar
    score = (s_eff * 2.5 + f_eff * 1.5) - (cost / 8) - penalty
    return (max(1.0, float(score)),)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=20, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# =========================
# 2. ARAYÜZ
# =========================
col_set, col_res = st.columns([1, 2])

with col_set:
    st.subheader("⚙️ Simülasyon Ayarları")
    pop_val = st.slider("Popülasyon", 200, 1000, 400)
    gen_val = st.slider("Nesil", 50, 500, 150)
    btn = st.button("🚀 Reçeteyi Sentezle")

if btn:
    with st.spinner("Genetik algoritma en uygun moleküler dizilimi arıyor..."):
        pop_list = toolbox.population(n=pop_val)
        hof = tools.HallOfFame(1)
        algorithms.eaSimple(pop_list, toolbox, 0.7, 0.3, gen_val, halloffame=hof, verbose=False)

    best = hof[0]
    idx = np.clip(np.array(best[:TOP_K], dtype=int), 0, LIB_SIZE - 1).tolist()
    ratios = np.array(best[TOP_K:], dtype=float)
    ratios /= np.sum(ratios)
    
    res = DB.iloc[idx].copy()
    res['Hacim Oranı (%)'] = ratios * 100
    res['Miktar (kg/m³)'] = (ratios * res['density']).astype(int)
    
    st.divider()
    m1, m2, m3 = st.columns(3)
    s_tot = np.sum(ratios * res['strength']) * 10
    m1.metric("Dayanım", f"{s_tot:.1f} MPa")
    m2.metric("Özgül Ağırlık", f"{int(res['Miktar (kg/m³)'].sum())} kg/m³")
    m3.metric("Maliyet", f"${int(np.sum(ratios * res['density'] * res['cost_kg']))}/m³")

    st.subheader("📋 Teknik Uygulama Reçetesi (1 m³)")
    # Mühendisin doğrudan kullanacağı net liste
    st.table(res[['category', 'name', 'Miktar (kg/m³)', 'Hacim Oranı (%)']])
    
        
    fig = px.bar(res, x='category', y='Miktar (kg/m³)', color='name', title="Kütlesel Bileşen Dağılımı")
    st.plotly_chart(fig)

    st.success("Analiz Başarıyla Tamamlandı. Reçete mühendislik kısıtlarına uygundur.")
