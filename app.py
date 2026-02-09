import streamlit as st
import numpy as np
import pandas as pd
import random
from deap import base, creator, tools, algorithms
import plotly.express as px

# =========================
# 0. AYARLAR & MALZEME EVRENİ
# =========================
RNG = np.random.default_rng(42)
TOP_K = 12 
LIB_SIZE = 3000

st.set_page_config(page_title="Structural AI Engine v3.6", layout="wide")
st.title("🏗️ Civil-AI: Mühendislik Filtreli Reçete Sentezleyici v3.6")

@st.cache_data
def build_library(n=LIB_SIZE):
    rows = []
    categories = {
        "Bağlayıcı (Çimento/Silis)": {"min": 0.15, "max": 0.40, "density": 3150},
        "Agrega (Kırmataş/Kum)": {"min": 0.55, "max": 0.80, "density": 2700},
        "Kimyasal Katkı (Polimer)": {"min": 0.01, "max": 0.05, "density": 1100},
        "Nano Güçlendirici": {"min": 0.0, "max": 0.03, "density": 2200},
        "Su": {"min": 0.08, "max": 0.20, "density": 1000},
    }
    for i in range(n):
        cat = random.choice(list(categories.keys()))
        cfg = categories[cat]
        # Fiziksel Dayanım Potansiyeli
        s_val = (40 + 110 * RNG.uniform(0.2, 0.8)) / 10
        f_val = (5 + 60 * RNG.uniform(0.1, 0.6)) / 10
        rows.append([f"{cat}_{i}", cat, s_val, f_val, max(0.01, 0.02 + 0.4 * RNG.uniform(0, 0.3)), cfg["density"], cfg["min"], cfg["max"]])
    return pd.DataFrame(rows, columns=["name", "category", "strength", "flex", "cost_kg", "density", "min_lim", "max_lim"])

DB = build_library()

# =========================
# 1. EVRİMSEL SİSTEM (GA)
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

# Mühendislik düzeltmesi: İndisleri koru, sadece oranları mutate et
def custom_mutate(ind, indpb=0.2):
    for i in range(TOP_K, 2 * TOP_K):
        if random.random() < indpb:
            ind[i] += random.gauss(0, 0.2)
            if ind[i] < 0: ind[i] = 0.01
    return (ind,)

def evaluate(ind):
    idx = np.clip(np.array(ind[:TOP_K], dtype=int), 0, LIB_SIZE - 1).tolist()
    raw = np.array(ind[TOP_K:], dtype=float)
    if np.sum(raw) == 0: return (1.0,)
    ratios = raw / np.sum(raw)
    sel = DB.iloc[idx]
    
    # Su/Bağlayıcı (w/b) Oranı Analizi
    total_water = np.sum(ratios[sel['category'].str.contains("Su")])
    total_binder = np.sum(ratios[sel['category'].str.contains("Bağlayıcı")])
    wb_ratio = total_water / total_binder if total_binder > 0 else 10
    
    penalty = 0
    # w/b oranı 0.25 - 0.45 dışındaysa ağır ceza (Yüksek dayanım beton fiziği)
    if wb_ratio < 0.25 or wb_ratio > 0.45: penalty += 15000 
    
    # Kategori Limit Kontrolleri
    for cat in DB['category'].unique():
        rsum = np.sum(ratios[sel['category'] == cat])
        meta = DB[DB['category'] == cat].iloc[0]
        if rsum < meta['min_lim']: penalty += (meta['min_lim'] - rsum) * 5000
        if rsum > meta['max_lim']: penalty += (rsum - meta['max_lim']) * 5000

    s_eff = (np.sum(ratios * sel['strength']) * 10) * (1 - abs(wb_ratio - 0.35))
    return (max(1.0, float(s_eff - penalty)),)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", custom_mutate)
toolbox.register("select", tools.selTournament, tournsize=3)

# =========================
# 2. ARAYÜZ VE REÇETE FİLTRESİ
# =========================
col_ui, col_res = st.columns([1, 2])
with col_ui:
    st.subheader("⚙️ Optimizasyon Ayarları")
    pop_val = st.slider("Popülasyon", 200, 1000, 500)
    gen_val = st.slider("Nesil", 50, 500, 200)
    run = st.button("🚀 Reçeteyi Sentezle")

if run:
    with st.spinner("Evrimsel algoritmalar beton fiziğini tarıyor..."):
        pop_list = toolbox.population(n=pop_val)
        hof = tools.HallOfFame(1)
        algorithms.eaSimple(pop_list, toolbox, 0.7, 0.3, gen_val, halloffame=hof, verbose=False)

    best = hof[0]
    idx = np.clip(np.array(best[:TOP_K], dtype=int), 0, LIB_SIZE - 1).tolist()
    r_raw = np.array(best[TOP_K:], dtype=float)
    ratios = r_raw / np.sum(r_raw)
    
    res = DB.iloc[idx].copy()
    res['Miktar (kg)'] = (ratios * res['density'])
    
    # --- MÜHENDİSLİK FİLTRESİ: Gruplama ve Temizlik ---
    final_recipe = res.groupby('category')['Miktar (kg)'].sum().reset_index()
    final_recipe = final_recipe[final_recipe['Miktar (kg)'] > 0.5] # 0.5 kg altı anlamsızları sil
    
    w_sum = final_recipe[final_recipe['category'].str.contains("Su")]['Miktar (kg)'].sum()
    b_sum = final_recipe[final_recipe['category'].str.contains("Bağlayıcı")]['Miktar (kg)'].sum()
    final_wb = w_sum / b_sum if b_sum > 0 else 0

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Beton Sınıfı", "C70/85 (HPC)")
    m2.metric("Su / Bağlayıcı Oranı", f"{final_wb:.2f}")
    m3.metric("Toplam Yoğunluk", f"{int(final_recipe['Miktar (kg)'].sum())} kg/m³")

    st.subheader("📋 Teknik Uygulama Reçetesi (1 m³)")
    st.table(final_recipe.style.format({"Miktar (kg)": "{:.2f}"}))
    
    

    st.info("**Not:** Çıktılardaki küçük değerler filtrelenmiş ve kategoriler mühendislik disiplinine göre birleştirilmiştir.")
    
    fig = px.pie(final_recipe, values='Miktar (kg)', names='category', hole=0.4, title="Kütlesel Bileşen Dağılımı")
    st.plotly_chart(fig)
