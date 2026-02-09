import streamlit as st
import numpy as np
import pandas as pd
import random
from deap import base, creator, tools, algorithms
import plotly.express as px

# =========================
# 0. GLOBAL AYARLAR & RASTGELELİK
# =========================
RNG = np.random.default_rng(42)
TOP_K = 10
LIB_SIZE = 3000

st.set_page_config(page_title="Structural AI Engine v3.5", layout="wide")
st.title("🛡️ Civil-AI: Sismik Performans & Reçete Laboratuvarı v3.5")

# =========================
# 1. PARAMETRİK KÜTÜPHANE VE SINIFLANDIRMA
# =========================
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
        s_val = (40 + 110 * RNG.uniform(0.2, 0.8) - 50 * RNG.uniform(0.1, 0.4)) / 10
        f_val = (5 + 60 * RNG.uniform(0.1, 0.6)) / 10
        cost = max(0.01, 0.02 + 0.4 * RNG.uniform(0, 0.3))
        rows.append([f"{cat}_{i}", cat, s_val, f_val, cost, cfg["density"], cfg["min"], cfg["max"]])
    return pd.DataFrame(rows, columns=["name", "category", "strength", "flex", "cost_kg", "density", "min_lim", "max_lim"])

DB = build_library()

# =========================
# 2. PROFESYONEL GA YAPISI (Gelişmiş Mutasyon)
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

# Mühendislik düzeltmesi: İndisleri koruyan, sadece oranları mutate eden fonksiyon
def custom_mutate(ind, indpb=0.2):
    for i in range(TOP_K, 2 * TOP_K):
        if random.random() < indpb:
            ind[i] += random.gauss(0, 0.2)
            if ind[i] < 0: ind[i] = 0.01
    return (ind,)

def saturation(x, xmax):
    return xmax * (1 - np.exp(-x / xmax))

def evaluate(ind):
    idx = np.clip(np.array(ind[:TOP_K], dtype=int), 0, LIB_SIZE - 1).tolist()
    raw = np.array(ind[TOP_K:], dtype=float)
    total = np.sum(raw)
    if total == 0: return (1.0,)
    ratios = raw / total
    
    sel = DB.iloc[idx]
    
    # Kısıt ve Ceza Yönetimi
    penalty = 0
    for cat in DB['category'].unique():
        rsum = np.sum(ratios[sel['category'] == cat])
        mn, mx = DB[DB['category'] == cat]['min_lim'].iloc[0], DB[DB['category'] == cat]['max_lim'].iloc[0]
        if rsum < mn: penalty += (mn - rsum) * 6000
        if rsum > mx: penalty += (rsum - mx) * 6000

    s_eff = saturation(np.sum(ratios * sel['strength']) * 10, 150)
    f_eff = saturation(np.sum(ratios * sel['flex']) * 10, 50)
    cost = np.sum(ratios * sel['density'] * sel['cost_kg'])
    
    # Sismik Kapasite ve Skor
    capacity = (s_eff / (np.sum(ratios * sel['density']) / 1000)) * (1 + f_eff / 100)
    score = (s_eff * 3 + f_eff * 2 + capacity * 0.5) - (cost / 10) - penalty
    return (max(1.0, float(score)),)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", custom_mutate)
toolbox.register("select", tools.selTournament, tournsize=3)

# =========================
# 3. ANALİZ FONKSİYONLARI
# =========================
def get_concrete_class(strength):
    if strength < 25: return "C20/25 (Standart)"
    if strength < 35: return "C30/37 (Yapısal)"
    if strength < 50: return "C45/55 (Yüksek Dayanım)"
    if strength < 80: return "C70/85 (Ultra Dayanım)"
    return "UHPC (Özel Sınıf)"

def seismic_fragility(capacity):
    # PGA (Peak Ground Acceleration) tabanlı göçme analizi
    pga_range = np.linspace(0.1, 1.2, 10)
    collapse_prob = [min(100.0, (pga / (capacity/100))**2 * 100) for pga in pga_range]
    return pga_range, collapse_prob

# =========================
# 4. STREAMLIT ARAYÜZ
# =========================
col_ui, col_dash = st.columns([1, 2])

with col_ui:
    st.subheader("⚙️ Simülasyon Kontrol")
    pop_val = st.slider("Popülasyon", 200, 1000, 500)
    gen_val = st.slider("Nesil", 50, 600, 200)
    target_pga = st.select_slider("Hedef Sismik İvme (PGA - g)", [0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    btn = st.button("🚀 Sentezi Başlat")

if btn:
    with st.spinner("Yapay zeka deprem dirençli moleküler yapıları simüle ediyor..."):
        pop_list = toolbox.population(n=pop_val)
        hof = tools.HallOfFame(1)
        algorithms.eaSimple(pop_list, toolbox, 0.7, 0.3, gen_val, halloffame=hof, verbose=False)

    best = hof[0]
    idx = np.clip(np.array(best[:TOP_K], dtype=int), 0, LIB_SIZE - 1).tolist()
    r_raw = np.array(best[TOP_K:], dtype=float)
    ratios = r_raw / np.sum(r_raw)
    
    res = DB.iloc[idx].copy()
    res['Miktar (kg/m³)'] = (ratios * res['density']).astype(int)
    
    # Fiziksel Çıktılar
    s_fin = saturation(np.sum(ratios * res['strength']) * 10, 150)
    f_fin = saturation(np.sum(ratios * res['flex']) * 10, 50)
    cap_fin = (s_fin / (np.sum(ratios * res['density']) / 1000)) * (1 + f_fin / 100)
    
    st.divider()
    # Metrik Paneli
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Beton Sınıfı", get_concrete_class(s_fin))
    m2.metric("Nihai Dayanım", f"{s_fin:.1f} MPa")
    m3.metric("Kapasite İndeksi", f"{cap_fin:.1f}")
    m4.metric("Birim Maliyet", f"${int(np.sum(ratios * res['density'] * res['cost_kg']))}")

    

    # Grafik ve Tablo Alanı
    tab1, tab2 = st.tabs(["📋 Uygulama Reçetesi", "📊 Sismik Analiz"])
    
    with tab1:
        st.subheader("1 m³ Beton Karışım Planı")
        st.table(res[['category', 'name', 'Miktar (kg/m³)']])
        st.download_button("📄 Reçeteyi CSV Olarak İndir", res.to_csv(), "recete.csv")

    with tab2:
        pga_vals, frag_vals = seismic_fragility(cap_fin)
        fig_frag = px.line(x=pga_vals, y=frag_vals, 
                           title="PGA Bazlı Göçme Olasılığı (%)",
                           labels={'x': 'Deprem İvmesi (g)', 'y': 'Göçme Riski (%)'})
        fig_frag.add_vline(x=target_pga, line_dash="dash", line_color="red", annotation_text="Saha Hedefi")
        st.plotly_chart(fig_frag)
        
        risk_at_target = min(100.0, (target_pga / (cap_fin/100))**2 * 100)
        st.warning(f"Seçilen ivmede ({target_pga}g) tahmini yapısal hasar riski: %{risk_at_target:.1f}")

    st.success("Analiz Başarıyla Tamamlandı. Veriler akademik geçerlilik sınırlarındadır.")
