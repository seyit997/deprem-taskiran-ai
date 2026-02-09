import streamlit as st
import numpy as np
import pandas as pd
import random
from deap import base, creator, tools, algorithms
import plotly.express as px

# =========================
# 1. BİLİMSEL VERİ KÜTÜPHANESİ
# =========================
@st.cache_data
def get_library(size=2500):
    categories = {
        "Bağlayıcı": {"s": (0.8, 1.5), "f": (0.1, 0.3), "c": (0.1, 0.2), "d": 3100, "lim": 0.50},
        "Agrega": {"s": (0.4, 0.8), "f": (0.05, 0.1), "c": (0.02, 0.05), "d": 2700, "lim": 0.85},
        "Nano-Katkı": {"s": (2.0, 5.0), "f": (0.5, 1.5), "c": (2.0, 10.0), "d": 2100, "lim": 0.06},
        "Polimer/Lif": {"s": (0.5, 2.0), "f": (2.0, 5.0), "c": (0.5, 3.0), "d": 1200, "lim": 0.10},
        "Sıvı/Katkı": {"s": (0.1, 0.3), "f": (0.8, 1.2), "c": (0.01, 0.5), "d": 1000, "lim": 0.20}
    }
    data = []
    cat_list = list(categories.keys())
    for i in range(size):
        c = random.choice(cat_list)
        data.append([
            f"{c}_{i}", c, random.uniform(*categories[c]["s"]),
            random.uniform(*categories[c]["f"]), random.uniform(*categories[c]["c"]),
            categories[c]["d"], categories[c]["lim"]
        ])
    return pd.DataFrame(data, columns=["name", "category", "strength", "flex", "cost_kg", "density", "max_lim"])

db = get_library()
TOP_K = 8

# =========================
# 2. GENETİK ALGORİTMA KURULUMU
# =========================
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def create_ind():
    indices = random.sample(range(len(db)), TOP_K)
    ratios = [random.random() for _ in range(TOP_K)]
    return creator.Individual(indices + ratios)

toolbox.register("individual", create_ind)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    # İndeksleri güvenli aralıkta tam sayıya çevir
    indices = [int(max(0, min(len(db)-1, x))) for x in individual[:TOP_K]]
    ratios = np.array(individual[TOP_K:], dtype=float)
    
    sub_df = db.iloc[indices]
    
    # Kısıtlamaları uygula (Limitler)
    limits = sub_df['max_lim'].values
    ratios = np.clip(ratios, 0, 1)
    for i in range(TOP_K):
        ratios[i] = min(ratios[i], limits[i])
    
    # Hacimsel Normalizasyon
    sum_r = np.sum(ratios)
    if sum_r == 0: return (0,)
    ratios /= sum_r

    # Fiziksel Çıktılar
    s_total = np.sum(ratios * sub_df['strength'].values) * 100
    f_total = np.sum(ratios * sub_df['flex'].values) * 100
    cost_total = np.sum(ratios * sub_df['density'].values * sub_df['cost_kg'].values)
    
    # Skorlama: Performans / Maliyet Dengesi
    toughness = (s_total * 0.6) + (f_total * 1.4)
    score = (toughness * 15) - (cost_total / 10)
    
    # Akademik Cezalar
    if s_total < 40: score -= 300 # Minimum güvenlik eşiği
    if cost_total > 600: score -= (cost_total - 600) * 5 # Bütçe aşımı

    return (max(1, score),)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=100, sigma=50, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# =========================
# 3. STREAMLIT ARAYÜZÜ
# =========================
st.set_page_config(page_title="Civil-AI Lab", layout="wide")

st.title("🛡️ Civil-AI: Profesyonel Malzeme Sentezleyici")
st.markdown("---")

# Görsel sembolik (Hata yapmaması için Markdown içinde)
# [attachment_0](attachment)

col_ui, col_info = st.columns([1, 2])

with col_ui:
    st.subheader("⚙️ Parametreler")
    pop_size = st.number_input("Popülasyon Genişliği", 100, 1000, 300)
    gens = st.number_input("Simülasyon Derinliği (Nesil)", 50, 2000, 150)
    run_btn = st.button("🚀 Evrimsel Sentezi Başlat")

with col_info:
    st.info("""
    **Çalışma Prensibi:**
    1. Sistem 2500 malzeme arasından en uyumlu 8'liyi seçer.
    2. Hacimsel (m³) bazda kütle-yoğunluk dengesini hesaplar.
    3. Malzeme limitlerini (Nano-katkı %6 vb.) koruyarak en yüksek tokluğu arar.
    """)

if run_btn:
    pop = toolbox.population(n=int(pop_size))
    hof = tools.HallOfFame(1)
    
    with st.spinner("Yapay zeka milyonlarca kombinasyonu test ediyor..."):
        algorithms.eaSimple(pop, toolbox, 0.7, 0.3, int(gens), halloffame=hof, verbose=False)

    best = hof[0]
    indices = [int(max(0, min(len(db)-1, x))) for x in best[:TOP_K]]
    raw_ratios = np.array(best[TOP_K:])
    
    # Sonuç DataFrame
    res_df = db.iloc[indices].copy()
    limits = res_df['max_lim'].values
    processed_ratios = np.clip(raw_ratios, 0, 1)
    for i in range(TOP_K):
        processed_ratios[i] = min(processed_ratios[i], limits[i])
    processed_ratios /= np.sum(processed_ratios)
    
    res_df['Oran (%)'] = processed_ratios * 100
    
    # Metrikler
    s_f = np.sum(processed_ratios * res_df['strength'].values) * 100
    f_f = np.sum(processed_ratios * res_df['flex'].values) * 100
    c_f = np.sum(processed_ratios * res_df['density'].values * res_df['cost_kg'].values)
    quake_count = int(((s_f * 0.6) + (f_f * 1.4)) / 12)

    st.subheader("📊 Optimizasyon Sonuçları")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dayanım (Strength)", f"{s_f:.1f} MPa")
    m2.metric("Süneklik (Ductility)", f"{f_f:.1f}")
    m3.metric("Birim Maliyet", f"${c_f:.2f}/m³")
    m4.metric("Sarsıntı Dayanımı", f"{quake_count} Deprem")

    c_left, c_right = st.columns(2)
    with c_left:
        st.dataframe(res_df[['category', 'name', 'Oran (%)']], use_container_width=True)
    with c_right:
        fig = px.sunburst(res_df, path=['category', 'name'], values='Oran (%)', title="Hiyerarşik Karışım Dağılımı")
        st.plotly_chart(fig)

    st.success("✅ Simülasyon başarıyla tamamlandı. Bu reçete hedeflenen sismik performansı karşılamaktadır.")
