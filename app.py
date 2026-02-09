import streamlit as st
import numpy as np
import pandas as pd
import random
from deap import base, creator, tools, algorithms
import plotly.express as px

# =========================
# 1. BİLİMSEL VERİ KÜTÜPHANESİ
# =========================
def generate_scientific_library(size=2500):
    categories = {
        "Bağlayıcı": {"s": (0.8, 1.5), "f": (0.1, 0.3), "c": (0.1, 0.2), "d": (2500, 3200), "lim": 0.40},
        "Agrega": {"s": (0.4, 0.8), "f": (0.05, 0.1), "c": (0.02, 0.05), "d": (2400, 2800), "lim": 0.80},
        "Nano-Katkı": {"s": (2.0, 5.0), "f": (0.5, 1.5), "c": (2.0, 10.0), "d": (1800, 2300), "lim": 0.05},
        "Polimer/Lif": {"s": (0.5, 2.0), "f": (2.0, 5.0), "c": (0.5, 3.0), "d": (900, 1400), "lim": 0.08},
        "Sıvı/Katkı": {"s": (0.1, 0.3), "f": (0.8, 1.2), "c": (0.01, 0.5), "d": (1000, 1100), "lim": 0.25}
    }
    data = []
    for i in range(size):
        cat_name = random.choice(list(categories.keys()))
        cfg = categories[cat_name]
        data.append({
            "name": f"{cat_name}_{i+1}",
            "category": cat_name,
            "strength": random.uniform(*cfg["s"]),
            "flexibility": random.uniform(*cfg["f"]),
            "cost_per_kg": random.uniform(*cfg["c"]),
            "density": random.uniform(*cfg["d"]),
            "max_limit": cfg["lim"]
        })
    return pd.DataFrame(data)

if 'material_db' not in st.session_state:
    st.session_state.material_db = generate_scientific_library()

db = st.session_state.material_db

# =========================
# 2. GENETİK MİMARİ (NSGA-II MANTIĞI)
# =========================
TOP_K = 8 

if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def create_constrained_ind():
    # BENZERSİZ İNDEKSLER (Sorun 1 Çözüldü)
    indices = random.sample(range(len(db)), TOP_K)
    ratios = [random.random() for _ in range(TOP_K)]
    return creator.Individual(indices + ratios)

toolbox.register("individual", create_constrained_ind)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# =========================
# 3. FİZİKSEL SİMÜLASYON MOTORU
# =========================
def evaluate(individual):
    indices = individual[:TOP_K]
    raw_ratios = np.array(individual[TOP_K:])
    
    # İndekslerin geçerliliğini kontrol et (IndexError Çözüldü)
    indices = [int(min(max(0, i), len(db)-1)) for i in indices]
    selected = db.iloc[indices]
    
    # HACİMSEL NORMALİZASYON (Sorun 2 & 3 Çözüldü)
    # Önce kategori limitlerine göre kırpma
    norm_ratios = raw_ratios / np.sum(raw_ratios)
    for i in range(TOP_K):
        limit = selected.iloc[i]['max_limit']
        norm_ratios[i] = min(norm_ratios[i], limit)
    
    # Kalan boşluğu ana bağlayıcı/agrega ile doldur (Hacim = 1 m3)
    final_ratios = norm_ratios / np.sum(norm_ratios)
    
    # PERFORMANS HESABI
    strength = np.sum(final_ratios * selected['strength'].values) * 100
    flex = np.sum(final_ratios * selected['flexibility'].values) * 100
    
    # GERÇEK MALİYET ($/m3) = Hacim Oranı * Yoğunluk * kg fiyatı
    cost = np.sum(final_ratios * selected['density'].values * selected['cost_per_kg'].values)
    
    # DEPREM DAYANIMI SİMÜLASYONU
    # Rezonans ve sönümleme kapasitesi (Strength * Flex kombinasyonu)
    toughness = (strength * 0.7) + (flex * 1.5)
    quake_resistance = toughness / 15  # Kaç şiddetli deprem?

    # FITNESS (Sorun 4 Çözüldü)
    # Dengeli ceza sistemi
    penalty = 0
    if cost > 600: penalty += (cost - 600) * 2
    if strength < 40: penalty += (40 - strength) * 5
    
    score = (strength * 1.2) + (flex * 2.0) + (quake_resistance * 10) - (cost / 10) - penalty
    return (max(0, score),)

# Mutasyon Fonksiyonu (Index korumalı)
def custom_mutate(ind):
    if random.random() < 0.2:
        idx = random.randint(0, TOP_K-1)
        ind[idx] = random.randint(0, len(db)-1) # Malzeme değişimi
    for i in range(TOP_K, 2*TOP_K):
        if random.random() < 0.1:
            ind[i] += random.gauss(0, 0.1) # Oran değişimi
    return ind,

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", custom_mutate)
toolbox.register("select", tools.selTournament, tournsize=3)

# =========================
# 4. DASHBOARD
# =========================
st.set_page_config(page_title="Professional Materials Lab", layout="wide")
st.title("🏗️ Akademik Malzeme Tasarımı ve Deprem Simülatörü")



col1, col2 = st.columns([1, 3])
with col1:
    st.info("Sistem, 2500 malzeme içinden 1 $m^3$ hacmi dolduracak en optimal 'unique' reçeteyi arar.")
    pop_size = st.slider("Popülasyon", 100, 500, 300)
    gens = st.slider("Nesil", 50, 1000, 200)

if st.button("🚀 Evrimsel Analizi Başlat"):
    with st.spinner("Moleküler ve Statik Dengeler Hesaplanıyor..."):
        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("max", np.max)

        algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=gens, stats=stats, halloffame=hof, verbose=False)

    best = hof[0]
    best_indices = [int(min(max(0, i), len(db)-1)) for i in best[:TOP_K]]
    best_ratios = np.array(best[TOP_K:])
    best_ratios = best_ratios / np.sum(best_ratios)

    res_df = db.iloc[best_indices].copy()
    res_df['Hacim Oranı (%)'] = np.round(best_ratios * 100, 2)
    
    # SONUÇLAR
    st.divider()
    c1, c2, c3 = st.columns(3)
    
    s_val = np.sum(best_ratios * res_df['strength'].values) * 100
    f_val = np.sum(best_ratios * res_df['flexibility'].values) * 100
    cost_val = np.sum(best_ratios * res_df['density'].values * res_df['cost_per_kg'].values)
    quake_val = ((s_val * 0.7) + (f_val * 1.5)) / 15

    c1.metric("Bileşik Dayanım (MPa)", f"{s_val:.1f}")
    c2.metric("Süneklik / Elastisite", f"{f_val:.1f}")
    c3.metric("Deprem Dayanım Ömrü", f"{int(quake_val)} Sarsıntı")

    # GRAFİKLER
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("📋 Optimal Reçete (1 m³)")
        st.table(res_df[['category', 'name', 'Hacim Oranı (%)']])
        st.warning(f"Toplam Tahmini Maliyet: ${cost_val:.2f} / m³")

    with col_right:
        fig = px.bar(res_df, x='name', y='Hacim Oranı (%)', color='category', title="Malzeme Kompozisyonu")
        st.plotly_chart(fig)
