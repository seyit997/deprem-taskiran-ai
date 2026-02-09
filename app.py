import streamlit as st
import numpy as np
import pandas as pd
import random
from deap import base, creator, tools, algorithms
import plotly.express as px

# =========================
# 1. BÜYÜK VERİ: SENTETİK MALZEME JENERATÖRÜ
# =========================
# Doğadaki tüm malzemeleri temsil eden 2000+ maddelik sentetik kütüphane
def generate_global_material_library(size=2000):
    categories = ["Bağlayıcı", "Agrega", "Nano-Teknolojik", "Polimer", "Geri Dönüşüm", "Doğal Lif"]
    data = []
    for i in range(size):
        cat = random.choice(categories)
        data.append({
            "name": f"{cat}_{i+1}",
            "category": cat,
            "strength": random.uniform(0.1, 5.0),    # Dayanım spektrumu
            "flexibility": random.uniform(0.1, 3.0), # Süneklik spektrumu
            "cost": random.uniform(0.01, 10.0),      # Ucuz kumdan pahalı CNT'ye
            "density": random.uniform(500, 4000),    # Hafif beton - Ağır çelik
            "degradation": random.uniform(0.05, 0.5) # Çevresel bozulma
        })
    return pd.DataFrame(data)

# Veritabanını oluştur
if 'material_db' not in st.session_state:
    st.session_state.material_db = generate_global_material_library(2500)

db = st.session_state.material_db

# =========================
# 2. GENETİK ALGORİTMA AYARLARI
# =========================
# Genetik algoritma "Binlerce madde arasından en iyi 10'luyu seç ve oranla" şeklinde çalışacak
TOP_K = 12 # Karışımda kullanılacak maksimum farklı madde sayısı

if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Birey: [Malzeme_Index_1, Oran_1, Malzeme_Index_2, Oran_2 ...]
toolbox.register("attr_idx", random.randint, 0, len(db) - 1)
toolbox.register("attr_float", random.random)

def create_individual():
    ind = []
    for _ in range(TOP_K):
        ind.append(random.randint(0, len(db) - 1)) # Malzeme seçimi
        ind.append(random.random())               # Miktar
    return creator.Individual(ind)

toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# =========================
# 3. HIZLI FİZİKSEL DEĞERLENDİRME (Vektörize)
# =========================
def evaluate(individual):
    indices = individual[0::2]
    raw_ratios = np.array(individual[1::2])
    ratios = raw_ratios / np.sum(raw_ratios)
    
    # Seçilen malzemelerin verilerini çek
    selected_materials = db.iloc[indices]
    
    # Performans hesaplama (Matris çarpımı hızıyla)
    strength = np.sum(ratios * selected_materials['strength'].values) * 100
    flex = np.sum(ratios * selected_materials['flexibility'].values) * 100
    cost = np.sum(ratios * selected_materials['cost'].values * selected_materials['density'].values)
    degradation = np.sum(ratios * selected_materials['degradation'].values) * 50

    # Hedefler
    cost_penalty = max(0, cost - 500) * 5
    fitness = (min(strength, 150) * 2) + (min(flex, 100) * 1.5) - degradation - cost_penalty
    
    return (max(0, fitness),)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0.5, sigma=0.2, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=4)

# =========================
# 4. ARAYÜZ
# =========================
st.set_page_config(page_title="Global Malzeme Evrimi", layout="wide")
st.title("🌐 Global Evrimsel Malzeme Sentezleyici")
st.write(f"Şu anda veritabanında **{len(db)}** farklı madde (doğal ve sentetik) taranıyor.")



col_a, col_b = st.columns(2)
pop_size = col_a.slider("Popülasyon Genişliği", 200, 1000, 500)
gens = col_b.slider("Simülasyon Derinliği (Nesil)", 100, 2000, 500)

if st.button("🧬 Binlerce Madde İçinde Evrimi Başlat"):
    with st.spinner("Yapay zeka doğadaki elementleri kombine ediyor..."):
        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
        
        algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=gens, halloffame=hof, verbose=False)

    # Sonuçları İşle
    best = hof[0]
    best_indices = best[0::2]
    best_ratios = np.array(best[1::2])
    best_ratios = best_ratios / np.sum(best_ratios)
    
    res_df = db.iloc[best_indices].copy()
    res_df['Karışım Oranı (%)'] = np.round(best_ratios * 100, 2)
    
    # Grafik ve Tablo
    st.subheader("🏆 Evrim Sonucu Oluşan En Güçlü Hibrit Karışım")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.write(res_df[['name', 'category', 'Karışım Oranı (%)']])
    with c2:
        fig = px.sunburst(res_df, path=['category', 'name'], values='Karışım Oranı (%)', title="Malzeme Dağılımı")
        st.plotly_chart(fig)

    # Performans Metrikleri
    st.divider()
    m1, m2, m3 = st.columns(3)
    final_strength = np.sum(best_ratios * res_df['strength'].values) * 100
    final_flex = np.sum(best_ratios * res_df['flexibility'].values) * 100
    final_cost = np.sum(best_ratios * res_df['cost'].values * res_df['density'].values)
    
    m1.metric("Bileşik Dayanım", f"{final_strength:.2f} MPa")
    m2.metric("Süneklik Katsayısı", f"{final_flex:.2f}")
    m3.metric("Tahmini Maliyet", f"${final_cost:.2f} /m³")
