import streamlit as st
import numpy as np
import pandas as pd
import random
from deap import base, creator, tools, algorithms
import plotly.express as px

# =========================
# 1. VERİ KÜTÜPHANESİ (HIZLI VE GÜVENLİ)
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
# 2. GENETİK YAPI (FIXED)
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

# =========================
# 3. EVALUATE (KEYERROR FİXED)
# =========================
def evaluate(individual):
    # İndeksleri tam sayıya zorla ve sınırla
    indices = [int(max(0, min(len(db)-1, x))) for x in individual[:TOP_K]]
    ratios = np.array(individual[TOP_K:], dtype=float)
    
    # Veriyi tek seferde çek (iloc[i]['column'] hatasından kaçınmak için values kullan)
    sub_df = db.iloc[indices]
    strengths = sub_df['strength'].values
    flexibilities = sub_df['flex'].values
    costs_kg = sub_df['cost_kg'].values
    densities = sub_df['density'].values
    limits = sub_df['max_lim'].values

    # 1. Kısıt: Fiziksel Limitler
    # Oranları normalize etmeden önce limitlere göre kırp
    ratios = np.clip(ratios, 0, 1)
    for i in range(TOP_K):
        ratios[i] = min(ratios[i], limits[i])
    
    # 2. Hacimsel Normalizasyon (Sum = 1.0 m3)
    sum_r = np.sum(ratios)
    if sum_r == 0: return (0,)
    ratios = ratios / sum_r

    # 3. Hesaplamalar
    s_total = np.sum(ratios * strengths) * 100
    f_total = np.sum(ratios * flexibilities) * 100
    cost_total = np.sum(ratios * densities * costs_kg)
    
    # Deprem Simülasyonu (Empirik Formül)
    toughness = (s_total * 0.6) + (f_total * 1.4)
    quake_res = toughness / 12

    # Fitness: Dayanım ve Esnekliği ödüllendir, maliyeti cezalandır
    score = (s_total * 1.5) + (f_total * 2.5) + (quake_res * 20)
    score -= (cost_total / 8) # Maliyet baskısı
    
    # Ceza: Eğer çok pahalıysa veya dayanım çok düşükse
    if cost_total > 550: score -= (cost_total - 550) * 3
    if s_total < 40: score -= 200

    return (max(1, score),)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint) # Not: Daha güvenli bir cx için cxUniform denenebilir
toolbox.register("mutate", tools.mutGaussian, mu=100, sigma=50, indpb=0.1) # İndeksler için geniş mutasyon
toolbox.register("select", tools.selTournament, tournsize=3)

# =========================
# 4. ARAYÜZ
# =========================
st.set_page_config(page_title="Pro-Material AI", layout="wide")
st.title("🛡️ Civil-AI: Profesyonel Malzeme Sentezleyici")

[attachment_0](attachment)

col1, col2 = st.columns([1, 2])
with col1:
    pop_size = st.number_input("Popülasyon", 50, 1000, 300)
    gens = st.number_input("Nesil", 10, 2000, 100)
    btn = st.button("🧬 Evrimi Simüle Et")

if btn:
    pop = toolbox.population(n=int(pop_size))
    hof = tools.HallOfFame(1)
    
    with st.spinner("Genetik algoritma çaprazlanıyor..."):
        algorithms.eaSimple(pop, toolbox, 0.7, 0.2, int(gens), halloffame=hof, verbose=False)

    best = hof[0]
    indices = [int(max(0, min(len(db)-1, x))) for x in best[:TOP_K]]
    raw_ratios = np.array(best[TOP_K:])
    
    # Nihai gösterim için tekrar hesapla
    final_df = db.iloc[indices].copy()
    limits = final_df['max_lim'].values
    processed_ratios = np.clip(raw_ratios, 0, 1)
    for i in range(TOP_K):
        processed_ratios[i] = min(processed_ratios[i], limits[i])
    processed_ratios /= np.sum(processed_ratios)
    
    final_df['Reçete Oranı (%)'] = processed_ratios * 100
    
    # METRİKLER
    s_f = np.sum(processed_ratios * final_df['strength'].values) * 100
    f_f = np.sum(processed_ratios * final_df['flex'].values) * 100
    c_f = np.sum(processed_ratios * final_df['density'].values * final_df['cost_kg'].values)
    q_f = int(((s_f * 0.6) + (f_f * 1.4)) / 12)

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dayanım", f"{s_f:.1f} MPa")
    m2.metric("Süneklik", f"{f_f:.1f}")
    m3.metric("Maliyet", f"${c_f:.2f}/m³")
    m4.metric("Deprem Ömrü", f"{q_f} Şiddetli Sarsıntı")

    # GÖRSEL
    c_left, c_right = st.columns(2)
    with c_left:
        st.dataframe(final_df[['category', 'name', 'Reçete Oranı (%)']], use_container_width=True)
    with c_right:
        fig = px.pie(final_df, values='Reçete Oranı (%)', names='name', hole=0.4, title="Hacimsel Dağılım")
        st.plotly_chart(fig)
