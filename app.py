import streamlit as st
import numpy as np
import pandas as pd
import random
from deap import base, creator, tools, algorithms
import plotly.express as px

# =========================
# 0. GLOBAL SETTINGS
# =========================
RNG = np.random.default_rng(42)
TOP_K = 12
LIB_SIZE = 3000

st.set_page_config(page_title="Structural AI Engine v3.1", layout="wide")
st.title("🏗️ Structural AI Engine v3.1 — Professional Engineering Simulator")

# =========================
# 1. PARAMETRIC MATERIAL PHYSICS LIBRARY
# =========================
@st.cache_data
def build_material_library(n=LIB_SIZE):
    rows = []
    categories = {
        "Binder": {"min": 0.15, "max": 0.40, "density": 3150},
        "Aggregate": {"min": 0.55, "max": 0.80, "density": 2700},
        "Organic": {"min": 0.00, "max": 0.10, "density": 1400},
        "Polymer": {"min": 0.00, "max": 0.05, "density": 1100},
        "Nano": {"min": 0.00, "max": 0.03, "density": 2200},
        "Water": {"min": 0.08, "max": 0.20, "density": 1000},
    }
    
    for i in range(n):
        cat = random.choice(list(categories.keys()))
        cfg = categories[cat]
        silica = RNG.uniform(0.2, 0.8)
        nano_v = RNG.uniform(0.0, 0.3)
        polymer_v = RNG.uniform(0.0, 0.4)
        water_v = RNG.uniform(0.1, 0.4)

        strength = (40 + 120 * silica + 30 * nano_v + 15 * polymer_v - 60 * water_v) / 10
        ductility = (5 + 40 * RNG.uniform(0, 0.5) + 60 * polymer_v + 20 * nano_v - 30 * silica) / 10
        cost = (0.02 + 0.5 * nano_v + 0.3 * polymer_v)

        rows.append([
            f"{cat}_{i}", cat, max(0.1, strength), max(0.1, ductility), cost,
            cfg["density"], cfg["min"], cfg["max"]
        ])

    return pd.DataFrame(rows, columns=["name", "category", "strength", "ductility", "cost_kg", "density", "min_lim", "max_lim"])

DB = build_material_library()

# =========================
# 2. EVOLUTIONARY STRUCTURE
# =========================
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def create_individual():
    idx = random.sample(range(len(DB)), TOP_K)
    ratios = [random.random() for _ in range(TOP_K)]
    return creator.Individual(idx + ratios)

toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# =========================
# 3. PHYSICAL MODELS & EVALUATION
# =========================
def saturation(x, xmax):
    return xmax * (1 - np.exp(-x / xmax))

def evaluate(ind):
    # CRITICAL FIX: Ensure indices are within [0, LIB_SIZE-1] and are integers
    idx = np.clip(np.array(ind[:TOP_K], dtype=int), 0, LIB_SIZE - 1).tolist()
    raw = np.array(ind[TOP_K:], dtype=float)
    
    if np.sum(raw) <= 0: return (1.0,)
    ratios = raw / np.sum(raw)
    
    sel = DB.iloc[idx] # No more IndexError here

    # Constraints & Penalties
    penalty = 0
    for cat in sel['category'].unique():
        rsum = np.sum(ratios[sel['category'] == cat])
        rows = sel[sel['category'] == cat]
        mn, mx = rows['min_lim'].iloc[0], rows['max_lim'].iloc[0]
        if rsum < mn: penalty += (mn - rsum) * 5000
        if rsum > mx: penalty += (rsum - mx) * 5000

    strength = saturation(np.sum(ratios * sel['strength']) * 10, 150)
    duct = saturation(np.sum(ratios * sel['ductility']) * 10, 50)
    density = np.sum(ratios * sel['density'])
    cost = np.sum(ratios * sel['density'] * sel['cost_kg'])
    
    capacity = (strength / (density / 1000)) * (1 + duct / 100)
    
    # Monte Carlo Failure Probability
    collapses = 0
    for _ in range(20):
        Sa = RNG.lognormal(0.0, 0.6, 15)
        if np.sum((Sa / capacity)**2) >= 1.0: collapses += 1
    
    reliability = 1 - (collapses / 20)
    aps = (strength * 2 + duct * 1.5 + reliability * 100) * reliability - (cost / 10) - penalty
    
    return (max(1.0, float(aps)),)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=20, indpb=0.15)
toolbox.register("select", tools.selTournament, tournsize=3)

# =========================
# 4. UI & EXECUTION
# =========================
col1, col2 = st.columns([1, 2])
with col1:
    pop_size = st.slider("Population", 200, 1000, 400)
    gens = st.slider("Generations", 50, 500, 150)
    run = st.button("🚀 Run Engine v3.1")

if run:
    with st.spinner("Analyzing Material Matrix..."):
        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
        algorithms.eaSimple(pop, toolbox, 0.7, 0.2, gens, halloffame=hof, verbose=False)

    best = hof[0]
    idx = np.clip(np.array(best[:TOP_K], dtype=int), 0, LIB_SIZE - 1).tolist()
    ratios = np.array(best[TOP_K:])
    ratios /= np.sum(ratios)
    res = DB.iloc[idx].copy()
    res['Ratio %'] = ratios * 100

    st.divider()
    # Metrics
    s_f = saturation(np.sum(ratios * res['strength']) * 10, 150)
    d_f = saturation(np.sum(ratios * res['ductility']) * 10, 50)
    c_f = np.sum(ratios * res['density'] * res['cost_kg'])
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Structural Strength", f"{s_f:.1f} MPa")
    m2.metric("Ductility Index", f"{d_f:.1f}")
    m3.metric("Final Cost", f"${c_f:.2f} / m³")

    [attachment_0](attachment)

    l, r = st.columns(2)
    with l:
        st.subheader("Composition Table")
        st.dataframe(res[['category', 'name', 'Ratio %']], use_container_width=True)
    with r:
        fig = px.pie(res, values='Ratio %', names='name', hole=0.4, title="Mix Design")
        st.plotly_chart(fig)
