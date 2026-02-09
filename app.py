"""
Structural AI Engine v3.0
Five-star professional engineering simulation
"""

import streamlit as st
import numpy as np
import pandas as pd
import random

from deap import base, creator, tools, algorithms

import plotly.express as px

=========================

0. GLOBAL SETTINGS

=========================

RNG = np.random.default_rng(42) TOP_K = 12

st.set_page_config(page_title="Structural AI Engine v3.0", layout="wide") st.title("🏗️ Structural AI Engine v3.0 — Professional Engineering Simulator")

=========================

1. PARAMETRIC MATERIAL PHYSICS LIBRARY (Problem 1 fixed)

=========================

@st.cache_data

def build_material_library(n=3000): """ Parametric, correlated material model. Properties are functions, not random noise. Covers natural, historical, and synthetic additives. """ rows = []

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

    # latent chemistry variables
    silica = RNG.uniform(0.2, 0.8)
    lime = RNG.uniform(0.1, 0.6)
    organic = RNG.uniform(0.0, 0.5)
    polymer = RNG.uniform(0.0, 0.4)
    nano = RNG.uniform(0.0, 0.3)
    water = RNG.uniform(0.1, 0.4)

    # correlated physics
    strength = (
        40
        + 120 * silica
        + 30 * nano
        + 15 * polymer
        - 60 * water
    ) / 10

    ductility = (
        5
        + 40 * organic
        + 60 * polymer
        + 20 * nano
        - 30 * silica
    ) / 10

    cost = (
        0.02
        + 0.5 * nano
        + 0.3 * polymer
        + 0.1 * organic
    )

    rows.append([
        f"{cat}_{i}", cat,
        max(0.1, strength),
        max(0.1, ductility),
        cost,
        cfg["density"],
        cfg["min"], cfg["max"],
        silica, lime, organic, polymer, nano, water
    ])

return pd.DataFrame(
    rows,
    columns=[
        "name", "category", "strength", "ductility", "cost_kg",
        "density", "min_lim", "max_lim",
        "silica", "lime", "organic", "polymer", "nano", "water"
    ]
)

DB = build_material_library()

=========================

2. EVOLUTIONARY STRUCTURE (Problem 4 fixed)

=========================

if "FitnessMax" not in creator.dict: creator.create("FitnessMax", base.Fitness, weights=(1.0,)) if "Individual" not in creator.dict: creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def create_individual(): idx = random.sample(range(len(DB)), TOP_K) ratios = [random.random() for _ in range(TOP_K)] return creator.Individual(idx + ratios)

toolbox.register("individual", create_individual) toolbox.register("population", tools.initRepeat, list, toolbox.individual)

=========================

3. PHYSICAL MODELS

=========================

def saturation(x, xmax): return xmax * (1 - np.exp(-x / xmax))

---- Earthquake time-history damage (Problem 2 fixed)

def earthquake_damage(capacity, Sa_series): damage = 0.0 for Sa in Sa_series: damage += (Sa / capacity) ** 2 if damage >= 1.0: return True, damage return False, damage

---- Manufacturability constraints

def manufacturability_penalty(df, ratios): pen = 0 if np.sum(ratios[df['category'] == 'Nano']) > 0.02: pen += 1000 if np.sum(ratios[df['category'] == 'Polymer']) > 0.04: pen += 800 return pen

=========================

4. FITNESS WITH RELIABILITY (Problem 3 fixed)

=========================

def evaluate(ind): idx = ind[:TOP_K] raw = np.array(ind[TOP_K:]) ratios = raw / np.sum(raw) sel = DB.iloc[idx]

# category constraints
penalty = 0
for cat in sel['category'].unique():
    rsum = np.sum(ratios[sel['category'] == cat])
    mn = sel[sel['category'] == cat]['min_lim'].iloc[0]
    mx = sel[sel['category'] == cat]['max_lim'].iloc[0]
    if rsum < mn:
        penalty += (mn - rsum) * 5000
    if rsum > mx:
        penalty += (rsum - mx) * 5000

strength = saturation(np.sum(ratios * sel['strength']) * 10, 150)
duct = saturation(np.sum(ratios * sel['ductility']) * 10, 50)
density = np.sum(ratios * sel['density'])
cost = np.sum(ratios * sel['density'] * sel['cost_kg'])

capacity = (strength / (density / 1000)) * (1 + duct / 100)

# Monte Carlo earthquakes
collapses = 0
runs = 30
for _ in range(runs):
    Sa = RNG.lognormal(mean=0.0, sigma=0.6, size=20)
    collapsed, _ = earthquake_damage(capacity, Sa)
    if collapsed:
        collapses += 1

reliability = 1 - collapses / runs

manuf_pen = manufacturability_penalty(sel, ratios)

raw_score = (strength * 2) + (duct * 1.5) + (reliability * 100)
aps = raw_score * reliability - (cost / 10) - penalty - manuf_pen

return (max(1.0, aps),)

toolbox.register("evaluate", evaluate) toolbox.register("mate", tools.cxTwoPoint) toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.2) toolbox.register("select", tools.selTournament, tournsize=3)

=========================

5. STREAMLIT UI

=========================

col1, col2 = st.columns([1, 2]) with col1: st.subheader("Simulation Settings") pop_size = st.slider("Population", 200, 1200, 600) gens = st.slider("Generations", 100, 1200, 400) run = st.button("🚀 Run Full Professional Simulation")

if run: pop = toolbox.population(n=pop_size) hof = tools.HallOfFame(1) algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.25, ngen=gens, halloffame=hof, verbose=False)

best = hof[0]
idx = best[:TOP_K]
ratios = np.array(best[TOP_K:])
ratios /= np.sum(ratios)
res = DB.iloc[idx].copy()
res['Ratio %'] = ratios * 100

strength = saturation(np.sum(ratios * res['strength']) * 10, 150)
duct = saturation(np.sum(ratios * res['ductility']) * 10, 50)
density = np.sum(ratios * res['density'])
cost = np.sum(ratios * res['density'] * res['cost_kg'])
capacity = (strength / (density / 1000)) * (1 + duct / 100)

st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Realistic Strength", f"{strength:.1f} MPa")
m2.metric("Ductility", f"{duct:.1f}")
m3.metric("Cost", f"${cost:.2f} / m³")
m4.metric("Earthquake Capacity", f"{capacity:.1f}")

st.progress(min(1.0, capacity / 200), text="Adjusted Performance Score")

left, right = st.columns(2)
with left:
    st.subheader("Optimized Composition")
    st.dataframe(res[['category', 'name', 'Ratio %']], use_container_width=True)
with right:
    fig = px.bar(res, x='category', y='Ratio %', color='name', title="Category Distribution")
    st.plotly_chart(fig)

END

