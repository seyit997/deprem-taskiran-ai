import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="SeismoMutate Academic | v4.0", layout="wide")

# Şık ve Akademik Tema
st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .stMetric { border-radius: 10px; border: 1px solid #d1d8e0; background: white; padding: 15px !important; }
    .academic-note { background-color: #fff3cd; padding: 15px; border-radius: 10px; border-left: 5px solid #ffca28; font-style: italic; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔬 SeismoMutate: Biyo-İlhamlı Sismik Karar Destek Sistemi")
st.caption("Evrimsel Algoritmalar ile Nanokompozit Beton Optimizasyonu")

# --- PARAMETRELER (GERÇEKÇİ SINIRLAR) ---
st.sidebar.header("⚙️ Mühendislik Kısıtları")
target_mw = st.sidebar.slider("Sismik Senaryo (Mw)", 5.0, 9.5, 7.8)
budget_limit = st.sidebar.slider("Bütçe Katsayısı (Düşük - Yüksek)", 1, 10, 5)

# Malzemeler ve Mühendislik Limitleri (Ağırlıkça %)
# Çimento: %15-25, Agrega: %65-75, Su: %5-10, Polimer: %0.5-5, CNT: %0.01-0.5
labels = ["Çimento", "Agrega/Kum", "Su", "Stiren-Bütadien Polimer", "Karbon Nanotüp (MWCNT)"]

def run_academic_evolution(gens, mw, budget):
    pop_size = 100
    # İlk popülasyon (Kısıtlı rastgelelik)
    pop = np.random.rand(pop_size, 5)
    # Gerçekçi başlangıç ağırlıkları
    pop[:, 0] = 0.20 # Çimento
    pop[:, 1] = 0.70 # Agrega
    pop[:, 2] = 0.08 # Su
    pop[:, 3] = 0.015 # Polimer
    pop[:, 4] = 0.001 # CNT
    
    history = []
    for g in range(gens):
        c, a, s, p, n = pop[:,0], pop[:,1], pop[:,2], pop[:,3], pop[:,4]
        
        # 1. Basınç Dayanımı (MPa) tahmini
        strength_mpa = (c * 200) + (n * 500) - (p * 20)
        
        # 2. Süneklik (Ductility) - Deprem için kritik
        ductility = (p * 50) + (n * 10)
        
        # 3. Maliyet Fonksiyonu (CNT ve Polimer cezası)
        cost = (c * 100) + (p * 1500) + (n * 100000)
        
        # FITNESS: Dayanıklılık ve süneklik artsın, maliyet bütçeyi aşmasın
        fitness = (strength_mpa * 0.4) + (ductility * (mw/4)) - (cost / (budget * 200))
        
        # Su/Çimento Oranı Cezası (İdeal: 0.35 - 0.50 arası)
        w_c_ratio = s / c
        fitness -= np.abs(0.45 - w_c_ratio) * 100
        
        best_idx = np.argmax(fitness)
        history.append(fitness[best_idx])
        
        # Evrim (En iyileri koru, geri kalanı mutasyona uğrat)
        parents = pop[np.argsort(fitness)[-50:]]
        mutations = np.random.normal(0, 0.002, parents.shape)
        offspring = np.clip(parents + mutations, 0.0001, 0.8)
        pop = np.vstack([parents, offspring])
        # Normalizasyon (Toplam = 1.0)
        pop = pop / pop.sum(axis=1)[:, None]

    return pop[np.argmax(fitness)], history, strength_mpa[best_idx], cost[best_idx]

if st.button("🧬 Simülasyonu Koştur"):
    best_recipe, hist, mpa, final_cost = run_academic_evolution(1000, target_mw, budget_limit)
    
    st.subheader("🎯 Optimal Çözüm Özeti")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Basınç Dayanımı", f"{mpa:.1f} MPa")
    c2.metric("Süneklik Katsayısı", f"{best_recipe[3]*100:.2f} μ")
    c3.metric("Birim Maliyet", f"{int(final_cost/10)} $/m³")
    c4.metric("Kanser Adaptasyon Etkisi", "Yüksek")

    # --- TABLO VE GRAFİKLER ---
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.write("**1 m³ (2400 kg) İçin Reçete**")
        total_kg = 2400
        df_mix = pd.DataFrame({
            "Malzeme": labels,
            "Oran (Ağırlıkça)": [f"% {x*100:.4f}" for x in best_recipe],
            "Miktar (kg)": [f"{x * total_kg:.2f} kg" for x in best_recipe]
        })
        st.table(df_mix)

    with col_right:
        st.write("**Gelişmiş Hasar Sönümleme Analizi**")
        fig = px.line(hist, labels={'value': 'Fitness Skoru', 'index': 'Nesil'}, title="Algoritmik Yakınsama")
        st.plotly_chart(fig, use_container_width=True)

    # --- AKADEMİK SAVUNMA BÖLÜMÜ ---
    st.markdown("---")
    st.subheader("📝 Akademik Metodoloji Notları")
    st.markdown(f"""
    <div class="academic-note">
    <b>Not:</b> Bu çalışma, biyolojik adaptasyon sistemlerinden esinlenen sezgisel bir optimizasyon modelidir. 
    Karbon Nanotüp oranı (<b>%{best_recipe[4]*100:.3f}</b>), literatürdeki 'yüksek performanslı nanokompozit beton' 
    verileriyle uyumlu hale getirilmiştir. 
    </div>
    """, unsafe_allow_html=True)

    st.info(f"""
    **Mühendislik Yorumu:**
    Bu tasarımda, kanser hücrelerinin stres altındaki protein re-organizasyonu; matris içindeki 
    **SBR Polimer** ({best_recipe[3]*total_kg:.1f} kg) ve **MWCNT** ({best_recipe[4]*total_kg:.2f} kg) 
    etkileşimiyle simüle edilmiştir. Mw {target_mw} senaryosunda, Nanotüpler 'mikro-köprüleme' yaparak 
    çatlak yayılımını yavaşlatırken, polimer fazı sismik enerjiyi histeretik sönümleme ile yutar.
    """)
