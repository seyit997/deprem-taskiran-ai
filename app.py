import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random

# Sayfa Konfigürasyonu
st.set_page_config(page_title="SeismoMutate AI", layout="wide")

st.title("🏗️ SeismoMutate: Evrimsel Malzeme Mühendisliği")
st.markdown("""
Bu platform, **kanser hücrelerinin adaptasyon yeteneğini** inşaat malzemelerine uyarlar. 
Genetik algoritmalar kullanarak, sismik şoklara karşı en dirençli moleküler yapıyı 'evrimleştirir'.
""")

# Sidebar - Parametreler
st.sidebar.header("Genetik Algoritma Ayarları")
pop_size = st.sidebar.slider("Popülasyon Büyüklüğü (Binalar)", 10, 500, 100)
mutation_rate = st.sidebar.slider("Mutasyon Oranı", 0.01, 0.5, 0.1)
generations = st.sidebar.number_input("Nesil Sayısı", 1, 100, 20)

# Simülasyon Fonksiyonu (Basitleştirilmiş Matematiksel Model)
def run_evolution(pop_size, mut_rate, gens):
    # Başlangıç popülasyonu (Esneklik ve Sertlik değerleri 0-1 arası)
    population = np.random.rand(pop_size, 2) 
    history = []

    for g in range(gens):
        # Fitness Fonksiyonu: Esneklik ve Sertlik arasındaki denge (Deprem Dayanımı)
        # Matematiksel Model: Fitness = sin(esneklik) * cos(sertlik) + hata payı
        fitness = np.sin(population[:, 0] * np.pi) * population[:, 1]
        
        best_idx = np.argmax(fitness)
        history.append(fitness[best_idx])
        
        # Seçilim ve Mutasyon
        new_pop = population[np.argsort(fitness)[-pop_size//2:]] # En iyi %50'yi seç
        offspring = new_pop + np.random.normal(0, mut_rate, new_pop.shape) # Mutasyon ekle
        population = np.vstack([new_pop, offspring])
        population = np.clip(population, 0, 1) # Değerleri 0-1 arasında tut

    return population, history

if st.button("Evrimi Başlat"):
    final_pop, fitness_history = run_evolution(pop_size, mutation_rate, generations)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Grafik 1: Dayanıklılık Artışı (Evrim)")
        fig_line = px.line(x=range(generations), y=fitness_history, 
                          labels={'x':'Nesil', 'y':'En Yüksek Dayanıklılık Skoru'},
                          title="Nesiller Boyunca Malzeme Gelişimi")
        st.plotly_chart(fig_line)

    with col2:
        st.subheader("Grafik 2: Malzeme Özellik Dağılımı")
        df = pd.DataFrame(final_pop, columns=['Esneklik', 'Sertlik'])
        fig_scatter = px.scatter(df, x='Esneklik', y='Sertlik', 
                                title="Son Nesil Malzeme Adayları")
        st.plotly_chart(fig_scatter)

    st.success(f"Simülasyon Tamamlandı! En iyi malzeme skoru: {max(fitness_history):.4f}")
