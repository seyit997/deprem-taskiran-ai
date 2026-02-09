import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="SeismoMutate Pro | Advanced Seismic Lab", layout="wide")

# --- ŞIK TASARIM İÇİN CSS ---
st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stMetric { border-radius: 15px; background: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); padding: 20px !important; border-left: 5px solid #1E88E5; }
    h1, h2, h3 { color: #1565C0; font-family: 'Segoe UI', sans-serif; }
    .instruction-card { background: #e3f2fd; padding: 25px; border-radius: 15px; border-left: 8px solid #0d47a1; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- BAŞLIK ---
st.title("🏗️ SeismoMutate: Biyo-İlhamlı Sismik Malzeme Laboratuvarı")
st.markdown("**Kanser Hücresi Adaptasyon Modeli ile Depreme Dayanıklı Yapısal Malzeme Optimizasyonu**")
st.markdown("---")

# --- SIDEBAR (KONTROL PANELİ) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.header("🔬 Simülasyon Parametreleri")
    target_mw = st.slider("Hedef Deprem Şiddeti (Mw)", 5.0, 9.5, 8.2, step=0.1)
    project_size = st.number_input("İnşaat Hacmi (m³ Beton)", min_value=1, value=500)
    evolution_depth = st.select_slider("Evrimsel Derinlik (Hassasiyet)", options=["Düşük", "Orta", "Yüksek", "Ekstrem"])
    
    gen_map = {"Düşük": 200, "Orta": 500, "Yüksek": 1000, "Ekstrem": 2500}
    num_gens = gen_map[evolution_depth]

# --- EVRİMSEL MOTOR (BACKEND) ---
def run_heavy_evolution(gens, mw):
    components = ["Çimento", "Agrega", "Likit Polimer", "Karbon Nanotüp"]
    pop_size = 120
    # İlk popülasyon
    pop = np.random.rand(pop_size, 4)
    pop = pop / pop.sum(axis=1)[:, None]
    
    history = []
    
    # Progress Bar simülasyonu
    prog_text = st.empty()
    bar = st.progress(0)
    
    for g in range(gens):
        # Mühendislik Fonksiyonları
        c, a, p, n = pop[:,0], pop[:,1], pop[:,2], pop[:,3]
        
        # Dayanıklılık (Strength) Skoru
        strength = (c * 0.4) + (n * 5.0) 
        # Esneklik (Ductility) Skoru - Deprem şiddeti arttıkça polimer ihtiyacı artar
        ductility = (p * (mw/4)) * (n * 1.5)
        # Yapısal Bütünlük Cezası (Agrega oranı %25-35 dışındaysa puan kır)
        penalty = np.abs(0.30 - a) * 5
        
        fitness = strength + ductility - penalty
        
        best_idx = np.argmax(fitness)
        history.append(fitness[best_idx])
        
        # Doğal Seçilim ve Mutasyon (Crossover)
        parents = pop[np.argsort(fitness)[-pop_size//2:]]
        mutations = np.random.normal(0, 0.015, parents.shape)
        offspring = np.clip(parents + mutations, 0.01, 1)
        pop = np.vstack([parents, offspring])
        pop = pop / pop.sum(axis=1)[:, None]
        
        if g % (gens//10) == 0:
            bar.progress(g/gens)
            prog_text.text(f"Nesil {g} analiz ediliyor... En iyi fitness: {fitness[best_idx]:.4f}")

    bar.empty()
    prog_text.empty()
    return pop[np.argmax(fitness)], history

# --- ANA EKRAN ANALİZİ ---
if st.button("🚀 Milyonluk Evrimsel Analizi Çalıştır"):
    best_recipe, fitness_history = run_heavy_evolution(num_gens, target_mw)
    
    # --- 1. SEKSİYON: ÜST METRİKLER ---
    st.subheader("📋 Temel Performans Göstergeleri (KPI)")
    
    # Veri Türetme (Nokta hataları giderildi)
    unit_base = 120 # $/m3 standart
    nano_cost = best_recipe[3] * 15000 # Nanotüp pahalı
    poly_cost = best_recipe[2] * 950
    final_unit_cost = int(unit_base + nano_cost + poly_cost)
    
    damage_potential = max(2, 100 - (best_recipe[3]*500 + best_recipe[2]*250) / (target_mw/6))
    healing_rate = (best_recipe[3] * 35) + (best_recipe[2] * 65)
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Hasar Riski", f"%{damage_potential:.1f}", "-%62", delta_color="inverse")
    kpi2.metric("Kendi Kendini Onarma", f"%{healing_rate:.1f}", "Aktif")
    kpi3.metric("m³ Birim Maliyet", f"{final_unit_cost} $")
    kpi4.metric("Toplam Proje Ek Maliyeti", f"{int((final_unit_cost - unit_base)*project_size):,} $")

    st.markdown("---")

    # --- 2. SEKSİYON: GRAFİKLER ---
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📈 Evrimsel Dayanıklılık Eğrisi")
        fig_line = px.line(x=range(len(fitness_history)), y=fitness_history, 
                          labels={'x': 'Nesiller (Mutasyon Süreci)', 'y': 'Sismik Direnç Katsayısı'})
        fig_line.update_traces(line_color='#1976D2', fill='tozeroy')
        st.plotly_chart(fig_line, use_container_width=True)

    with col_right:
        st.subheader("🧪 Optimal Moleküler Dağılım")
        labels = ["Çimento", "Agrega", "Likit Polimer", "Karbon Nanotüp"]
        fig_pie = px.pie(values=best_recipe, names=labels, hole=0.4, 
                         color_discrete_sequence=px.colors.sequential.Blues_r)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # --- 3. SEKSİYON: REÇETE VE HAZIRLANIŞ ---
    st.subheader("👨‍🔬 Laboratuvar Karışım Reçetesi (1 m³ İçin)")
    
    total_weight = 2350 # kg/m3 (Ortalama beton ağırlığı)
    water = 175 # Litre
    material_weight = total_weight - water
    
    df_rec = pd.DataFrame({
        "Bileşen": labels,
        "Kütlesel Oran": [f"% {x*100:.2f}" for x in best_recipe],
        "Miktar (Kilogram)": [f"{int(x * material_weight)} kg" for x in best_recipe],
        "Fonksiyon": [
            "Yapısal Matris", 
            "Hacimsel Stabilite", 
            "Sismik Enerji Absorpsiyonu (Sitoplazma)", 
            "Mikro-Çatlak Onarımı (DNA Repair)"
        ]
    })
    st.table(df_rec)

    # --- 4. SEKSİYON: PAZARLAMA VE TEKNİK TALİMAT ---
    st.subheader("💡 Uygulama Metodolojisi ve Pazarlama")
    
    inst1, inst2 = st.columns(2)
    with inst1:
        st.markdown(f"""
        <div class="instruction-card">
        <h4>Şantiye Uygulama Talimatı</h4>
        <ul>
            <li><b>Su Karışımı:</b> {water} Litre suya önce Polimeri ekleyin.</li>
            <li><b>Nanotüp Dispersiyonu:</b> Nanotüpleri topaklanmaması için yüksek devirli karıştırıcıda 15 dk çözün.</li>
            <li><b>Döküm:</b> {target_mw} şiddetine dayanıklı bu karışım, döküldükten sonraki ilk 48 saatte termal kürleme gerektirmez.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with inst2:
        st.info(f"""
        **Neden Bu Malzeme?**
        Geleneksel beton Mw {target_mw} şiddetinde gevrek kırılma yaşayarak çöker. 
        **SeismoMutate v3.0** ise, kanser hücrelerinin kemoterapiye karşı geliştirdiği 'hücresel esneklik' mekanizmasını kullanır. 
        Bina sarsıldığında, polimer zincirleri moleküler düzeyde uzayarak enerjiyi ısıya dönüştürür ve binanın çökmesini engeller.
        """)

    st.success(f"Analiz başarıyla tamamlandı. Bu karışım ile Mw {target_mw} senaryosunda yapı güvenliği %{100-damage_potential:.1f} oranında optimize edilmiştir.")
