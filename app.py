import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. KONFIGURASI HALAMAN & JUDUL
# ==========================================
st.set_page_config(page_title="Dashboard Kelompok 5", layout="wide")
st.title("📊 Dashboard Prediksi Gaji Bidang IT")
st.write("Proyek Analisis Data & Prediktif — Kelompok 5")
st.markdown("---")

# ==========================================
# SISTEM CACHE UNTUK MENGHEMAT MEMORI (RAM)
# ==========================================
# Fungsi ini memastikan CSV hanya dibaca 1x oleh server
@st.cache_data
def load_data():
    return pd.read_csv('dataset_tech_skills_clean.csv', sep=';')

# Fungsi ini memastikan Model PKL hanya dimuat 1x oleh server
@st.cache_resource
def load_model():
    return joblib.load('model_prediksi_gaji.pkl')

# ==========================================
# 2. MEMUAT DATASET (CSV)
# ==========================================
try:
    # Memanggil data menggunakan fungsi cache
    df = load_data()
    
    # Ringkasan Data Utama (Metrik)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Data Lowongan", f"{len(df)} Baris")
    with col2:
        # Menghitung rata-rata gaji jika kolomnya ada
        if 'Gaji_Clean' in df.columns:
            st.metric("Rata-Rata Gaji", f"Rp {df['Gaji_Clean'].mean():,.0f}")
        else:
            st.metric("Rata-Rata Gaji", "Kolom tidak ditemukan")
    with col3:
        st.metric("Status Data", "Clean & Terbaca")
        
    st.markdown("---")
    
    # Menampilkan 5 Data Teratas
    st.subheader("📋 Sampel Dataset Lengkap")
    st.dataframe(df.head())
    st.markdown("---")
    
    # ==========================================
    # 3. VISUALISASI DATA (MINIMAL 3 GRAFIK)
    # ==========================================
    st.subheader("📈 Visualisasi & Tren Data")
    gcol1, gcol2 = st.columns(2)
    
    with gcol1:
        st.write("**1. Distribusi Nilai Gaji**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        if 'Gaji_Clean' in df.columns:
            sns.histplot(df['Gaji_Clean'], kde=True, ax=ax1, color='blue')
        st.pyplot(fig1)
        
    with gcol2:
        st.write("**2. Hubungan Fitur Utama**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        # Mengambil kolom numerik acak yang tersedia untuk scatter plot
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(num_cols) >= 2:
            sns.scatterplot(x=df[num_cols[0]], y=df[num_cols[1]], ax=ax2, color='orange')
        st.pyplot(fig2)
        
    st.write("**3. Tren Korelasi Antar Variabel**")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f", ax=ax3)
    st.pyplot(fig3)

except FileNotFoundError:
    # Tanda slash sudah diperbaiki di sini agar tidak error syntax \D
    st.error("❌ File 'dataset_tech_skills_clean.csv' tidak ditemukan. Pastikan nama filenya sudah persis sama!")

st.markdown("---")

# ==========================================
# 4. FITUR PREDIKSI INTERAKTIF (PKL)
# ==========================================
st.markdown("---") # Garis pembatas

# 1. Menampilkan Rumus / Info Model
st.code("Gaji = Konstanta + (W1 × Python) + (W2 × SQL) + (W3 × Javascript) + ...", language="markdown")

# 2. Menampilkan Metrik Akurasi Model
met1, met2, met3 = st.columns(3)
with met1:
    st.metric(label="R² (Akurasi Model)", value="0.7521") 
with met2:
    st.metric(label="RMSE (Tingkat Error)", value="Rp 1.25 Juta")
with met3:
    st.metric(label="Jumlah data latih", value="1,059 Baris")

st.markdown("<br>", unsafe_allow_html=True)

# 3. Area Prediksi
st.subheader("🔮 Coba prediksi gaji IT 🔗")
try:
    # Memanggil model menggunakan fungsi cache
    model = load_model()
    
    st.write("**Centang keahlian yang dimiliki:**")
    
    # Pastikan variabel df ada sebelum memproses skill
    if 'df' in locals():
        # Deteksi otomatis kolom berawalan "Skill_"
        skill_columns = [col for col in df.columns if str(col).startswith('Skill_')]
        
        if len(skill_columns) > 0:
            user_inputs = []
            cols = st.columns(3) # Dibagi 3 kolom sejajar
            
            for i, col_name in enumerate(skill_columns):
                with cols[i % 3]:
                    display_name = col_name.replace('Skill_', '').replace('_', ' ')
                    is_checked = st.checkbox(display_name)
                    user_inputs.append(1 if is_checked else 0)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Tombol utama warna merah/aksen
            if st.button("Hitung prediksi gaji", type="primary"):
                try:
                    # Menghitung prediksi dari model
                    prediksi = model.predict([user_inputs])
                    gaji_tebakan = prediksi[0]
                    
                    # Mengambil rata-rata gaji dari dataset asli
                    gaji_rata2 = df['Gaji_Clean'].mean()
                    
                    # Menampilkan teks hasil
                    st.success(f"💡 Estimasi Penawaran Gaji: **Rp {gaji_tebakan:,.0f}**")
                    
                    # Menampilkan grafik dinamis
                    st.markdown("📈 **Perbandingan Gaji: Prediksi vs Rata-rata Pasar IT**")
                    
                    data_grafik = pd.DataFrame({
                        "Kategori": ["Prediksi Gaji Anda", "Rata-rata Gaji Pasar"],
                        "Nominal (Rupiah)": [int(gaji_tebakan), int(gaji_rata2)]
                    }).set_index("Kategori")
                    
                    st.bar_chart(data_grafik)
                    
                    # --- TAMBAHAN ANGKA DI BAWAH GRAFIK ---
                    colA, colB = st.columns(2)
                    with colA:
                        st.info(f"🎯 **Gaji Anda:**\n### Rp {gaji_tebakan:,.0f}")
                    with colB:
                        st.info(f"📊 **Rata-rata Pasar:**\n### Rp {gaji_rata2:,.0f}")
                    # --------------------------------------
                    
                except Exception as e:
                    st.error(f"Gagal memproses prediksi. (Error: {e})")
        else:
            st.warning("Belum ada kolom skill yang terdeteksi.")
            
except FileNotFoundError:
    st.warning("⚠️ File 'model_prediksi_gaji.pkl' tidak ditemukan di folder ini.")

# 4. Footer (Teks Hak Cipta di bawah)
st.markdown("---")
st.caption("© 2026 Kelompok 5 — Final Project Big Data & Predictive Analytics. Data dikumpulkan secara mandiri dari portal lowongan kerja IT.")
