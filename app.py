import streamlit as st
import h2o
from h2o.automl import H2OAutoML
from h2o.frame import H2OFrame

# Inisialisasi H2O
h2o.init()

# Muat model terbaik dari AutoML
best_model = h2o.load_model("best_h2o_model.zip")  # Pastikan nama file sesuai

# Judul Aplikasi
st.title("Prediksi Harga Rumah dengan H2O AutoML")

# Input data dari pengguna
CRIM = st.number_input("Tingkat Kriminalitas (CRIM)", value=0.0)
ZN = st.number_input("Persentase Lahan Pemukiman (ZN)", value=0.0)
INDUS = st.number_input("Persentase Lahan Komersial Non-Retail (INDUS)", value=0.0)
CHAS = st.selectbox("Berada di Dekat Sungai? (CHAS)", [0, 1])
NOX = st.number_input("Konsentrasi Nitrat Oksida (NOX)", value=0.0)
RM = st.number_input("Rata-rata Jumlah Kamar (RM)", value=0.0)
AGE = st.number_input("Persentase Pemilik Rumah Berusia Tua (AGE)", value=0.0)
DIS = st.number_input("Jarak ke Pusat Pekerjaan (DIS)", value=0.0)
RAD = st.number_input("Aksesibilitas Jalan Raya (RAD)", value=0.0)
TAX = st.number_input("Tingkat Pajak Properti (TAX)", value=0.0)
PTRATIO = st.number_input("Rasio Murid terhadap Guru (PTRATIO)", value=0.0)
B = st.number_input("Proporsi Populasi Kulit Hitam (B)", value=0.0)
LSTAT = st.number_input("Persentase Populasi Berstatus Sosial Rendah (LSTAT)", value=0.0)

# Tombol untuk Prediksi
if st.button("Prediksi"):
    # Konversi input menjadi H2OFrame
    data = H2OFrame([{
        "CRIM": CRIM,
        "ZN": ZN,
        "INDUS": INDUS,
        "CHAS": CHAS,
        "NOX": NOX,
        "RM": RM,
        "AGE": AGE,
        "DIS": DIS,
        "RAD": RAD,
        "TAX": TAX,
        "PTRATIO": PTRATIO,
        "B": B,
        "LSTAT": LSTAT
    }])
    
    # Prediksi dengan model H2O
    prediction = best_model.predict(data)
    st.write(f"Hasil Prediksi Harga Rumah: ${prediction.as_data_frame().iloc[0, 0]:,.2f}")
