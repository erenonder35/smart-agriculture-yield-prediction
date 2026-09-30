import streamlit as st
import pandas as pd
import joblib

# 1. Sayfa Ayarları (Tarayıcı Sekmesi ve İkon)
st.set_page_config(page_title="Akıllı Tarım Yapay Zekası", page_icon="🌱", layout="centered")

# Başlık ve Açıklama
st.title("🌱 Akıllı Tarım: Hasat Kalitesi Tahminleyici")
st.markdown("""
Bu sistem, 16 yıllık İzmir iklim verileri ve sensör ölçümleriyle eğitilmiş **XGBoost Yapay Zeka Modeli** kullanmaktadır. 
Lütfen aşağıdaki sensör değerlerini değiştirerek tahmini hasat kalitesini görün.
""")

# 2. Kaydedilen Yapay Zeka Modellerini Yükleme (Hızlandırmak için önbelleğe alıyoruz)
@st.cache_resource
def load_models():
    model = joblib.load('models/xgboost_tarim_modeli.pkl')
    scaler = joblib.load('models/tarim_scaler.pkl')
    le = joblib.load('models/label_encoder.pkl')
    return model, scaler, le

try:
    model, scaler, le = load_models()
except FileNotFoundError:
    st.error("Model dosyaları bulunamadı! Lütfen önce xgboost eğitim kodunu çalıştırdığınızdan emin olun.")
    st.stop()

# 3. Kullanıcıdan Veri Alma Arayüzü (Slider'lar)
st.header("📊 Sensör ve İklim Verileri")

# Ekranı iki sütuna bölüyoruz ki şık dursun
col1, col2 = st.columns(2)

with col1:
    st.subheader("Toprak Değerleri")
    azot = st.slider("Azot (N) mg/kg", 20, 140, 80)
    fosfor = st.slider("Fosfor (P) mg/kg", 10, 100, 55)
    potasyum = st.slider("Potasyum (K) mg/kg", 15, 120, 65)
    ph = st.slider("pH Seviyesi", 4.5, 9.0, 6.5, step=0.1)
    nem = st.slider("Toprak Nemi (%)", 20, 90, 50)

with col2:
    st.subheader("İklim Değerleri (İzmir)")
    sicaklik = st.slider("Hava Sıcaklığı (°C)", -5.0, 45.0, 25.0, step=0.1)
    gunes = st.slider("Günlük Güneşlenme (Saat)", 0.0, 15.0, 8.0, step=0.1)
    yagis = st.slider("Yağış Miktarı (mm)", 0, 150, 20)

# 4. Tahmin Butonu ve Sonuç Ekranı
st.markdown("<br>", unsafe_allow_html=True) # Boşluk ekler
if st.button("🚀 Hasat Kalitesini Tahmin Et", use_container_width=True):
    
    # Girdileri modele uygun Pandas Tablosu formatına çevirme
    input_data = pd.DataFrame({
        'Azot_N_(mg/kg)': [azot],
        'Fosfor_P_(mg/kg)': [fosfor],
        'Potasyum_K_(mg/kg)': [potasyum],
        'Toprak_Nemi_Yuzde': [nem],
        'pH_Seviyesi': [ph],
        'Hava_Sicakligi_C': [sicaklik],
        'Gunluk_Guneslenme_Saat': [gunes],
        'Yagis_Miktari_mm': [yagis]
    })

    # Veriyi ölçeklendirme (Eğitimdeki formata getirme)
    scaled_data = scaler.transform(input_data)
    
    # Tahmin yapma (XGBoost 0,1,2 döndürür)
    prediction_encoded = model.predict(scaled_data)
    
    # Çıkan 0,1,2 rakamını tekrar A, B, C harfine çevirme
    prediction_label = le.inverse_transform(prediction_encoded)[0]

    # Sonucu Ekrana Yazdırma
    st.markdown("---")
    st.subheader("🤖 Yapay Zeka Analiz Sonucu")
    
    if prediction_label == 'A':
        st.success(f"🌟 **Mükemmel! Tahmini Hasat Kalitesi: {prediction_label} Sınıfı (Premium)**")
        st.balloons() # Ekranda balonlar uçar!
    elif prediction_label == 'B':
        st.info(f"👍 **İyi. Tahmini Hasat Kalitesi: {prediction_label} Sınıfı (Standart)**")
    else:
        st.warning(f"⚠️ **Dikkat! Tahmini Hasat Kalitesi: {prediction_label} Sınıfı (Düşük)**. Toprak veya iklim şartlarını iyileştirmeniz gerekebilir.")