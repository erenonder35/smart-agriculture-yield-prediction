import pandas as pd
import numpy as np
import requests

print("Open-Meteo API'sine bağlanılıyor ve 16 YILLIK İzmir verileri çekiliyor... Lütfen bekleyin.")

# 1. GERÇEK VERİ ÇEKİMİ: Open-Meteo API (İzmir Koordinatları)
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 38.4237,
    "longitude": 27.1428,
    "start_date": "2006-01-01", # Tarihi geriye çektik
    "end_date": "2023-03-20", 
    "daily": ["temperature_2m_mean", "precipitation_sum", "sunshine_duration"],
    "timezone": "auto"
}

response = requests.get(url, params=params)
data = response.json()

hava_verisi = pd.DataFrame({
    'Tarih': data['daily']['time'],
    'Hava_Sicakligi_C': data['daily']['temperature_2m_mean'],
    'Yagis_Miktari_mm': data['daily']['precipitation_sum'],
    'Gunluk_Guneslenme_Saat': data['daily']['sunshine_duration']
})

hava_verisi['Gunluk_Guneslenme_Saat'] = hava_verisi['Gunluk_Guneslenme_Saat'] / 3600

# Tam 6000 satır almak için veriyi kesiyoruz
hava_verisi = hava_verisi.head(6000)
n_samples = len(hava_verisi)

# --- VERİ ÖN İŞLEME ---
hava_verisi['Hava_Sicakligi_C'] = hava_verisi['Hava_Sicakligi_C'].fillna(hava_verisi['Hava_Sicakligi_C'].mean())
hava_verisi['Yagis_Miktari_mm'] = hava_verisi['Yagis_Miktari_mm'].fillna(0)
hava_verisi['Gunluk_Guneslenme_Saat'] = hava_verisi['Gunluk_Guneslenme_Saat'].fillna(hava_verisi['Gunluk_Guneslenme_Saat'].mean())

# 2. SENTETİK TOPRAK SENSÖRÜ VERİLERİ 
np.random.seed(42)
azot_n = np.random.randint(20, 140, n_samples)
fosfor_p = np.random.randint(10, 100, n_samples)
potasyum_k = np.random.randint(15, 120, n_samples)
toprak_nemi = np.random.randint(20, 90, n_samples)
ph_seviyesi = np.round(np.random.uniform(4.5, 9.0, n_samples), 1)

# 3. VERİLERİ BİRLEŞTİRME 
df = pd.DataFrame({
    'Tarih': hava_verisi['Tarih'],
    'Azot_N_(mg/kg)': azot_n,
    'Fosfor_P_(mg/kg)': fosfor_p,
    'Potasyum_K_(mg/kg)': potasyum_k,
    'Toprak_Nemi_Yuzde': toprak_nemi,
    'pH_Seviyesi': ph_seviyesi,
    'Hava_Sicakligi_C': np.round(hava_verisi['Hava_Sicakligi_C'], 1), 
    'Gunluk_Guneslenme_Saat': np.round(hava_verisi['Gunluk_Guneslenme_Saat'], 1), 
    'Yagis_Miktari_mm': np.round(hava_verisi['Yagis_Miktari_mm'], 1) 
})

# 4. HEDEF DEĞİŞKEN BELİRLEME
def determine_quality(row):
    score = (row['Azot_N_(mg/kg)'] + row['Fosfor_P_(mg/kg)'] + row['Potasyum_K_(mg/kg)']) / 3
    ideal_ph = 6.0 <= row['pH_Seviyesi'] <= 7.5
    ideal_temp = 20 <= row['Hava_Sicakligi_C'] <= 30
    ideal_moisture = 40 <= row['Toprak_Nemi_Yuzde'] <= 70
    
    final_score = score
    if ideal_ph: final_score += 20
    if ideal_temp: final_score += 15
    if ideal_moisture: final_score += 15
    
    if final_score > 110: return 'A'
    elif final_score > 75: return 'B'
    else: return 'C'

df['Hasat_Kalitesi'] = df.apply(determine_quality, axis=1)

noise_indices = np.random.choice(n_samples, size=int(n_samples * 0.15), replace=False)
classes = ['A', 'B', 'C']
for idx in noise_indices:
    current_val = df.at[idx, 'Hasat_Kalitesi']
    df.at[idx, 'Hasat_Kalitesi'] = np.random.choice([c for c in classes if c != current_val])

# 5. EXCEL'E KAYDETME
file_path = 'data/izmir_api_iklim_tarim_verisi_6000.xlsx'
df.to_excel(file_path, index=False)
print(f"BAŞARILI!  veri setin hazır: {file_path}")