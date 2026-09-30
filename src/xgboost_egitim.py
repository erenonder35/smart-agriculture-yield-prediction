import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

print("16 Yıllık Veri Seti Yükleniyor...")
df = pd.read_excel('data/izmir_api_iklim_tarim_verisi_6000.xlsx')

X = df.drop(['Tarih', 'Hasat_Kalitesi'], axis=1)
y = df['Hasat_Kalitesi']

# --- XGBOOST'A ÖZEL DÖNÜŞÜM (LABEL ENCODING) ---
# 'A', 'B', 'C' harflerini makinenin anlayacağı 0, 1, 2 rakamlarına çeviriyoruz
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Veriyi bölme ve ölçeklendirme (Öncekiyle tamamen aynı mantık)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("🚀 XGBoost Modeli Eğitiliyor (Hatalardan ders çıkarılıyor)...")
# XGBoost Modeli Kurulumu
xgb_model = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=5, 
    random_state=42,
    eval_metric='mlogloss'
)
xgb_model.fit(X_train_scaled, y_train)

# Tahmin yapma
y_pred = xgb_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*50)
print(f"🔥 XGBOOST DOĞRULUK SKORU (ACCURACY): %{accuracy * 100:.2f}")
print("="*50 + "\n")

# Çıktıları tekrar A, B, C harflerine çevirerek raporluyoruz ki okuması kolay olsun
hedef_isimleri = le.classes_
print("Sınıflandırma Raporu (Precision, Recall, F1-Score):")
print(classification_report(y_test, y_pred, target_names=hedef_isimleri))

# Modeli Web Arayüzü İçin Kaydetme (Gelecek adım için hazırlık)
import joblib
joblib.dump(xgb_model, 'models/xgboost_tarim_modeli.pkl')
joblib.dump(scaler, 'models/tarim_scaler.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
print("\nXGBoost Modeli web sitesi arayüzü (Streamlit) için kaydedildi!")