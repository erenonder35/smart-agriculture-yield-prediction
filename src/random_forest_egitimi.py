import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. Veriyi Yükleme
print("16 Yıllık Veri Seti Yükleniyor...")
df = pd.read_excel('data/izmir_api_iklim_tarim_verisi_6000.xlsx')

# 2. Bağımsız Değişkenler (X) ve Hedef Değişken (y) Ayırımı
X = df.drop(['Tarih', 'Hasat_Kalitesi'], axis=1)
y = df['Hasat_Kalitesi']

# 3. Eğitim ve Test Setlerine Bölme (%80 Eğitim, %20 Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 4. Veri Ölçeklendirme (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model Kurulumu ve Eğitimi (Random Forest)
print("🚀 Yapay Zeka Modeli Eğitiliyor (Random Forest)...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 6. Tahmin ve Başarı Ölçümü
y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*50)
print(f"🌲 RANDOM FOREST DOĞRULUK SKORU (ACCURACY): %{accuracy * 100:.2f}")
print("="*50 + "\n")

print("Sınıflandırma Raporu (Precision, Recall, F1-Score):")
print(classification_report(y_test, y_pred))

# 7. Özellik Önemi (Feature Importance) Görselleştirme
feature_importances = rf_model.feature_importances_
features_df = pd.DataFrame({
    'Özellik (Sensör)': X.columns,
    'Önem Derecesi': feature_importances
}).sort_values(by='Önem Derecesi', ascending=False)

plt.figure(figsize=(10, 6))
# FutureWarning almamak için hue parametresi de eklendi
sns.barplot(x='Önem Derecesi', y='Özellik (Sensör)', data=features_df, hue='Özellik (Sensör)', palette='viridis', legend=False)
plt.title('Hangi Sensör Verisi Hasat Kalitesini Daha Çok Etkiliyor?', fontsize=14, fontweight='bold')
plt.xlabel('Etki Ağırlığı')
plt.ylabel('Sensörler')
plt.tight_layout()

# 8. İleride kullanmak istersen diye modeli kaydetme
joblib.dump(rf_model, 'models/random_forest_tarim_modeli.pkl')
print("\nModel başarıyla kaydedildi! Özellik önemi grafiği ekrana yansıtılıyor...")

plt.show()