Akıllı Tarım: Hasat Kalitesi Tahmin Modeli
İzmir bölgesine ait iklim ve tarım verilerini kullanarak hasat kalitesini öngören makine öğrenmesi projesi. Geliştirilen modeller, kullanıcı dostu bir web arayüzü üzerinden interaktif tahminler sunmaktadır.

🚀 Kullanılan Teknolojiler
Dil: Python

Modeller: XGBoost, Random Forest

Arayüz: Streamlit

Veri İşleme: Pandas, Scikit-learn

📊 Veri Seti
İzmir bölgesinden API aracılığıyla çekilen 6000 satırlık sıcaklık, nem ve toprak verisi (data/izmir_api_iklim_tarim_verisi_6000.xlsx) kullanılarak modeller eğitilmiş ve test edilmiştir.

⚙️ Kurulum ve Çalıştırma
Depoyu klonlayın: git clone [https://github.com/kullanici-adiniz/akilli-tarim.git](https://github.com/kullanici-adiniz/akilli-tarim.git)

Bağımlılıkları yükleyin: pip install -r requirements.txt

Uygulamayı başlatın: streamlit run app/app.py

📄 Dokümantasyon
Projenin detaylı analiz adımlarına, veri ön işleme süreçlerine ve model performans metriklerine docs/ klasörü altındaki proje raporu ve proje sunumu üzerinden ulaşabilirsiniz.