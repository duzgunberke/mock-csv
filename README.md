# IT Sektörü Maaş Tahmin Sistemi Projesi

## Proje Özeti

Bu proje, IT sektöründe çalışan profesyonellerin maaşlarını çeşitli faktörlere dayanarak tahmin eden kapsamlı bir sistemdir. Python ve makine öğrenmesi teknolojileri kullanılarak geliştirilen bu uygulama, insan kaynakları uzmanlarına ve IT profesyonellerine piyasa standartlarına uygun maaş değerlendirmesi yapma imkanı sunuyor.

Sistem, çalışanın rolü, deneyimi, konumu, eğitim seviyesi ve teknik becerileri gibi faktörleri analiz ederek kişiselleştirilmiş maaş tahmini yapar ve bu tahmini piyasa verileriyle karşılaştırarak detaylı öneriler sunar. Web arayüzü sayesinde kullanıcı dostu bir deneyim sağlarken, arkasındaki güçlü makine öğrenmesi algoritmaları ile yüksek doğrulukta tahminler elde edilir.

## Teknolojiler ve Araçlar

- **Backend**: Python, Pandas, NumPy, scikit-learn
- **Makine Öğrenmesi**: Regresyon modelleri, Ensemble öğrenme teknikleri
- **Frontend**: Streamlit
- **Veri Görselleştirme**: Matplotlib, Seaborn, Plotly
- **Veri Kaynağı**: Türkiye IT sektöründen 10.000+ çalışan verisi
- **Kod Organizasyonu**: Modüler yapı, nesne yönelimli tasarım prensipleri
- **Versiyon Kontrolü**: Git

## Teknik Detaylar ve Kullanılan Yöntemler

### Veri İşleme Teknikleri

- **Önişleme Pipeline'ları**: scikit-learn'ün `Pipeline` ve `ColumnTransformer` modülleri ile standartlaştırılmış veri işleme adımları
- **Kategorik Değişken Kodlama**: One-hot encoding ve Label encoding
- **Ölçeklendirme**: StandardScaler ve MinMaxScaler kullanılarak sayısal değişkenlerin normalizasyonu
- **Aykırı Değer Tespiti**: IQR (Interquartile Range) ve Z-score metotları
- **Eksik Veri İşleme**: Çeşitli imputation stratejileri (ortalama, medyan, mod)
- **Öznitelik Mühendisliği**: Doğrusal ve etkileşimli öznitelikler oluşturma

### Makine Öğrenmesi Algoritmaları

1. **Temel Modeller**:
   - Çoklu Doğrusal Regresyon (MLR)
   - Karar Ağaçları Regresyonu (Decision Trees)
   - Rastgele Orman Regresyonu (Random Forest)
   - Ridge Regresyon (L2 regularizasyon)

2. **Ensemble Teknikleri**:
   - **Voting Regressor**: Birden fazla modelin tahminlerinin ağırlıklı/ağırlıksız ortalaması
   - **Stacking Regressor**: Meta-model kullanarak alt modellerin tahminlerinin birleştirilmesi
   - **Bagging**: Bootstrap Aggregating ile model varyansının azaltılması
   - **Boosting (AdaBoost)**: Zayıf öğrenicilerin ardışık olarak güçlendirilmesi

3. **Hiperparametre Optimizasyonu**:
   - Grid Search ve Random Search teknikleri
   - Cross-validation ile model geçerliliğinin sağlanması
   - Metrik odaklı model seçimi (MAPE minimizasyonu)

### Yazılım Mimarisi ve Tasarım Paternleri

- **Modüler Yapı**: Her bir fonksiyonel bileşen ayrı modüller halinde tasarlanmıştır:
  - `data_loader.py`: Veri yükleme ve yönetimi
  - `preprocess.py`: Veri önişleme ve dönüştürme
  - `train_model.py`: Model eğitimi ve değerlendirme
  - `ensemble_models.py`: Ensemble model oluşturma ve yönetimi
  - `evaluate_model.py`: Model metriklerinin hesaplanması ve görselleştirilmesi
  - `predict_model.py`: Tahmin işlemleri ve öneri algoritmaları

- **Singleton Pattern**: Veri ve model bileşenlerinin tek bir instance'ını yönetmek için kullanılmıştır
- **Factory Pattern**: Dinamik model oluşturma ve parametrelendirme
- **Strategy Pattern**: Farklı değerlendirme stratejilerinin uygulanması
- **Observer Pattern**: Model eğitim sürecinin izlenmesi ve loglanması

### Performans Optimizasyonu

- **Paralel İşleme**: n_jobs parametresi ile çoklu çekirdek kullanımı
- **Veri Kademelendirme**: Büyük veri setleri için kademeli işleme
- **Seçici Öznitelik Kullanımı**: Öznitelik önem derecelerine göre boyut azaltma
- **Hafıza Optimizasyonu**: Veri tipleri ve yapıları için bellek verimliliği

### Web Uygulaması Yapısı

- **Streamlit Framework**: Reaktif ve etkileşimli web arayüzü
- **Componentler**: Modüler UI bileşenleri (form, dashboard, raporlama)
- **Önbellek Mekanizması**: Hesaplama yoğun işlemler için @st.cache dekoratörü
- **Asenkron İşleme**: Uzun süren işlemler için progress bar ve background tasks

### Model Persistance ve Deployment

- **Model Serileştirme**: pickle kullanarak eğitilmiş modellerin kaydedilmesi
- **Versiyon Yönetimi**: Eğitim tarihi ve performans metrikleriyle model versiyonlama
- **Otomatik Setup**: Projenin tek komutla kurulumunu sağlayan setup.py
- **Dockerfile**: Container tabanlı deployment için yapılandırma

## Teknik Zorluklar ve Çözümler

1. **Veri Kalitesi**: Veri kalitesini artırmak için kapsamlı önişleme pipeline'ları geliştirildi
2. **Model Doğruluğu**: Ensemble teknikleri kullanılarak tek modellerin sınırlamalarının üstesinden gelindi
3. **Genellenebilirlik**: Farklı rol ve deneyim segmentleri için özelleştirilmiş alt modeller 
4. **Hataya Dayanıklılık**: Fallback mekanizmaları ile sistem sağlamlığı artırıldı
5. **Performans-Doğruluk Dengesi**: Optimum model kompleksitesi için düzenli deneyler yapıldı
