import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import re
import logging
from datetime import datetime

# Loglama yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(file_path):
    """
    Veri setini yükler.
    
    Args:
        file_path (str): CSV dosyasının yolu
        
    Returns:
        pandas.DataFrame: Yüklenen veri seti
    """
    try:
        logger.info(f"Veri yükleniyor: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Veri başarıyla yüklendi. Boyut: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Veri yüklenirken hata oluştu: {str(e)}")
        raise


def basic_eda(df):
    """
    Temel keşifsel veri analizi gerçekleştirir ve sonuçları döndürür.
    
    Args:
        df (pandas.DataFrame): Analiz edilecek veri seti
        
    Returns:
        dict: Veri seti hakkında temel bilgiler içeren sözlük
    """
    logger.info("Temel keşifsel veri analizi gerçekleştiriliyor")
    
    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_stats": df.describe().to_dict()
    }
    
    # Kategorik değişkenlerin benzersiz değerlerini sayalım
    categorical_columns = df.select_dtypes(include=['object']).columns
    unique_values = {}
    for col in categorical_columns:
        unique_values[col] = df[col].nunique()
    
    info["categorical_unique_counts"] = unique_values
    
    logger.info("Temel EDA tamamlandı")
    return info


def identify_column_types(df):
    """
    Veri setindeki sütunları tipine göre sınıflandırır.
    
    Args:
        df (pandas.DataFrame): Sınıflandırılacak veri seti
        
    Returns:
        dict: Sütun tiplerinin listesini içeren sözlük
    """
    logger.info("Sütun tipleri belirleniyor")
    
    # Veri tipleri
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Tarih sütunları (string olarak saklanabilir)
    date_pattern = re.compile(r'tarih|date|zaman|time', re.IGNORECASE)
    potential_date_columns = [col for col in df.columns if date_pattern.search(col)]
    
    # Hedef değişkeni (Maaş_TL) hariç tutmak
    if 'Maaş_TL' in numeric_features:
        numeric_features.remove('Maaş_TL')
    
    column_types = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "potential_date_columns": potential_date_columns,
        "target": "Maaş_TL"  # Varsayılan hedef değişken
    }
    
    logger.info(f"Sütun tipleri belirlendi: {len(numeric_features)} sayısal, {len(categorical_features)} kategorik")
    return column_types


def convert_date_columns(df, date_columns=None):
    """
    Tarih sütunlarını datetime formatına dönüştürür.
    
    Args:
        df (pandas.DataFrame): İşlenecek veri seti
        date_columns (list, optional): Dönüştürülecek tarih sütunları. None ise, potansiyel tarih sütunları otomatik belirlenir.
        
    Returns:
        pandas.DataFrame: Tarih sütunları dönüştürülmüş veri seti
    """
    logger.info("Tarih sütunları dönüştürülüyor")
    
    df_processed = df.copy()
    
    if date_columns is None:
        # Potansiyel tarih sütunlarını bul
        date_pattern = re.compile(r'tarih|date|zaman|time', re.IGNORECASE)
        date_columns = [col for col in df.columns if date_pattern.search(col)]
    
    for col in date_columns:
        if col in df.columns:
            try:
                df_processed[col] = pd.to_datetime(df[col], errors='coerce')
                logger.info(f"{col} sütunu başarıyla datetime formatına dönüştürüldü")
            except Exception as e:
                logger.warning(f"{col} sütunu dönüştürülemedi: {str(e)}")
    
    return df_processed


def handle_missing_values(df, numeric_strategy='median', categorical_strategy='most_frequent'):
    """
    Eksik değerleri işler.
    
    Args:
        df (pandas.DataFrame): İşlenecek veri seti
        numeric_strategy (str): Sayısal değişkenler için doldurma stratejisi ('mean', 'median', 'most_frequent')
        categorical_strategy (str): Kategorik değişkenler için doldurma stratejisi ('most_frequent', 'constant')
        
    Returns:
        pandas.DataFrame: Eksik değerleri işlenmiş veri seti
    """
    logger.info(f"Eksik değerler işleniyor. Sayısal strateji: {numeric_strategy}, Kategorik strateji: {categorical_strategy}")
    
    # Eksik değer sayısını kontrol et
    missing_values = df.isnull().sum().sum()
    if missing_values == 0:
        logger.info("Eksik değer yok, işlem yapılmadı")
        return df
    
    # Kopyasını oluştur
    df_processed = df.copy()
    
    # Sayısal değişkenler için
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_features:
        if df[col].isnull().sum() > 0:
            if numeric_strategy == 'mean':
                df_processed[col] = df[col].fillna(df[col].mean())
            elif numeric_strategy == 'median':
                df_processed[col] = df[col].fillna(df[col].median())
            elif numeric_strategy == 'most_frequent':
                df_processed[col] = df[col].fillna(df[col].mode()[0])
            logger.info(f"{col} sütunundaki {df[col].isnull().sum()} eksik değer dolduruldu")
    
    # Kategorik değişkenler için
    categorical_features = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_features:
        if df[col].isnull().sum() > 0:
            if categorical_strategy == 'most_frequent':
                df_processed[col] = df[col].fillna(df[col].mode()[0])
            elif categorical_strategy == 'constant':
                df_processed[col] = df[col].fillna('Bilinmiyor')
            logger.info(f"{col} sütunundaki {df[col].isnull().sum()} eksik değer dolduruldu")
    
    missing_after = df_processed.isnull().sum().sum()
    logger.info(f"Toplam {missing_values - missing_after} eksik değer işlendi. Kalan eksik değer sayısı: {missing_after}")
    
    return df_processed


def handle_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Aykırı değerleri işler.
    
    Args:
        df (pandas.DataFrame): İşlenecek veri seti
        columns (list): İşlenecek sütunlar (None ise tüm sayısal sütunlar)
        method (str): Kullanılacak yöntem ('iqr' veya 'zscore')
        threshold (float): Aykırı değer için eşik değeri
        
    Returns:
        pandas.DataFrame: Aykırı değerleri işlenmiş veri seti
    """
    logger.info(f"Aykırı değerler işleniyor. Yöntem: {method}")
    
    # Kopyasını oluştur
    df_processed = df.copy()
    
    # İşlenecek sütunları belirle
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Hedef değişkeni (Maaş_TL) dışarıda bırakalım, onu da işlemek istersek ayrıca belirtmek lazım
    if 'Maaş_TL' in columns and len(columns) > 1:
        columns = columns.drop('Maaş_TL')
    
    outliers_removed = 0
    
    for col in columns:
        if method == 'iqr':
            # IQR yöntemi
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Aykırı değerleri işle (caps değerleri)
            outliers = ((df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)).sum()
            df_processed.loc[df_processed[col] < lower_bound, col] = lower_bound
            df_processed.loc[df_processed[col] > upper_bound, col] = upper_bound
            
        elif method == 'zscore':
            # Z-score yöntemi
            z_scores = (df_processed[col] - df_processed[col].mean()) / df_processed[col].std()
            outliers = (abs(z_scores) > threshold).sum()
            
            # Aykırı değerleri işle
            df_processed.loc[abs(z_scores) > threshold, col] = np.sign(z_scores) * threshold * df_processed[col].std() + df_processed[col].mean()
        
        if outliers > 0:
            logger.info(f"{col} sütununda {outliers} aykırı değer işlendi")
        outliers_removed += outliers
    
    logger.info(f"Toplam {outliers_removed} aykırı değer işlendi")
    return df_processed


def encode_categorical_features(df, columns=None, method='onehot', max_categories=10):
    """
    Kategorik değişkenleri kodlar.
    
    Args:
        df (pandas.DataFrame): İşlenecek veri seti
        columns (list): İşlenecek sütunlar (None ise tüm kategorik sütunlar)
        method (str): Kodlama yöntemi ('onehot' veya 'label')
        max_categories (int): One-hot encoding için maksimum benzersiz değer sayısı
        
    Returns:
        pandas.DataFrame: Kategorik değişkenleri kodlanmış veri seti
    """
    logger.info(f"Kategorik değişkenler kodlanıyor. Yöntem: {method}")
    
    # Kopyasını oluştur
    df_processed = df.copy()
    
    # İşlenecek sütunları belirle
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns
    
    if method == 'onehot':
        # One-hot encoding
        for col in columns:
            # Eğer çok fazla benzersiz değer varsa, label encoding kullan
            if df[col].nunique() <= max_categories:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df_processed = pd.concat([df_processed, dummies], axis=1)
                df_processed.drop(col, axis=1, inplace=True)
                logger.info(f"{col} sütunu one-hot encoding ile kodlandı ({df[col].nunique()} kategori)")
            else:
                # Çok fazla kategoride Label Encoding uygula
                logger.info(f"{col} sütunu çok fazla benzersiz değere sahip ({df[col].nunique()}). Label encoding uygulanıyor.")
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df[col].astype(str))
                
    elif method == 'label':
        # Label encoding
        for col in columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df[col].astype(str))
            logger.info(f"{col} sütunu label encoding ile kodlandı")
    
    logger.info("Kategorik değişkenler kodlandı")
    return df_processed


def scale_features(df, columns=None, method='standard'):
    """
    Sayısal değişkenleri ölçeklendirir.
    
    Args:
        df (pandas.DataFrame): İşlenecek veri seti
        columns (list): İşlenecek sütunlar (None ise tüm sayısal sütunlar)
        method (str): Ölçeklendirme yöntemi ('standard', 'minmax', 'robust')
        
    Returns:
        pandas.DataFrame: Ölçeklendirilmiş veri seti
        object: Ölçeklendirici nesne (daha sonra transform için)
    """
    logger.info(f"Öznitelikler ölçeklendiriliyor. Yöntem: {method}")
    
    # Kopyasını oluştur
    df_processed = df.copy()
    
    # İşlenecek sütunları belirle
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Hedef değişkeni (Maaş_TL) dışarıda bırakalım
    if 'Maaş_TL' in columns and len(columns) > 1:
        columns = columns.drop('Maaş_TL')
    
    # Ölçeklendirme
    if method == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    
    df_processed[columns] = scaler.fit_transform(df_processed[columns])
    
    logger.info(f"{len(columns)} sayısal öznitelik {method} yöntemiyle ölçeklendirildi")
    return df_processed, scaler


def feature_engineering_it_salary(df):
    """
    IT sektörü maaş verisi için öznitelik mühendisliği yapar.
    
    Args:
        df (pandas.DataFrame): İşlenecek veri seti
        
    Returns:
        pandas.DataFrame: Yeni öznitelikler eklenmiş veri seti
    """
    logger.info("IT sektörü maaş verisi için öznitelik mühendisliği yapılıyor")
    
    # Kopyasını oluştur
    df_processed = df.copy()
    
    # Deneyim / Yaş oranı (Kariyere başlama yaşı ile ilgili bir gösterge)
    if 'Deneyim_Yıl' in df.columns and 'Yaş' in df.columns:
        df_processed['Deneyim_Yaş_Oranı'] = df['Deneyim_Yıl'] / df['Yaş']
        logger.info("'Deneyim_Yaş_Oranı' özniteliği eklendi")
    
    # İşte kalma süresi (yıl)
    if 'İşe_Başlama_Tarihi' in df.columns:
        df_temp = convert_date_columns(df, ['İşe_Başlama_Tarihi'])
        current_date = datetime.now()
        df_processed['İşte_Kalma_Süresi_Yıl'] = (current_date - df_temp['İşe_Başlama_Tarihi']).dt.days / 365.25
        logger.info("'İşte_Kalma_Süresi_Yıl' özniteliği eklendi")
    
    # Terfi alma süresi
    if 'İşe_Başlama_Tarihi' in df.columns and 'Son_Terfi_Tarihi' in df.columns:
        df_temp = convert_date_columns(df, ['İşe_Başlama_Tarihi', 'Son_Terfi_Tarihi'])
        has_promotion = ~df_temp['Son_Terfi_Tarihi'].isna()
        df_processed['Terfi_Var'] = has_promotion.astype(int)
        
        # Terfi süresi hesaplama (terfi olanlar için)
        if has_promotion.sum() > 0:
            promotion_time = (df_temp.loc[has_promotion, 'Son_Terfi_Tarihi'] - 
                             df_temp.loc[has_promotion, 'İşe_Başlama_Tarihi']).dt.days / 365.25
            df_processed.loc[has_promotion, 'İşe_Başlama_Terfi_Arası_Yıl'] = promotion_time
            df_processed['İşe_Başlama_Terfi_Arası_Yıl'].fillna(-1, inplace=True)
            logger.info("'Terfi_Var' ve 'İşe_Başlama_Terfi_Arası_Yıl' öznitelikleri eklendi")
    
    # Sertifika sayısı (virgülle ayrılmış sertifika listesinden)
    if 'Sertifikalar' in df.columns:
        df_processed['Sertifika_Sayısı'] = df['Sertifikalar'].apply(
            lambda x: 0 if pd.isna(x) or x == '' else len(str(x).split(','))
        )
        logger.info("'Sertifika_Sayısı' özniteliği eklendi")
    
    # Kullanılan teknoloji sayısı
    if 'Kullandığı_Teknolojiler' in df.columns:
        df_processed['Teknoloji_Sayısı'] = df['Kullandığı_Teknolojiler'].apply(
            lambda x: 0 if pd.isna(x) or x == '' else len(str(x).split(','))
        )
        logger.info("'Teknoloji_Sayısı' özniteliği eklendi")
    
    # Ana programlama dili tecrübesi (varsa)
    if 'Ana_Programlama_Dili' in df.columns and 'Deneyim_Yıl' in df.columns:
        df_processed['Ana_Dil_Var'] = (~df['Ana_Programlama_Dili'].isna() & 
                                     (df['Ana_Programlama_Dili'] != '')).astype(int)
        # Ana dil deneyimini deneyim yılının %70'i olarak varsayalım (basitleştirme)
        df_processed['Ana_Dil_Deneyim'] = df_processed['Ana_Dil_Var'] * df['Deneyim_Yıl'] * 0.7
        logger.info("'Ana_Dil_Var' ve 'Ana_Dil_Deneyim' öznitelikleri eklendi")
    
    # Uzaktan çalışma faktörü
    if 'Çalışma_Şekli' in df.columns and 'Uzaktan_Çalışma_Oranı' in df.columns:
        df_processed['Uzaktan_Çalışma_Faktörü'] = df['Uzaktan_Çalışma_Oranı'] / 100
        logger.info("'Uzaktan_Çalışma_Faktörü' özniteliği eklendi")
    
    # İngilizce seviyesi numerik dönüşüm
    if 'İngilizce_Seviyesi' in df.columns:
        level_map = {
            'Başlangıç': 1,
            'Orta': 2,
            'İleri': 3,
            'İleri Düzey': 4,
            'Anadil': 5
        }
        df_processed['İngilizce_Seviyesi_Sayısal'] = df['İngilizce_Seviyesi'].map(level_map).fillna(0)
        logger.info("'İngilizce_Seviyesi_Sayısal' özniteliği eklendi")
    
    # Şehir kategorisi (büyük şehir, orta büyüklükte şehir, küçük şehir)
    if 'Şehir' in df.columns:
        # Büyük şehirler
        buyuk_sehirler = ['İstanbul', 'Ankara', 'İzmir']
        # Orta büyüklükte şehirler
        orta_sehirler = ['Bursa', 'Antalya', 'Kocaeli', 'Konya', 'Adana', 'Gaziantep']
        
        def sehir_kategorisi(sehir):
            if sehir in buyuk_sehirler:
                return 'Büyük'
            elif sehir in orta_sehirler:
                return 'Orta'
            else:
                return 'Küçük'
        
        df_processed['Şehir_Kategorisi'] = df['Şehir'].apply(sehir_kategorisi)
        logger.info("'Şehir_Kategorisi' özniteliği eklendi")
    
    # Tecrübe - Eğitim grubu
    if 'Deneyim_Yıl' in df.columns and 'Eğitim_Seviyesi' in df.columns:
        # Tecrübe grupları
        def tecrube_grubu(yil):
            if yil < 2:
                return 'Junior'
            elif yil < 5:
                return 'Mid-Level'
            elif yil < 10:
                return 'Senior'
            else:
                return 'Expert'
        
        # Eğitim grubu
        def egitim_grubu(seviye):
            if seviye in ['Doktora', 'Yüksek Lisans']:
                return 'İleri'
            elif seviye in ['Lisans', 'Önlisans']:
                return 'Standart'
            else:
                return 'Temel'
        
        df_processed['Tecrübe_Grubu'] = df['Deneyim_Yıl'].apply(tecrube_grubu)
        df_processed['Eğitim_Grubu'] = df['Eğitim_Seviyesi'].apply(egitim_grubu)
        df_processed['Tecrübe_Eğitim_Grubu'] = df_processed['Tecrübe_Grubu'] + "_" + df_processed['Eğitim_Grubu']
        logger.info("'Tecrübe_Grubu', 'Eğitim_Grubu' ve 'Tecrübe_Eğitim_Grubu' öznitelikleri eklendi")
    
    logger.info(f"Öznitelik mühendisliği tamamlandı. Toplam {df_processed.shape[1] - df.shape[1]} yeni öznitelik eklendi")
    return df_processed


def create_preprocessing_pipeline(numeric_features, categorical_features, ordinal_features=None, ordinal_categories=None):
    """
    Ön işleme pipeline'ı oluşturur.
    
    Args:
        numeric_features (list): Sayısal öznitelikler
        categorical_features (list): Kategorik öznitelikler
        ordinal_features (list, optional): Sıralı kategorik öznitelikler
        ordinal_categories (dict, optional): Sıralı kategoriler için kategori dizileri
        
    Returns:
        sklearn.pipeline.Pipeline: Ön işleme pipeline'ı
    """
    logger.info("Ön işleme pipeline'ı oluşturuluyor")
    
    transformers = []
    
    # Sayısal öznitelikler için işlem adımları
    if numeric_features:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_transformer, numeric_features))
    
    # Kategorik öznitelikler için işlem adımları
    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_transformer, categorical_features))
    
    # Sıralı kategorik öznitelikler için işlem adımları
    if ordinal_features and ordinal_categories:
        from sklearn.preprocessing import OrdinalEncoder
        ordinal_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder(categories=ordinal_categories))
        ])
        transformers.append(('ord', ordinal_transformer, ordinal_features))
    
    # Column Transformer ile birleştirme
    preprocessor = ColumnTransformer(transformers=transformers)
    
    logger.info(f"Ön işleme pipeline'ı oluşturuldu: {len(transformers)} transformer")
    return preprocessor


def prepare_it_salary_data(df, target_column='Maaş_TL', drop_columns=None, feature_eng=True, handle_outliers_flag=True):
    """
    IT sektörü maaş verilerini modelleme için hazırlar.
    
    Args:
        df (pandas.DataFrame): İşlenecek veri seti
        target_column (str): Hedef değişken adı
        drop_columns (list, optional): Düşürülecek sütunlar
        feature_eng (bool): Öznitelik mühendisliği uygulanacak mı?
        handle_outliers_flag (bool): Aykırı değerler işlenecek mi?
        
    Returns:
        pandas.DataFrame: İşlenmiş veri seti
        dict: Sütun tipleri bilgisi
    """
    logger.info(f"IT sektörü maaş verileri hazırlanıyor. Hedef değişken: {target_column}")
    
    # Kopyasını oluştur
    df_processed = df.copy()
    
    # Düşürülecek sütunlar
    if drop_columns:
        df_processed = df_processed.drop(columns=drop_columns, errors='ignore')
        logger.info(f"{len(drop_columns)} sütun düşürüldü: {drop_columns}")
    
    # Tarih sütunlarını dönüştür
    date_columns = [col for col in df_processed.columns if 'Tarih' in col or 'tarihi' in col]
    if date_columns:
        df_processed = convert_date_columns(df_processed, date_columns)
    
    # Eksik değerleri işle
    df_processed = handle_missing_values(df_processed)
    
    # Aykırı değerleri işle (hedef değişken hariç)
    if handle_outliers_flag:
        outlier_columns = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if target_column in outlier_columns:
            outlier_columns.remove(target_column)
        df_processed = handle_outliers(df_processed, columns=outlier_columns)
    
    # Öznitelik mühendisliği
    if feature_eng:
        df_processed = feature_engineering_it_salary(df_processed)
    
    # Sütun tiplerini belirle
    column_types = identify_column_types(df_processed)
    
    logger.info(f"IT sektörü maaş verileri hazırlandı. Boyut: {df_processed.shape}")
    return df_processed, column_types


def prepare_data_for_modeling(df, target_column='Maaş_TL', categorical_encoding='onehot', 
                             scaling=True, test_size=0.2, random_state=42):
    """
    Veri setini modelleme için hazırlar ve eğitim/test setlerine böler.
    
    Args:
        df (pandas.DataFrame): İşlenecek veri seti
        target_column (str): Hedef değişken adı
        categorical_encoding (str): Kategorik değişken kodlama yöntemi
        scaling (bool): Ölçeklendirme yapılacak mı?
        test_size (float): Test seti oranı
        random_state (int): Rastgele sayı üreteci için tohum değeri
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessor) içeren tuple
    """
    from sklearn.model_selection import train_test_split
    
    logger.info("Veri seti modelleme için hazırlanıyor")
    
    # Hedef değişkeni ayır
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Sütun tiplerini belirle
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Veriyi böl
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Önişleme pipeline'ı oluştur
    preprocessor = create_preprocessing_pipeline(
        numeric_features=numeric_features, 
        categorical_features=categorical_features
    )
    
    # Pipeline'ı eğit ve dönüştür
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Özelliklerin isimlerini al (one-hot encoding sonrası)
    feature_names = []
    if numeric_features:
        feature_names.extend(numeric_features)
    
    if categorical_features and categorical_encoding == 'onehot':
        # OneHotEncoder'ın ürettiği özellik isimlerini al
        encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = []
        for i, col in enumerate(categorical_features):
            cat_values = encoder.categories_[i]
            for val in cat_values[:-1]:  # drop_first=True olduğu için sonuncuyu atlıyoruz
                cat_features.append(f"{col}_{val}")
        feature_names.extend(cat_features)
    
    logger.info(f"Veri seti modelleme için hazırlandı. Eğitim seti: {X_train_processed.shape}, Test seti: {X_test_processed.shape}")
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names


def select_important_features(X, y, method='f_regression', k=10):
    """
    En önemli özellikleri seçer.
    
    Args:
        X (numpy.ndarray): Özellik matrisi
        y (numpy.ndarray): Hedef değişken
        method (str): Özellik seçim yöntemi ('f_regression', 'mutual_info', veya 'mrmr')
        k (int): Seçilecek özellik sayısı
        
    Returns:
        tuple: (Seçilmiş özellik indeksleri, Skor değerleri)
    """
    logger.info(f"Özellik seçimi yapılıyor. Yöntem: {method}, k: {k}")
    
    if method == 'f_regression':
        from sklearn.feature_selection import f_regression
        
        # F değerlerini hesapla
        f_values, p_values = f_regression(X, y)
        
        # En yüksek F değerine sahip k özelliği seç
        indices = np.argsort(f_values)[::-1][:k]
        scores = f_values[indices]
        
    elif method == 'mutual_info':
        from sklearn.feature_selection import mutual_info_regression
        
        # Karşılıklı bilgi değerlerini hesapla
        mi_values = mutual_info_regression(X, y)
        
        # En yüksek MI değerine sahip k özelliği seç
        indices = np.argsort(mi_values)[::-1][:k]
        scores = mi_values[indices]
        
    elif method == 'mrmr':
        try:
            # mRMR ayrı bir paket gerektiriyor, eğer yüklü değilse normal f_regression kullan
            from mrmr import mrmr_regression
            
            # Uygun formata dönüştür
            if isinstance(X, np.ndarray):
                X_df = pd.DataFrame(X)
            else:
                X_df = X
                
            # mRMR uygula
            selected_features = mrmr_regression(X_df, y, K=k)
            
            # İndeksleri ve skorları ayarla
            indices = [int(f.replace('X', '')) for f in selected_features]
            scores = np.ones(len(indices))  # mRMR doğrudan skor vermez
        except ImportError:
            logger.warning("mRMR paketi bulunamadı. F-regression kullanılıyor.")
            return select_important_features(X, y, method='f_regression', k=k)
    else:
        raise ValueError(f"Bilinmeyen özellik seçim yöntemi: {method}")
    
    logger.info(f"Özellik seçimi tamamlandı. {len(indices)} özellik seçildi")
    return indices, scores


def extract_feature_importance(model, feature_names=None):
    """
    Modelden özellik önem derecelerini çıkarır.
    
    Args:
        model: Eğitilmiş model
        feature_names (list, optional): Özellik isimleri
        
    Returns:
        pandas.DataFrame: Özellik önem dereceleri
    """
    logger.info("Modelden özellik önem dereceleri çıkarılıyor")
    
    # Model türüne göre özellik önem derecelerini çıkar
    if hasattr(model, 'feature_importances_'):
        # Ağaç tabanlı modeller (Random Forest, Decision Tree, XGBoost, vb.)
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Lineer modeller (Lineer Regresyon, Ridge, Lasso, vb.)
        importance = np.abs(model.coef_)  # Mutlak değer al
    else:
        logger.warning("Model özellik önem derecelerini desteklemiyor")
        return None
    
    # Özellik isimleri yoksa indeksler kullan
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(importance))]
    
    # Özellik isimlerinin uzunluğunu kontrol et
    if len(feature_names) != len(importance):
        logger.warning(f"Özellik isimleri ({len(feature_names)}) ve önem dereceleri ({len(importance)}) boyutları uyumsuz")
        feature_names = [f"Feature_{i}" for i in range(len(importance))]
    
    # DataFrame oluştur ve önem derecesine göre sırala
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    logger.info("Özellik önem dereceleri çıkarıldı")
    return feature_importance


def prepare_new_data(new_data, preprocessor, target_column=None):
    """
    Yeni gelen verileri model için hazırlar (aynı önişleme adımlarını uygular).
    
    Args:
        new_data (pandas.DataFrame): Yeni veri seti
        preprocessor: Eğitim verileri için kullanılan ön işleme pipeline'ı
        target_column (str, optional): Hedef değişken adı (varsa)
        
    Returns:
        numpy.ndarray: İşlenmiş yeni veriler
    """
    logger.info("Yeni veriler hazırlanıyor")
    
    # Hedef değişkeni ayır (varsa)
    if target_column and target_column in new_data.columns:
        X_new = new_data.drop(target_column, axis=1)
        y_new = new_data[target_column]
    else:
        X_new = new_data
        y_new = None
    
    # Önişleme pipeline'ını uygula
    X_new_processed = preprocessor.transform(X_new)
    
    logger.info(f"Yeni veriler hazırlandı. Boyut: {X_new_processed.shape}")
    return X_new_processed, y_new


if __name__ == "__main__":
    # Test işlevselliği
    try:
        csv_path = "../../data/raw/turkiye_it_sektoru_calisanlari.csv"
        df = load_data(csv_path)
        
        # Temel EDA
        eda_results = basic_eda(df)
        print(f"Veri seti boyutu: {eda_results['shape']}")
        print(f"Eksik değerler: {sum(eda_results['missing_values'].values())}")
        
        # Sütunları tanımla
        column_types = identify_column_types(df)
        
        # Eksik değerleri işle
        df_processed = handle_missing_values(df)
        
        # Aykırı değerleri işle
        df_processed = handle_outliers(df_processed)
        
        # Kategorik değişkenleri kodla
        df_processed = encode_categorical_features(df_processed)
        
        # Ölçeklendirme
        df_processed, _ = scale_features(df_processed)
        
        # Öznitelik mühendisliği
        df_processed = feature_engineering_it_salary(df_processed)
        
        # IT maaş verilerini hazırla
        df_prepared, column_types = prepare_it_salary_data(df)
        
        # Tam pipeline kullanımı
        X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data_for_modeling(df_prepared)
        
        print(f"İşleme tamamlandı. X_train boyutu: {X_train.shape}")
        
        print("Ön işleme modülü başarıyla test edildi.")
    except Exception as e:
        logger.error(f"Test sırasında hata oluştu: {str(e)}")