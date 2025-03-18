import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from datetime import datetime
import json

# Loglama yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Proje kök dizinini bulmak için yardımcı fonksiyon
def get_project_root():
    """Proje kök dizinini döndürür."""
    current_path = Path(os.path.abspath(__file__))
    # src/data/data_loader.py'dan 2 seviye yukarı çıkıyoruz
    project_root = current_path.parent
    return project_root


def load_it_salary_data(file_path=None):
    """
    IT sektörü maaş verilerini yükler.
    
    Args:
        file_path (str, optional): Veri dosyasının yolu. Belirtilmezse varsayılan konum kullanılır.
        
    Returns:
        pandas.DataFrame: Yüklenen maaş veri seti
    """
    if file_path is None:
        # Varsayılan dosya yolu
        project_root = get_project_root()
        file_path = os.path.join(project_root, 'data', 'raw', 'turkiye_it_sektoru_calisanlari.csv')
    
    try:
        logger.info(f"IT sektörü maaş verileri yükleniyor: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"IT sektörü maaş verileri başarıyla yüklendi. Boyut: {df.shape}")
        
        # Tarih sütunlarını dönüştür
        date_cols = [col for col in df.columns if 'Tarih' in col or 'tarihi' in col]
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                logger.info(f"{col} sütunu datetime türüne dönüştürüldü")
            except Exception as e:
                logger.warning(f"{col} sütunu datetime türüne dönüştürülemedi: {str(e)}")
        
        return df
    except Exception as e:
        logger.error(f"Veri yüklenirken hata oluştu: {str(e)}")
        raise


def load_external_economic_data(data_type='kur', date_range=None):
    """
    Dış kaynaklardan ekonomik veri yükler (kur verileri, enflasyon, TÜFE vb.).
    
    Args:
        data_type (str): Yüklenecek veri türü ('kur', 'enflasyon', 'tufe', vb.)
        date_range (tuple, optional): Başlangıç ve bitiş tarihleri (YYYY-MM-DD formatında)
        
    Returns:
        pandas.DataFrame: Yüklenen dış veri
    """
    project_root = get_project_root()
    external_data_dir = os.path.join(project_root, 'data', 'external')
    
    if data_type == 'kur':
        file_path = os.path.join(external_data_dir, 'doviz_kurlari.csv')
    elif data_type == 'enflasyon':
        file_path = os.path.join(external_data_dir, 'enflasyon.csv')
    elif data_type == 'tufe':
        file_path = os.path.join(external_data_dir, 'tufe.csv')
    elif data_type == 'ufe':
        file_path = os.path.join(external_data_dir, 'ufe.csv')
    elif data_type == 'bist':
        file_path = os.path.join(external_data_dir, 'bist100.csv')
    else:
        raise ValueError(f"Bilinmeyen veri türü: {data_type}")
    
    try:
        logger.info(f"{data_type} verileri yükleniyor")
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # Tarih sütununu datetime'a dönüştür
            date_cols = [col for col in df.columns if 'tarih' in col.lower() or 'date' in col.lower()]
            if date_cols:
                for col in date_cols:
                    df[col] = pd.to_datetime(df[col])
            
            # Tarih aralığı filtreleme
            if date_range and date_cols:
                start_date, end_date = date_range
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                df = df[(df[date_cols[0]] >= start_date) & (df[date_cols[0]] <= end_date)]
            
            logger.info(f"{data_type} verileri başarıyla yüklendi. Boyut: {df.shape}")
            return df
        else:
            logger.warning(f"{file_path} bulunamadı")
            return None
            
    except Exception as e:
        logger.error(f"{data_type} verileri yüklenirken hata oluştu: {str(e)}")
        return None


def load_new_employee_data(file_path):
    """
    Yeni çalışan verilerini yükler.
    
    Args:
        file_path (str): Veri dosyasının yolu
        
    Returns:
        pandas.DataFrame: Yüklenen çalışan verisi
    """
    try:
        logger.info(f"Yeni çalışan verileri yükleniyor: {file_path}")
        
        # Dosya uzantısına göre yükleme yöntemi
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
        elif file_ext in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        elif file_ext == '.json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Desteklenmeyen dosya uzantısı: {file_ext}")
        
        logger.info(f"Yeni çalışan verileri başarıyla yüklendi. Boyut: {df.shape}")
        return df
    
    except Exception as e:
        logger.error(f"Yeni çalışan verileri yüklenirken hata oluştu: {str(e)}")
        raise


def save_processed_data(df, file_name=None, folder=None):
    """
    İşlenmiş veriyi kaydeder.
    
    Args:
        df (pandas.DataFrame): Kaydedilecek veri seti
        file_name (str, optional): Dosya adı. Belirtilmezse zaman damgalı bir isim oluşturulur.
        folder (str, optional): Klasör yolu. Belirtilmezse varsayılan konum kullanılır.
        
    Returns:
        str: Kaydedilen dosyanın tam yolu
    """
    if folder is None:
        project_root = get_project_root()
        folder = os.path.join(project_root, 'data', 'processed')
    
    # Klasörün var olduğundan emin ol
    os.makedirs(folder, exist_ok=True)
    
    if file_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"processed_data_{timestamp}.csv"
    
    file_path = os.path.join(folder, file_name)
    
    try:
        logger.info(f"İşlenmiş veri kaydediliyor: {file_path}")
        df.to_csv(file_path, index=False)
        logger.info("Veri başarıyla kaydedildi")
        return file_path
    except Exception as e:
        logger.error(f"Veri kaydedilirken hata oluştu: {str(e)}")
        raise


def save_model_results(model_name, metrics, feature_importance=None, model_params=None, folder=None):
    """
    Model sonuçlarını ve ilgili bilgileri kaydeder.
    
    Args:
        model_name (str): Model adı
        metrics (dict): Performans metrikleri
        feature_importance (pandas.DataFrame, optional): Özellik önem dereceleri
        model_params (dict, optional): Model parametreleri
        folder (str, optional): Klasör yolu. Belirtilmezse varsayılan konum kullanılır.
        
    Returns:
        str: Kaydedilen dosyanın tam yolu
    """
    if folder is None:
        project_root = get_project_root()
        folder = os.path.join(project_root, 'models', 'results')
    
    # Klasörün var olduğundan emin ol
    os.makedirs(folder, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{model_name}_results_{timestamp}.json"
    file_path = os.path.join(folder, file_name)
    
    # Sonuçları bir araya getir
    results = {
        "model_name": model_name,
        "timestamp": timestamp,
        "metrics": metrics,
        "parameters": model_params if model_params else {}
    }
    
    try:
        logger.info(f"Model sonuçları kaydediliyor: {file_path}")
        
        # JSON dosyasına kaydet
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        # Özellik önem derecelerini ayrı bir CSV dosyasına kaydet (varsa)
        if feature_importance is not None:
            fi_file_name = f"{model_name}_feature_importance_{timestamp}.csv"
            fi_file_path = os.path.join(folder, fi_file_name)
            feature_importance.to_csv(fi_file_path, index=False)
            logger.info(f"Özellik önem dereceleri kaydedildi: {fi_file_path}")
        
        logger.info("Model sonuçları başarıyla kaydedildi")
        return file_path
    
    except Exception as e:
        logger.error(f"Model sonuçları kaydedilirken hata oluştu: {str(e)}")
        raise


def split_data_with_time(df, date_column, test_size=0.2, valid_size=0.2):
    """
    Veri setini tarihe göre eğitim, doğrulama ve test alt kümelerine ayırır.
    
    Args:
        df (pandas.DataFrame): Bölünecek veri seti
        date_column (str): Tarih sütunu adı
        test_size (float): Test seti oranı
        valid_size (float): Doğrulama seti oranı
        
    Returns:
        tuple: (train_df, valid_df, test_df) içeren tuple
    """
    logger.info(f"Veri seti tarihe göre bölünüyor. Tarih sütunu: {date_column}")
    
    # Tarihe göre sırala
    df_sorted = df.sort_values(by=date_column)
    
    # Bölme indekslerini hesapla
    n = len(df_sorted)
    test_start_idx = int(n * (1 - test_size))
    valid_start_idx = int(test_start_idx * (1 - valid_size))
    
    # Eğitim, doğrulama ve test setlerini oluştur
    train_df = df_sorted.iloc[:valid_start_idx].copy()
    valid_df = df_sorted.iloc[valid_start_idx:test_start_idx].copy()
    test_df = df_sorted.iloc[test_start_idx:].copy()
    
    logger.info(f"Veri seti tarihe göre bölündü: Eğitim {train_df.shape}, Doğrulama {valid_df.shape}, Test {test_df.shape}")
    return train_df, valid_df, test_df


def create_data_splits(df, target_column='Maaş_TL', test_size=0.2, val_size=0.25, random_state=42):
    """
    Veri setini eğitim, doğrulama ve test alt kümelerine ayırır.
    
    Args:
        df (pandas.DataFrame): Bölünecek veri seti
        target_column (str): Hedef değişken adı
        test_size (float): Test seti oranı (tüm veriye göre)
        val_size (float): Doğrulama seti oranı (eğitim verisi içinden)
        random_state (int): Rastgele sayı üreteci için tohum değeri
        
    Returns:
        dict: Veri bölmeleri ve meta verileri içeren sözlük
    """
    from sklearn.model_selection import train_test_split
    
    logger.info("Veri seti bölünüyor")
    
    # Önce test setini ayır
    df_train_val, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    # Sonra doğrulama setini ayır
    df_train, df_val = train_test_split(
        df_train_val, test_size=val_size, random_state=random_state
    )
    
    # Feature ve target ayrımı
    X_train = df_train.drop(target_column, axis=1)
    y_train = df_train[target_column]
    
    X_val = df_val.drop(target_column, axis=1)
    y_val = df_val[target_column]
    
    X_test = df_test.drop(target_column, axis=1)
    y_test = df_test[target_column]
    
    # Meta veriler
    metadata = {
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_column": target_column,
        "test_size": test_size,
        "val_size": val_size,
        "random_state": random_state,
        "train_shape": df_train.shape,
        "val_shape": df_val.shape,
        "test_shape": df_test.shape,
        "feature_names": X_train.columns.tolist(),
    }
    
    # Veri istatistikleri
    train_stats = {
        "mean": float(y_train.mean()),
        "median": float(y_train.median()),
        "min": float(y_train.min()),
        "max": float(y_train.max()),
        "std": float(y_train.std())
    }
    metadata["target_stats"] = train_stats
    
    result = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "metadata": metadata
    }
    
    logger.info(f"Veri seti 3 parçaya bölündü: Eğitim {X_train.shape}, Doğrulama {X_val.shape}, Test {X_test.shape}")
    return result


def save_data_splits(splits_data, folder=None):
    """
    Bölünmüş veri setlerini ve meta verileri kaydeder.
    
    Args:
        splits_data (dict): Veri bölmelerini ve meta verileri içeren sözlük
        folder (str, optional): Klasör yolu. Belirtilmezse varsayılan konum kullanılır.
        
    Returns:
        str: Kaydedilen klasörün tam yolu
    """
    if folder is None:
        project_root = get_project_root()
        folder = os.path.join(project_root, 'data', 'splits')
    
    # Klasörün var olduğundan emin ol
    os.makedirs(folder, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    split_folder = os.path.join(folder, f"split_{timestamp}")
    os.makedirs(split_folder, exist_ok=True)
    
    try:
        logger.info(f"Veri bölmeleri kaydediliyor: {split_folder}")
        
        # X_train, X_val, X_test kaydet
        for name in ['X_train', 'X_val', 'X_test']:
            splits_data[name].to_csv(os.path.join(split_folder, f"{name}.csv"), index=False)
        
        # y_train, y_val, y_test kaydet
        for name in ['y_train', 'y_val', 'y_test']:
            pd.DataFrame({splits_data['metadata']['target_column']: splits_data[name]}).to_csv(
                os.path.join(split_folder, f"{name}.csv"), index=False
            )
        
        # Meta verileri JSON olarak kaydet
        with open(os.path.join(split_folder, "metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(splits_data['metadata'], f, ensure_ascii=False, indent=4)
        
        logger.info("Veri bölmeleri başarıyla kaydedildi")
        return split_folder
    
    except Exception as e:
        logger.error(f"Veri bölmeleri kaydedilirken hata oluştu: {str(e)}")
        raise


def load_data_splits(split_folder):
    """
    Kaydedilmiş veri bölmelerini yükler.
    
    Args:
        split_folder (str): Veri bölmelerinin bulunduğu klasör yolu
        
    Returns:
        dict: Yüklenen veri bölmeleri ve meta veriler
    """
    try:
        logger.info(f"Veri bölmeleri yükleniyor: {split_folder}")
        
        # Meta verileri yükle
        with open(os.path.join(split_folder, "metadata.json"), 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        target_column = metadata['target_column']
        
        # X_train, X_val, X_test yükle
        X_train = pd.read_csv(os.path.join(split_folder, "X_train.csv"))
        X_val = pd.read_csv(os.path.join(split_folder, "X_val.csv"))
        X_test = pd.read_csv(os.path.join(split_folder, "X_test.csv"))
        
        # y_train, y_val, y_test yükle
        y_train = pd.read_csv(os.path.join(split_folder, f"y_train.csv"))[target_column]
        y_val = pd.read_csv(os.path.join(split_folder, f"y_val.csv"))[target_column]
        y_test = pd.read_csv(os.path.join(split_folder, f"y_test.csv"))[target_column]
        
        result = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "metadata": metadata
        }
        
        logger.info(f"Veri bölmeleri yüklendi: Eğitim {X_train.shape}, Doğrulama {X_val.shape}, Test {X_test.shape}")
        return result
        
    except Exception as e:
        logger.error(f"Veri bölmeleri yüklenirken hata oluştu: {str(e)}")
        raise


def load_test_employee(file_path=None):
    """
    Test için bir IT çalışanı verisi yükler.
    
    Args:
        file_path (str, optional): Veri dosyasının yolu. Belirtilmezse örnek bir çalışan oluşturulur.
        
    Returns:
        pandas.DataFrame: Yüklenen test çalışan verisi
    """
    if file_path and os.path.exists(file_path):
        try:
            logger.info(f"Test çalışan verisi yükleniyor: {file_path}")
            
            # Dosya uzantısına göre yükleme yöntemi
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.csv':
                employee_df = pd.read_csv(file_path)
            elif file_ext in ['.xls', '.xlsx']:
                employee_df = pd.read_excel(file_path)
            elif file_ext == '.json':
                employee_df = pd.read_json(file_path)
            else:
                raise ValueError(f"Desteklenmeyen dosya uzantısı: {file_ext}")
            
            logger.info(f"Test çalışan verisi başarıyla yüklendi. Boyut: {employee_df.shape}")
            return employee_df
            
        except Exception as e:
            logger.error(f"Test çalışan verisi yüklenirken hata oluştu: {str(e)}")
            logger.info("Varsayılan test çalışanı oluşturuluyor...")
    
    # Örnek bir çalışan oluştur
    logger.info("Örnek test çalışanı oluşturuluyor")
    
    test_employee = {
        "Cinsiyet": ["Erkek"],
        "Yaş": [30],
        "Şehir": ["İstanbul"],
        "İlçe": ["Kadıköy"],
        "Rol_Kategorisi": ["Yazılım Geliştirme"],
        "Rol": ["Full-Stack Geliştirici"],
        "Kıdem": ["Mid."],
        "Deneyim_Yıl": [4.5],
        "Eğitim_Seviyesi": ["Lisans"],
        "Eğitim_Alanı": ["Bilgisayar Mühendisliği"],
        "Çalışma_Şekli": ["Tam Zamanlı"],
        "Uzaktan_Çalışma_Oranı": [20],
        "Ana_Programlama_Dili": ["Python"],
        "İngilizce_Seviyesi": ["İleri"],
        "Toplam_Proje_Sayısı": [12],
        "Teknik_Beceri_Puanı": [75],
        "Soft_Skill_Puanı": [65]
    }
    
    employee_df = pd.DataFrame(test_employee)
    logger.info("Örnek test çalışanı oluşturuldu")
    return employee_df


def combine_salary_with_economic_data(salary_df, economic_data_dfs, date_column='İşe_Başlama_Tarihi'):
    """
    Maaş verileriyle ekonomik verileri birleştirir.
    
    Args:
        salary_df (pandas.DataFrame): Maaş veri seti
        economic_data_dfs (dict): Ekonomik veri setleri sözlüğü (anahtar: veri türü, değer: DataFrame)
        date_column (str): Maaş verisindeki tarih sütunu
        
    Returns:
        pandas.DataFrame: Birleştirilmiş veri seti
    """
    logger.info("Maaş verileri ile ekonomik veriler birleştiriliyor")
    
    # Tarih sütununu dönüştür
    if date_column in salary_df.columns:
        salary_df[date_column] = pd.to_datetime(salary_df[date_column])
    else:
        logger.warning(f"{date_column} sütunu maaş verisinde bulunamadı")
        return salary_df
    
    # Birleştirilmiş veriyi oluştur
    combined_df = salary_df.copy()
    
    # Her bir ekonomik veri türü için
    for data_type, eco_df in economic_data_dfs.items():
        if eco_df is None or eco_df.empty:
            logger.warning(f"{data_type} verisi boş, atlanıyor")
            continue
        
        # Ekonomik verideki tarih sütununu bul
        eco_date_cols = [col for col in eco_df.columns if 'tarih' in col.lower() or 'date' in col.lower()]
        if not eco_date_cols:
            logger.warning(f"{data_type} verisinde tarih sütunu bulunamadı, atlanıyor")
            continue
            
        eco_date_col = eco_date_cols[0]
        eco_df[eco_date_col] = pd.to_datetime(eco_df[eco_date_col])
        
        # Her bir maaş kaydı için en yakın ekonomik veri kaydını bul
        for i, row in combined_df.iterrows():
            record_date = row[date_column]
            
            # En yakın tarihi bul
            closest_idx = (eco_df[eco_date_col] - record_date).abs().idxmin()
            closest_record = eco_df.loc[closest_idx]
            
            # Ekonomik veri sütunlarını ekle (tarih sütunu hariç)
            for col in eco_df.columns:
                if col != eco_date_col:
                    combined_df.at[i, f"{data_type}_{col}"] = closest_record[col]
        
        logger.info(f"{data_type} verileri başarıyla birleştirildi")
    
    logger.info(f"Maaş verileri ile ekonomik veriler birleştirildi. Yeni boyut: {combined_df.shape}")
    return combined_df


def merge_datasets(main_df, other_df, on, how='left'):
    """
    İki veri setini birleştirir.
    
    Args:
        main_df (pandas.DataFrame): Ana veri seti
        other_df (pandas.DataFrame): Diğer veri seti
        on (str or list): Birleştirme anahtarı
        how (str): Birleştirme yöntemi ('left', 'right', 'inner', 'outer')
        
    Returns:
        pandas.DataFrame: Birleştirilmiş veri seti
    """
    logger.info(f"Veri setleri birleştiriliyor. Yöntem: {how}")
    
    try:
        merged_df = pd.merge(main_df, other_df, on=on, how=how)
        logger.info(f"Veri setleri birleştirildi. Yeni boyut: {merged_df.shape}")
        return merged_df
    except Exception as e:
        logger.error(f"Veri setleri birleştirilirken hata oluştu: {str(e)}")
        raise


def create_sample_data(n=100, random_state=42):
    """
    Test için örnek IT sektörü maaş verisi oluşturur.
    
    Args:
        n (int): Oluşturulacak örnek sayısı
        random_state (int): Rastgele sayı üreteci için tohum değeri
        
    Returns:
        pandas.DataFrame: Oluşturulan örnek veri seti
    """
    logger.info(f"Örnek veri seti oluşturuluyor. Boyut: {n}")
    
    np.random.seed(random_state)
    
    # Kategorik özellikler için değerler
    cinsiyetler = ["Erkek", "Kadın"]
    sehirler = ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya"]
    roller = ["Frontend Geliştirici", "Backend Geliştirici", "Full-Stack Geliştirici", 
              "Veri Bilimci", "DevOps Mühendisi", "Siber Güvenlik Uzmanı"]
    kategoriler = ["Yazılım Geliştirme", "Veri", "DevOps ve Altyapı", "Güvenlik"]
    kidemler = ["Jr.", "Mid.", "Sr.", "Lead"]
    egitim_seviyeleri = ["Önlisans", "Lisans", "Yüksek Lisans", "Doktora"]
    
    # Veri seti
    data = {
        "Cinsiyet": np.random.choice(cinsiyetler, size=n, p=[0.7, 0.3]),
        "Yaş": np.random.randint(22, 50, size=n),
        "Şehir": np.random.choice(sehirler, size=n, p=[0.45, 0.20, 0.15, 0.10, 0.10]),
        "Rol_Kategorisi": np.random.choice(kategoriler, size=n),
        "Rol": np.random.choice(roller, size=n),
        "Kıdem": np.random.choice(kidemler, size=n, p=[0.3, 0.4, 0.2, 0.1]),
        "Deneyim_Yıl": np.random.uniform(0.5, 15, size=n),
        "Eğitim_Seviyesi": np.random.choice(egitim_seviyeleri, size=n, p=[0.1, 0.6, 0.25, 0.05]),
        "Teknik_Beceri_Puanı": np.random.randint(50, 95, size=n),
        "Soft_Skill_Puanı": np.random.randint(40, 90, size=n)
    }
    
    df = pd.DataFrame(data)
    
    # Maaş oluştur (önceki özelliklere dayalı formül)
    base_salary = 15000
    
    # Temel katsayılar
    kidem_dict = {"Jr.": 0.8, "Mid.": 1.0, "Sr.": 1.5, "Lead": 2.0}
    kategori_dict = {"Yazılım Geliştirme": 1.0, "Veri": 1.1, "DevOps ve Altyapı": 1.05, "Güvenlik": 1.15}
    sehir_dict = {"İstanbul": 1.0, "Ankara": 0.9, "İzmir": 0.85, "Bursa": 0.8, "Antalya": 0.8}
    egitim_dict = {"Önlisans": 0.9, "Lisans": 1.0, "Yüksek Lisans": 1.1, "Doktora": 1.2}
    
    # Maaş hesaplama
    maaslar = []
    for i in range(n):
        kidem_carpani = kidem_dict[df.loc[i, "Kıdem"]]
        kategori_carpani = kategori_dict[df.loc[i, "Rol_Kategorisi"]]
        sehir_carpani = sehir_dict[df.loc[i, "Şehir"]]
        egitim_carpani = egitim_dict[df.loc[i, "Eğitim_Seviyesi"]]
        deneyim_carpani = 1.0 + (df.loc[i, "Deneyim_Yıl"] * 0.05)
        beceri_carpani = 1.0 + ((df.loc[i, "Teknik_Beceri_Puanı"] / 100) * 0.2)
        
        maas = base_salary * kidem_carpani * kategori_carpani * sehir_carpani * egitim_carpani * deneyim_carpani * beceri_carpani
        maas = int(maas * np.random.uniform(0.9, 1.1))  # Rastgele varyasyon ekle
        maaslar.append(maas)
    
    df["Maaş_TL"] = maaslar
    
    logger.info(f"Örnek veri seti oluşturuldu. Boyut: {df.shape}")
    return df


if __name__ == "__main__":
    # Test işlevselliği
    try:
        # Örnek veri seti oluştur ve kaydet
        sample_df = create_sample_data(n=100)
        sample_file_path = save_processed_data(sample_df, file_name="sample_it_salary_data.csv")
        print(f"Örnek veri seti kaydedildi: {sample_file_path}")
        
        # Örnek veri setini yükle
        loaded_df = load_it_salary_data(sample_file_path)
        print(f"Örnek veri seti yüklendi. Boyut: {loaded_df.shape}")
        
        # Veri setini böl
        splits = create_data_splits(loaded_df)
        split_folder = save_data_splits(splits)
        print(f"Veri bölmeleri kaydedildi: {split_folder}")
        
        # Veri bölmelerini yükle
        loaded_splits = load_data_splits(split_folder)
        print(f"Veri bölmeleri yüklendi: Eğitim {loaded_splits['X_train'].shape}, Test {loaded_splits['X_test'].shape}")
        
        # Test çalışanı oluştur
        test_employee = load_test_employee()
        print(f"Test çalışanı oluşturuldu: {test_employee.iloc[0]['Rol']}")
        
        print("Veri yükleme modülü başarıyla test edildi.")
    except Exception as e:
        logger.error(f"Test sırasında hata oluştu: {str(e)}")