import pandas as pd
import numpy as np
import os
import pickle
import logging
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import sys

# src klasörünü Python yoluna ekle
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(src_dir)

from src.data.data_loader import get_project_root, save_model_results
from src.data.preprocess import prepare_data_for_modeling, extract_feature_importance

# Loglama yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_model_class(model_name):
    """
    Model ismine göre model sınıfını döndürür.
    
    Args:
        model_name (str): Model ismi
        
    Returns:
        class: Model sınıfı
    """
    models = {
        'mlr': LinearRegression,
        'dt': DecisionTreeRegressor,
        'rf': RandomForestRegressor,
        'ridge': Ridge,
        'adaboost': AdaBoostRegressor
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Bilinmeyen model ismi: {model_name}. Desteklenen modeller: {list(models.keys())}")
    
    return models[model_name.lower()]


def train_model(X_train, y_train, model_name='rf', model_params=None, random_state=42):
    """
    Belirtilen model ile eğitim yapar.
    
    Args:
        X_train (numpy.ndarray): Eğitim verisi özellikleri
        y_train (numpy.ndarray): Eğitim verisi hedef değişkeni
        model_name (str): Kullanılacak model ismi ('mlr', 'dt', 'rf', 'ridge', 'adaboost')
        model_params (dict, optional): Model parametreleri
        random_state (int): Rastgele sayı üreteci için tohum değeri
        
    Returns:
        object: Eğitilmiş model
    """
    logger.info(f"{model_name.upper()} modeli eğitiliyor")
    
    # Model sınıfını al
    ModelClass = get_model_class(model_name)
    
    # Varsayılan parametreler
    default_params = {
        'random_state': random_state
    }
    
    # Kullanıcı parametrelerini varsayılan parametrelerle birleştir
    if model_params:
        params = {**default_params, **model_params}
    else:
        params = default_params
    
    # MLR için random_state gerekmez
    if model_name.lower() == 'mlr':
        if 'random_state' in params:
            del params['random_state']
    
    # Ridge için random_state gerekmez
    if model_name.lower() == 'ridge':
        if 'random_state' in params:
            del params['random_state']
    
    # Modeli oluştur
    model = ModelClass(**params)
    
    # Eğitim
    model.fit(X_train, y_train)
    logger.info(f"{model_name.upper()} modeli eğitildi")
    
    return model


def evaluate_model(model, X_test, y_test, X_train=None, y_train=None, feature_names=None):
    """
    Modeli değerlendirir ve sonuçları döndürür.
    
    Args:
        model: Eğitilmiş model
        X_test (numpy.ndarray): Test verisi özellikleri
        y_test (numpy.ndarray): Test verisi hedef değişkeni
        X_train (numpy.ndarray, optional): Eğitim verisi özellikleri
        y_train (numpy.ndarray, optional): Eğitim verisi hedef değişkeni
        feature_names (list, optional): Özellik isimleri
        
    Returns:
        dict: Değerlendirme metrikleri
        pandas.DataFrame: Özellik önem dereceleri
    """
    logger.info("Model değerlendiriliyor")
    
    # Test verisi üzerinde tahmin yap
    y_pred = model.predict(X_test)
    
    # Test metrikleri
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # Yüzde olarak ifade et
    
    metrics = {
        'test_mse': mse,
        'test_rmse': rmse,
        'test_mae': mae,
        'test_r2': r2,
        'test_mape': mape
    }
    
    # Eğitim metrikleri (isteğe bağlı)
    if X_train is not None and y_train is not None:
        y_train_pred = model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        train_mape = mean_absolute_percentage_error(y_train, y_train_pred) * 100
        
        metrics.update({
            'train_mse': train_mse,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'train_mape': train_mape,
            'overfitting_ratio_rmse': train_rmse / rmse if rmse > 0 else float('inf'),
            'overfitting_ratio_r2': train_r2 / r2 if r2 > 0 else float('inf'),
        })
    
    # Özellik önem dereceleri
    if feature_names:
        feature_importance = extract_feature_importance(model, feature_names)
    else:
        feature_importance = extract_feature_importance(model)
    
    logger.info(f"Model değerlendirildi. Test MAPE: {mape:.2f}%, Test R²: {r2:.4f}")
    return metrics, feature_importance


def hyperparameter_tuning(X_train, y_train, X_val, y_val, model_name='rf', param_grid=None, 
                         n_iter=10, cv=5, scoring='neg_mean_absolute_percentage_error', 
                         search_method='random', random_state=42):
    """
    Hiperparametre optimizasyonu yapar.
    
    Args:
        X_train (numpy.ndarray): Eğitim verisi özellikleri
        y_train (numpy.ndarray): Eğitim verisi hedef değişkeni
        X_val (numpy.ndarray): Doğrulama verisi özellikleri
        y_val (numpy.ndarray): Doğrulama verisi hedef değişkeni
        model_name (str): Kullanılacak model ismi
        param_grid (dict): Parametre ızgarası
        n_iter (int): Rastgele arama için iterasyon sayısı
        cv (int): Çapraz doğrulama katlama sayısı
        scoring (str): Optimizasyon metriği
        search_method (str): Arama yöntemi ('grid' veya 'random')
        random_state (int): Rastgele sayı üreteci için tohum değeri
        
    Returns:
        object: En iyi model
        dict: En iyi parametreler
        float: En iyi skor
    """
    logger.info(f"{model_name.upper()} için hiperparametre optimizasyonu başlatılıyor. Yöntem: {search_method}")
    
    # Model sınıfını al
    ModelClass = get_model_class(model_name)
    
    # Parametre ızgarası belirlenmemişse varsayılanları kullan
    if param_grid is None:
        if model_name.lower() == 'dt':
            param_grid = {
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2', None]
            }
        elif model_name.lower() == 'rf':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2', None]
            }
        elif model_name.lower() == 'ridge':
            param_grid = {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
            }
        elif model_name.lower() == 'adaboost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.5, 1.0],
                'loss': ['linear', 'square', 'exponential']
            }
        else:  # MLR için parametre ızgarası yok
            param_grid = {}
    
    # MLR için parametre ızgarası boşsa doğrudan modeli eğit ve değerlendir
    if model_name.lower() == 'mlr' and (not param_grid or len(param_grid) == 0):
        model = ModelClass()
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        val_mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100
        logger.info(f"MLR modeli için hiperparametre optimizasyonu atlandı. Doğrulama MAPE: {val_mape:.2f}%")
        return model, {}, -val_mape  # neg_mean_absolute_percentage_error ile uyumlu olması için negatif
    
    # Arama yöntemine göre optimizasyon yap
    if search_method.lower() == 'grid':
        search = GridSearchCV(
            ModelClass(random_state=random_state) if 'random_state' in ModelClass().get_params() else ModelClass(),
            param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=1
        )
    else:  # random search
        search = RandomizedSearchCV(
            ModelClass(random_state=random_state) if 'random_state' in ModelClass().get_params() else ModelClass(),
            param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            random_state=random_state,
            verbose=1
        )
    
    # Arama
    search.fit(X_train, y_train)
    
    # En iyi model
    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_
    
    # Doğrulama verisi ile değerlendir
    y_val_pred = best_model.predict(X_val)
    val_mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100
    val_r2 = r2_score(y_val, y_val_pred)
    
    logger.info(f"Hiperparametre optimizasyonu tamamlandı. En iyi parametreler: {best_params}")
    logger.info(f"Doğrulama metrikler: MAPE: {val_mape:.2f}%, R²: {val_r2:.4f}")
    
    return best_model, best_params, best_score


def save_model(model, model_name, metrics=None, feature_importance=None, model_params=None):
    """
    Modeli ve ilgili bilgileri kaydeder.
    
    Args:
        model: Eğitilmiş model
        model_name (str): Model ismi
        metrics (dict, optional): Model metrikleri
        feature_importance (pandas.DataFrame, optional): Özellik önem dereceleri
        model_params (dict, optional): Model parametreleri
        
    Returns:
        str: Model dosyasının yolu
    """
    project_root = get_project_root()
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = os.path.join(models_dir, f"{model_name}_{timestamp}.pkl")
    
    try:
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Model kaydedildi: {model_file}")
        
        # Model sonuçlarını kaydet (varsa)
        if metrics:
            results_file = save_model_results(
                model_name=model_name,
                metrics=metrics,
                feature_importance=feature_importance,
                model_params=model_params
            )
            logger.info(f"Model sonuçları kaydedildi: {results_file}")
        
        return model_file
    
    except Exception as e:
        logger.error(f"Model kaydedilirken hata oluştu: {str(e)}")
        raise


def train_and_evaluate_all_models(X_train, y_train, X_test, y_test, feature_names=None, 
                                 do_hyperparameter_tuning=False, X_val=None, y_val=None, 
                                 save_models=True, random_state=42):
    """
    Tüm modelleri eğitir, değerlendirir ve sonuçları karşılaştırır.
    
    Args:
        X_train (numpy.ndarray): Eğitim verisi özellikleri
        y_train (numpy.ndarray): Eğitim verisi hedef değişkeni
        X_test (numpy.ndarray): Test verisi özellikleri
        y_test (numpy.ndarray): Test verisi hedef değişkeni
        feature_names (list, optional): Özellik isimleri
        do_hyperparameter_tuning (bool): Hiperparametre optimizasyonu yapılacak mı?
        X_val (numpy.ndarray, optional): Doğrulama verisi özellikleri
        y_val (numpy.ndarray, optional): Doğrulama verisi hedef değişkeni
        save_models (bool): Modeller kaydedilecek mi?
        random_state (int): Rastgele sayı üreteci için tohum değeri
        
    Returns:
        dict: Eğitilmiş modeller
        dict: Değerlendirme metrikleri
        pandas.DataFrame: Modellerin karşılaştırması
    """
    logger.info("Tüm modeller eğitiliyor ve değerlendiriliyor")
    
    # Hiperparametre optimizasyonu için doğrulama seti kontrolü
    if do_hyperparameter_tuning and (X_val is None or y_val is None):
        if X_val is None:
            logger.warning("Hiperparametre optimizasyonu için doğrulama seti (X_val) belirtilmedi")
        if y_val is None:
            logger.warning("Hiperparametre optimizasyonu için doğrulama seti (y_val) belirtilmedi")
        logger.warning("Hiperparametre optimizasyonu atlanıyor")
        do_hyperparameter_tuning = False
    
    # Model isimleri
    model_names = ['mlr', 'dt', 'rf', 'ridge', 'adaboost']
    
    # Sonuçları tutacak sözlükler
    models = {}
    all_metrics = {}
    all_feature_importances = {}
    
    # Her model için eğitim ve değerlendirme yap
    for model_name in model_names:
        logger.info(f"--- {model_name.upper()} modeli eğitiliyor ---")
        
        # Hiperparametre optimizasyonu
        if do_hyperparameter_tuning:
            logger.info(f"{model_name.upper()} için hiperparametre optimizasyonu yapılıyor")
            model, best_params, _ = hyperparameter_tuning(
                X_train, y_train, X_val, y_val, model_name=model_name, random_state=random_state
            )
        else:
            logger.info(f"{model_name.upper()} varsayılan parametrelerle eğitiliyor")
            model = train_model(X_train, y_train, model_name=model_name, random_state=random_state)
            best_params = model.get_params()
        
        # Değerlendirme
        metrics, feature_importance = evaluate_model(
            model, X_test, y_test, X_train, y_train, feature_names=feature_names
        )
        
        # Sonuçları kaydet
        models[model_name] = model
        all_metrics[model_name] = metrics
        all_feature_importances[model_name] = feature_importance
        
        # Modeli kaydet
        if save_models:
            save_model(
                model=model,
                model_name=model_name,
                metrics=metrics,
                feature_importance=feature_importance,
                model_params=best_params
            )
    
    # Karşılaştırma tablosu oluştur
    comparison = []
    for model_name in model_names:
        metrics = all_metrics[model_name]
        comparison.append({
            'Model': model_name.upper(),
            'Test MAPE (%)': metrics['test_mape'],
            'Test R²': metrics['test_r2'],
            'Test RMSE': metrics['test_rmse'],
            'Train MAPE (%)': metrics['train_mape'] if 'train_mape' in metrics else None,
            'Train R²': metrics['train_r2'] if 'train_r2' in metrics else None,
            'Overfitting Ratio (R²)': metrics['overfitting_ratio_r2'] if 'overfitting_ratio_r2' in metrics else None
        })
    
    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values('Test MAPE (%)', ascending=True)
    
    logger.info("Tüm modeller eğitildi ve değerlendirildi")
    logger.info(f"En iyi model: {comparison_df.iloc[0]['Model']}, MAPE: {comparison_df.iloc[0]['Test MAPE (%)']:.2f}%, R²: {comparison_df.iloc[0]['Test R²']:.4f}")
    
    return models, all_metrics, comparison_df


def load_trained_model(model_file):
    """
    Eğitilmiş modeli yükler.
    
    Args:
        model_file (str): Model dosyasının yolu
        
    Returns:
        object: Eğitilmiş model
    """
    try:
        logger.info(f"Model yükleniyor: {model_file}")
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        logger.info("Model başarıyla yüklendi")
        return model
    except Exception as e:
        logger.error(f"Model yüklenirken hata oluştu: {str(e)}")
        raise


if __name__ == "__main__":
    # Test işlevselliği
    try:
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        
        # Yapay veri oluştur
        X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
        
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Tek bir model eğit ve değerlendir
        model = train_model(X_train, y_train, model_name='rf')
        metrics, feature_importance = evaluate_model(model, X_test, y_test, X_train, y_train, feature_names)
        print(f"RF Model Metrikleri: {metrics}")
        print(f"Özellik Önem Dereceleri (Top 5):\n{feature_importance.head()}")
        
        # Hiperparametre optimizasyonu
        best_model, best_params, _ = hyperparameter_tuning(
            X_train, y_train, X_val, y_val, model_name='rf', n_iter=2, cv=2
        )
        print(f"En İyi Parametreler: {best_params}")
        
        # Tüm modelleri eğit ve karşılaştır
        models, all_metrics, comparison = train_and_evaluate_all_models(
            X_train, y_train, X_test, y_test, feature_names,
            do_hyperparameter_tuning=True, X_val=X_val, y_val=y_val,
            save_models=False, random_state=42
        )
        print(f"Model Karşılaştırması:\n{comparison}")
        
        print("Model eğitim modülü başarıyla test edildi.")
    except Exception as e:
        logger.error(f"Test sırasında hata oluştu: {str(e)}")