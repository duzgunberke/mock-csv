import pandas as pd
import numpy as np
import os
import sys
import logging
import pickle
from pathlib import Path
from sklearn.ensemble import VotingRegressor, StackingRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from datetime import datetime

# src klasörünü Python yoluna ekle
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(src_dir)

from src.data.data_loader import get_project_root
from src.models.train_model import get_model_class, train_model, save_model
from src.models.evaluate_model import calculate_metrics

# Loglama yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_voting_ensemble(models, model_weights=None):
    """
    Oylama tabanlı topluluk modeli oluşturur.
    
    Args:
        models (list): Model nesneleri listesi
        model_weights (list, optional): Model ağırlıkları
        
    Returns:
        VotingRegressor: Oluşturulan topluluk modeli
    """
    logger.info("Oylama tabanlı topluluk modeli oluşturuluyor")
    
    # Model isimlerini oluştur
    model_names = [f'model_{i}' for i in range(len(models))]
    
    # Modelleri ve isimlerini birleştir
    named_models = list(zip(model_names, models))
    
    # Ağırlıklar belirtilmişse kontrol et
    if model_weights is not None:
        if len(model_weights) != len(models):
            logger.warning(f"Model sayısı ({len(models)}) ve ağırlık sayısı ({len(model_weights)}) uyuşmuyor. Eşit ağırlıklar kullanılıyor.")
            model_weights = None
    
    # Topluluk modeli oluştur
    ensemble = VotingRegressor(estimators=named_models, weights=model_weights)
    
    logger.info(f"Oylama modeli oluşturuldu: {len(models)} model")
    return ensemble


def create_stacking_ensemble(base_models, meta_model=None):
    """
    Yığınlama tabanlı topluluk modeli oluşturur.
    
    Args:
        base_models (list): Temel model nesneleri listesi
        meta_model: Meta model (belirtilmezse LinearRegression kullanılır)
        
    Returns:
        StackingRegressor: Oluşturulan topluluk modeli
    """
    logger.info("Yığınlama tabanlı topluluk modeli oluşturuluyor")
    
    # Model isimlerini oluştur
    model_names = [f'model_{i}' for i in range(len(base_models))]
    
    # Modelleri ve isimlerini birleştir
    named_models = list(zip(model_names, base_models))
    
    # Meta model kontrol et
    if meta_model is None:
        meta_model = Ridge()
    
    # Topluluk modeli oluştur
    ensemble = StackingRegressor(estimators=named_models, final_estimator=meta_model)
    
    logger.info(f"Yığınlama modeli oluşturuldu: {len(base_models)} temel model, meta model: {type(meta_model).__name__}")
    return ensemble


def create_bagging_ensemble(base_estimator, n_estimators=10, random_state=42):
    """
    Torbalama tabanlı topluluk modeli oluşturur.
    
    Args:
        base_estimator: Temel model
        n_estimators (int): Tahminleyici sayısı
        random_state (int): Rastgele sayı üreteci için tohum değeri
        
    Returns:
        BaggingRegressor: Oluşturulan topluluk modeli
    """
    logger.info(f"Torbalama tabanlı topluluk modeli oluşturuluyor: {n_estimators} tahminleyici")
    
    # Topluluk modeli oluştur
    ensemble = BaggingRegressor(
        estimator=base_estimator,
        n_estimators=n_estimators,
        random_state=random_state
    )
    
    logger.info(f"Torbalama modeli oluşturuldu: {type(base_estimator).__name__} temel model, {n_estimators} tahminleyici")
    return ensemble


def create_boosting_ensemble(X_train, y_train, base_estimator='dt', n_estimators=50, 
                           learning_rate=1.0, random_state=42):
    """
    Güçlendirme tabanlı topluluk modeli oluşturur (AdaBoost).
    
    Args:
        X_train (numpy.ndarray): Eğitim verisi özellikleri
        y_train (numpy.ndarray): Eğitim verisi hedef değişkeni
        base_estimator (str or object): Temel model ismi veya nesnesi
        n_estimators (int): Tahminleyici sayısı
        learning_rate (float): Öğrenme oranı
        random_state (int): Rastgele sayı üreteci için tohum değeri
        
    Returns:
        AdaBoostRegressor: Oluşturulan topluluk modeli
    """
    logger.info(f"Güçlendirme tabanlı topluluk modeli oluşturuluyor: {n_estimators} tahminleyici")
    
    # Temel model ayarla
    if isinstance(base_estimator, str):
        # Model sınıfını al
        model_class = None
        if base_estimator.lower() == 'dt':
            model_class = DecisionTreeRegressor
        else:
            try:
                model_class = get_model_class(base_estimator)
            except ValueError:
                logger.warning(f"Bilinmeyen model ismi: {base_estimator}. DecisionTreeRegressor kullanılıyor.")
                model_class = DecisionTreeRegressor
        
        # Temel model oluştur (basit parametrelerle)
        base_estimator = model_class(max_depth=3, random_state=random_state)
    
    # Topluluk modeli oluştur
    ensemble = AdaBoostRegressor(
        estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state
    )
    
    # Modeli eğit
    ensemble.fit(X_train, y_train)
    
    logger.info(f"Güçlendirme modeli oluşturuldu: {type(base_estimator).__name__} temel model, {n_estimators} tahminleyici")
    return ensemble


def create_ensemble_from_trained_models(X_train, y_train, models_dict, ensemble_type='voting', 
                                      weights=None, meta_model=None):
    """
    Eğitilmiş modellerden topluluk modeli oluşturur.
    
    Args:
        X_train (numpy.ndarray): Eğitim verisi özellikleri
        y_train (numpy.ndarray): Eğitim verisi hedef değişkeni
        models_dict (dict): Eğitilmiş modeller sözlüğü {model_name: model}
        ensemble_type (str): Topluluk türü ('voting', 'stacking')
        weights (list, optional): Oylama modeli için ağırlıklar
        meta_model: Yığınlama modeli için meta model
        
    Returns:
        object: Oluşturulan ve eğitilmiş topluluk modeli
    """
    logger.info(f"{ensemble_type} türünde topluluk modeli oluşturuluyor")
    
    # Modelleri listeye dönüştür
    models = list(models_dict.values())
    model_names = list(models_dict.keys())
    
    if ensemble_type.lower() == 'voting':
        # Oylama modeli oluştur
        ensemble = create_voting_ensemble(models, weights)
    elif ensemble_type.lower() == 'stacking':
        # Yığınlama modeli oluştur
        ensemble = create_stacking_ensemble(models, meta_model)
    else:
        raise ValueError(f"Bilinmeyen topluluk türü: {ensemble_type}. Desteklenen türler: 'voting', 'stacking'")
    
    # Topluluk modelini eğit
    ensemble.fit(X_train, y_train)
    
    logger.info(f"Topluluk modeli oluşturuldu ve eğitildi")
    return ensemble


def train_all_ensembles(X_train, y_train, X_test, y_test, models_dict=None,
                      base_models=['mlr', 'dt', 'rf', 'ridge'], save_models_flag=True, random_state=42):
    """
    Tüm topluluk modellerini eğitir ve değerlendirir.
    
    Args:
        X_train (numpy.ndarray): Eğitim verisi özellikleri
        y_train (numpy.ndarray): Eğitim verisi hedef değişkeni
        X_test (numpy.ndarray): Test verisi özellikleri
        y_test (numpy.ndarray): Test verisi hedef değişkeni
        models_dict (dict, optional): Eğitilmiş modeller sözlüğü. Verilmezse yeni modeller eğitilir.
        base_models (list): Temel model isimleri
        save_models_flag (bool): Modeller kaydedilecek mi?
        random_state (int): Rastgele sayı üreteci için tohum değeri
        
    Returns:
        dict: Eğitilmiş topluluk modelleri
        dict: Değerlendirme metrikleri
        pandas.DataFrame: Modellerin karşılaştırması
    """
    logger.info("Tüm topluluk modelleri eğitiliyor ve değerlendiriliyor")
    
    # Eğitilmiş modeller verilmemişse yeni modeller eğit
    if models_dict is None:
        logger.info("Temel modeller eğitiliyor")
        models_dict = {}
        
        for model_name in base_models:
            model = train_model(X_train, y_train, model_name=model_name, random_state=random_state)
            models_dict[model_name] = model
    
    # Topluluk modelleri
    ensembles = {}
    all_metrics = {}
    
    # 1. Oylama tabanlı topluluk (eşit ağırlıklı)
    voting_ensemble = create_ensemble_from_trained_models(
        X_train, y_train, models_dict, ensemble_type='voting'
    )
    ensembles['voting'] = voting_ensemble
    
    # 2. Oylama tabanlı topluluk (ağırlıklı)
    # Ağırlıkları belirlemek için test verisindeki başarıya göre hesapla
    weights = []
    for model_name, model in models_dict.items():
        y_pred = model.predict(X_test)
        r2 = calculate_metrics(y_test, y_pred)['r2']
        weights.append(max(0.1, r2))  # Minimum 0.1 ağırlık
    
    # Ağırlıkları normalize et
    weights = np.array(weights) / sum(weights)
    
    weighted_voting_ensemble = create_ensemble_from_trained_models(
        X_train, y_train, models_dict, ensemble_type='voting', weights=weights
    )
    ensembles['weighted_voting'] = weighted_voting_ensemble
    
    # 3. Yığınlama tabanlı topluluk (meta model: Ridge)
    stacking_ensemble = create_ensemble_from_trained_models(
        X_train, y_train, models_dict, ensemble_type='stacking', meta_model=Ridge()
    )
    ensembles['stacking'] = stacking_ensemble
    
    # 4. Torbalama tabanlı topluluk (DecisionTree tabanlı)
    dt_base = DecisionTreeRegressor(random_state=random_state)
    bagging_ensemble = create_bagging_ensemble(
        dt_base, n_estimators=50, random_state=random_state
    )
    bagging_ensemble.fit(X_train, y_train)
    ensembles['bagging'] = bagging_ensemble
    
    # 5. Güçlendirme tabanlı topluluk (AdaBoost)
    boosting_ensemble = create_boosting_ensemble(
        X_train, y_train, base_estimator='dt', n_estimators=50, random_state=random_state
    )
    ensembles['boosting'] = boosting_ensemble
    
    # Tüm modelleri değerlendir
    logger.info("Topluluk modelleri değerlendiriliyor")
    
    for model_name, model in ensembles.items():
        # Test tahminleri
        y_pred = model.predict(X_test)
        
        # Metrikleri hesapla
        metrics = calculate_metrics(y_test, y_pred)
        all_metrics[model_name] = metrics
        
        logger.info(f"{model_name} modeli değerlendirildi: MAPE={metrics['mape']:.2f}%, R²={metrics['r2']:.4f}")
        
        # Modeli kaydet
        if save_models_flag:
            save_model(model, f"ensemble_{model_name}")
    
    # Karşılaştırma tablosu oluştur
    comparison = []
    
    for model_name, metrics in all_metrics.items():
        comparison.append({
            'Model': f"Ensemble_{model_name}",
            'MAPE (%)': metrics['mape'],
            'R²': metrics['r2'],
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae']
        })
    
    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values('MAPE (%)', ascending=True)
    
    logger.info("Tüm topluluk modelleri eğitildi ve değerlendirildi")
    logger.info(f"En iyi topluluk modeli: {comparison_df.iloc[0]['Model']}, MAPE: {comparison_df.iloc[0]['MAPE (%)']:.2f}%, R²: {comparison_df.iloc[0]['R²']:.4f}")
    
    return ensembles, all_metrics, comparison_df


def get_best_ensemble_model(ensembles, metrics, criteria='mape'):
    """
    En iyi topluluk modelini seçer.
    
    Args:
        ensembles (dict): Eğitilmiş topluluk modelleri
        metrics (dict): Değerlendirme metrikleri
        criteria (str): Seçim kriteri ('mape', 'r2', 'rmse')
        
    Returns:
        tuple: (en_iyi_model_adı, en_iyi_model, en_iyi_metrik_değeri)
    """
    logger.info(f"En iyi topluluk modeli seçiliyor. Kriter: {criteria}")
    
    if criteria.lower() == 'mape':
        # En düşük MAPE
        best_model_name = min(metrics, key=lambda k: metrics[k]['mape'])
        best_value = metrics[best_model_name]['mape']
    elif criteria.lower() == 'r2':
        # En yüksek R²
        best_model_name = max(metrics, key=lambda k: metrics[k]['r2'])
        best_value = metrics[best_model_name]['r2']
    elif criteria.lower() == 'rmse':
        # En düşük RMSE
        best_model_name = min(metrics, key=lambda k: metrics[k]['rmse'])
        best_value = metrics[best_model_name]['rmse']
    else:
        raise ValueError(f"Bilinmeyen kriter: {criteria}. Desteklenen kriterler: 'mape', 'r2', 'rmse'")
    
    best_model = ensembles[best_model_name]
    
    logger.info(f"En iyi topluluk modeli seçildi: {best_model_name}, {criteria}={best_value:.4f}")
    return best_model_name, best_model, best_value


def save_ensemble_model(ensemble, model_name):
    """
    Topluluk modelini kaydeder.
    
    Args:
        ensemble: Eğitilmiş topluluk modeli
        model_name (str): Model adı
        
    Returns:
        str: Kaydedilen dosyanın yolu
    """
    project_root = get_project_root()
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = os.path.join(models_dir, f"ensemble_{model_name}_{timestamp}.pkl")
    
    try:
        with open(model_file, 'wb') as f:
            pickle.dump(ensemble, f)
        
        logger.info(f"Topluluk modeli kaydedildi: {model_file}")
        return model_file
    
    except Exception as e:
        logger.error(f"Topluluk modeli kaydedilirken hata oluştu: {str(e)}")
        raise


if __name__ == "__main__":
    # Test işlevselliği
    try:
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        
        # Yapay veri oluştur
        X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Temel modeller eğit
        mlr = LinearRegression().fit(X_train, y_train)
        rf = RandomForestRegressor(n_estimators=10, random_state=42).fit(X_train, y_train)
        ridge = Ridge(alpha=1.0).fit(X_train, y_train)
        
        models_dict = {
            'mlr': mlr,
            'rf': rf,
            'ridge': ridge
        }
        
        # Oylama modelini test et
        voting_ensemble = create_ensemble_from_trained_models(
            X_train, y_train, models_dict, ensemble_type='voting'
        )
        
        y_pred = voting_ensemble.predict(X_test)
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"Oylama modeli MSE: {mse:.4f}")
        
        # Yığınlama modelini test et
        stacking_ensemble = create_ensemble_from_trained_models(
            X_train, y_train, models_dict, ensemble_type='stacking'
        )
        
        y_pred = stacking_ensemble.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"Yığınlama modeli MSE: {mse:.4f}")
        
        # Tüm topluluk modellerini test et
        ensembles, metrics, comparison = train_all_ensembles(
            X_train, y_train, X_test, y_test, models_dict, save_models_flag=False
        )
        
        print(f"Topluluk modelleri karşılaştırması:")
        print(comparison)
        
        # En iyi modeli seç
        best_name, best_model, best_value = get_best_ensemble_model(ensembles, metrics)
        
        print(f"En iyi topluluk modeli: {best_name}, MAPE: {best_value:.4f}")
        
        print("Topluluk modeli modülü başarıyla test edildi.")
    except Exception as e:
        logger.error(f"Test sırasında hata oluştu: {str(e)}")