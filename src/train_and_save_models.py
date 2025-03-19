"""
IT sektörü maaş tahmin modellerini eğitip kaydeden script.
Bu script, veri setini yükleyip ön işleme yapar, 
çeşitli modelleri eğitir ve en iyi modeli kaydeder.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# src klasörünü Python yoluna ekle
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)
sys.path.append(src_dir)

from src.data.data_loader import get_project_root, load_it_salary_data
from src.data.preprocess import prepare_it_salary_data, prepare_data_for_modeling
from src.models.train_model import train_and_evaluate_all_models
from src.models.evaluate_model import compare_models
from src.models.ensemble_models import train_all_ensembles, get_best_ensemble_model, save_ensemble_model

# Loglama yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_models(data_path=None, save_dir=None, random_state=42):
    """
    Maaş tahmin modellerini eğitir ve kaydeder.
    
    Args:
        data_path (str, optional): Veri setinin yolu
        save_dir (str, optional): Modellerin kaydedileceği dizin
        random_state (int): Rastgele sayı üreteci için tohum değeri
        
    Returns:
        str: En iyi model dosyasının yolu
    """
    # Veri setini yükle
    if data_path is None:
        logger.info("Varsayılan veri seti yükleniyor...")
        df = load_it_salary_data()
    else:
        logger.info(f"Veri seti yükleniyor: {data_path}")
        df = pd.read_csv(data_path)
    
    logger.info(f"Veri seti yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
    
    # Veriyi ön işle
    logger.info("Veri ön işleniyor...")
    processed_df, column_types = prepare_it_salary_data(df, target_column='Maaş_TL')
    logger.info(f"Ön işleme tamamlandı: {processed_df.shape[0]} satır, {processed_df.shape[1]} sütun")
    
    # Eğitim/test verilerini ayır
    logger.info("Eğitim ve test verileri ayrılıyor...")
    X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names = prepare_data_for_modeling(
        processed_df, target_column='Maaş_TL', test_size=0.2, random_state=random_state
    )
    logger.info(f"Veri ayrıştırma tamamlandı. Eğitim: {X_train_processed.shape}, Test: {X_test_processed.shape}")
    
    # Önişleyiciyi kaydet
    logger.info("Önişleyici kaydediliyor...")
    project_root = get_project_root()
    prep_dir = os.path.join(project_root, 'models', 'preprocessing')
    os.makedirs(prep_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    preprocessor_file = os.path.join(prep_dir, f"salary_preprocessor_{timestamp}.pkl")
    
    import pickle
    with open(preprocessor_file, 'wb') as f:
        pickle.dump(preprocessor, f)
    logger.info(f"Önişleyici kaydedildi: {preprocessor_file}")
    
    # Temel modelleri eğit
    logger.info("Temel modeller eğitiliyor...")
    models, metrics, comparison_df = train_and_evaluate_all_models(
        X_train_processed, y_train, X_test_processed, y_test, 
        feature_names=feature_names, save_models=True, random_state=random_state
    )
    
    # En iyi temel modeli logla
    best_model_name = comparison_df.iloc[0]['Model']
    best_mape = comparison_df.iloc[0]['MAPE (%)']
    best_r2 = comparison_df.iloc[0]['R²']
    logger.info(f"En iyi temel model: {best_model_name}, MAPE: {best_mape:.2f}%, R²: {best_r2:.4f}")
    
    # Topluluk modellerini eğit
    logger.info("Topluluk modelleri eğitiliyor...")
    ensembles, ensemble_metrics, ensemble_comparison = train_all_ensembles(
        X_train_processed, y_train, X_test_processed, y_test, 
        models_dict=models, save_models_flag=True, random_state=random_state
    )
    
    # En iyi topluluk modelini bul
    logger.info("En iyi topluluk modeli seçiliyor...")
    best_ensemble_name, best_ensemble, best_value = get_best_ensemble_model(ensembles, ensemble_metrics, criteria='mape')
    logger.info(f"En iyi topluluk modeli: {best_ensemble_name}, MAPE: {best_value:.2f}%")
    
    # En iyi modeli seç (temel modeller ve topluluk modelleri arasından)
    all_metrics = metrics.copy()
    all_metrics.update(ensemble_metrics)
    
    best_overall_model_name = min(all_metrics, key=lambda k: all_metrics[k]['mape'])
    best_overall_mape = all_metrics[best_overall_model_name]['mape']
    best_overall_r2 = all_metrics[best_overall_model_name]['r2']
    
    logger.info(f"En iyi genel model: {best_overall_model_name}, MAPE: {best_overall_mape:.2f}%, R²: {best_overall_r2:.4f}")
    
    # En iyi modeli ayrıca kaydet
    if best_overall_model_name in models:
        best_overall_model = models[best_overall_model_name]
        model_type = "base"
    else:
        best_overall_model = ensembles[best_overall_model_name]
        model_type = "ensemble"
    
    # En iyi modeli final model olarak kaydet
    logger.info("En iyi model 'best_salary_model.pkl' olarak kaydediliyor...")
    best_model_file = os.path.join(project_root, 'models', 'best_salary_model.pkl')
    with open(best_model_file, 'wb') as f:
        pickle.dump(best_overall_model, f)
    
    # Model bilgilerini JSON olarak kaydet
    import json
    model_info = {
        'model_name': best_overall_model_name,
        'model_type': model_type,
        'metrics': {
            'mape': best_overall_mape,
            'r2': best_overall_r2,
            'rmse': all_metrics[best_overall_model_name]['rmse'],
            'mae': all_metrics[best_overall_model_name]['mae']
        },
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'feature_count': X_train_processed.shape[1],
        'training_samples': X_train_processed.shape[0],
        'random_state': random_state
    }
    
    best_model_info_file = os.path.join(project_root, 'models', 'best_model_info.json')
    with open(best_model_info_file, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=4)
    
    logger.info(f"En iyi model bilgileri kaydedildi: {best_model_info_file}")
    logger.info("Model eğitimi ve kaydetme işlemi tamamlandı!")
    
    return best_model_file


if __name__ == "__main__":
    try:
        # Veri seti yolu çalıştırma argümanı olarak verilebilir
        data_path = sys.argv[1] if len(sys.argv) > 1 else None
        
        # Modelleri eğit ve kaydet
        best_model = train_models(data_path=data_path)
        print(f"En iyi model kaydedildi: {best_model}")
        print("Maaş tahmin modelleri başarıyla eğitildi ve kaydedildi!")
    except Exception as e:
        logger.error(f"Hata oluştu: {str(e)}", exc_info=True)
        print(f"Hata: {str(e)}")
        sys.exit(1)