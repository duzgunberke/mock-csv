import pandas as pd
import numpy as np
import os
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import pickle
import json
from datetime import datetime

# src klasörünü Python yoluna ekle
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(src_dir)

from src.data.data_loader import get_project_root

# Loglama yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_metrics(y_true, y_pred):
    """
    Regresyon metriklerini hesaplar.
    
    Args:
        y_true (numpy.ndarray): Gerçek değerler
        y_pred (numpy.ndarray): Tahmin edilen değerler
        
    Returns:
        dict: Hesaplanan metrikler
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # Yüzde olarak
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }
    
    return metrics


def evaluate_model_performance(model, X_test, y_test, feature_names=None, output_dir=None):
    """
    Modelin performansını değerlendirir ve görselleştirir.
    
    Args:
        model: Eğitilmiş model
        X_test (numpy.ndarray): Test verisi özellikleri
        y_test (numpy.ndarray): Test verisi hedef değişkeni
        feature_names (list, optional): Özellik isimleri
        output_dir (str, optional): Çıktı dizini
        
    Returns:
        dict: Hesaplanan metrikler
        dict: Oluşturulan görselleştirmeler bilgileri
    """
    logger.info("Model performansı değerlendiriliyor")
    
    # Test verisi üzerinde tahmin yap
    y_pred = model.predict(X_test)
    
    # Metrikleri hesapla
    metrics = calculate_metrics(y_test, y_pred)
    logger.info(f"Hesaplanan metrikler: RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.2f}%, R²={metrics['r2']:.4f}")
    
    # Çıktı dizinini belirle
    if output_dir is None:
        project_root = get_project_root()
        output_dir = os.path.join(project_root, 'reports', 'figures')
    
    # Klasörün var olduğundan emin ol
    os.makedirs(output_dir, exist_ok=True)
    
    # Görselleştirme bilgilerini tut
    visualizations = {}
    
    # 1. Gerçek vs Tahmin grafiği
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Gerçek Maaşlar (TL)')
    plt.ylabel('Tahmin Edilen Maaşlar (TL)')
    plt.title('Gerçek vs Tahmin Edilen Maaşlar')
    
    # Eşit eksen ölçekleri
    plt.axis('equal')
    plt.tight_layout()
    
    # Kaydet
    actual_vs_predicted_path = os.path.join(output_dir, 'actual_vs_predicted.png')
    plt.savefig(actual_vs_predicted_path)
    plt.close()
    
    visualizations['actual_vs_predicted'] = actual_vs_predicted_path
    
    # 2. Rezidüel grafiği
    residuals = y_test - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Tahmin Edilen Maaşlar (TL)')
    plt.ylabel('Rezidüeller (TL)')
    plt.title('Rezidüel Analizi')
    plt.tight_layout()
    
    # Kaydet
    residuals_path = os.path.join(output_dir, 'residuals.png')
    plt.savefig(residuals_path)
    plt.close()
    
    visualizations['residuals'] = residuals_path
    
    # 3. Rezidüel dağılımı
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Rezidüeller (TL)')
    plt.ylabel('Frekans')
    plt.title('Rezidüellerin Dağılımı')
    plt.tight_layout()
    
    # Kaydet
    residual_dist_path = os.path.join(output_dir, 'residual_distribution.png')
    plt.savefig(residual_dist_path)
    plt.close()
    
    visualizations['residual_distribution'] = residual_dist_path
    
    # 4. Özellik Önem Dereceleri (model destekliyorsa)
    if feature_names is not None and (hasattr(model, 'feature_importances_') or hasattr(model, 'coef_')):
        # Özellik önem derecelerini al
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)  # Katsayıların mutlak değeri
        
        # Özellik isimlerinin uzunluğu kontrol et
        if len(feature_names) == len(importances):
            # Önem derecelerine göre sırala
            indices = np.argsort(importances)[-15:]  # En önemli 15 özellik
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Önem Derecesi')
            plt.title('Özellik Önem Dereceleri (Top 15)')
            plt.tight_layout()
            
            # Kaydet
            feature_imp_path = os.path.join(output_dir, 'feature_importance.png')
            plt.savefig(feature_imp_path)
            plt.close()
            
            visualizations['feature_importance'] = feature_imp_path
    
    # 5. MAPE grupları (gelir seviyesine göre)
    # Gelir gruplarına göre MAPE hesapla
    income_groups = pd.cut(y_test, bins=5)
    income_mape = pd.DataFrame({
        'Gerçek_Maaş': y_test,
        'Tahmin': y_pred,
        'Gelir_Grubu': income_groups
    })
    
    group_mape = income_mape.groupby('Gelir_Grubu').apply(
        lambda x: mean_absolute_percentage_error(x['Gerçek_Maaş'], x['Tahmin']) * 100
    ).reset_index(name='MAPE')
    
    plt.figure(figsize=(10, 6))
    plt.bar(group_mape.index, group_mape['MAPE'])
    plt.xlabel('Gelir Grubu')
    plt.ylabel('MAPE (%)')
    plt.title('Gelir Gruplarına Göre MAPE')
    plt.xticks(group_mape.index, [str(x) for x in group_mape['Gelir_Grubu']], rotation=45)
    plt.tight_layout()
    
    # Kaydet
    income_mape_path = os.path.join(output_dir, 'income_group_mape.png')
    plt.savefig(income_mape_path)
    plt.close()
    
    visualizations['income_group_mape'] = income_mape_path
    
    logger.info("Model performans değerlendirmesi tamamlandı")
    return metrics, visualizations


def evaluate_feature_importance(model, feature_names, output_dir=None):
    """
    Özellik önem derecelerini değerlendirir ve görselleştirir.
    
    Args:
        model: Eğitilmiş model
        feature_names (list): Özellik isimleri
        output_dir (str, optional): Çıktı dizini
        
    Returns:
        pandas.DataFrame: Özellik önem dereceleri
        str: Oluşturulan grafiğin yolu
    """
    logger.info("Özellik önem dereceleri değerlendiriliyor")
    
    # Modelden özellik önem derecelerini al
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importance_type = 'feature_importances'
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
        importance_type = 'coefficients'
    else:
        logger.warning("Model özellik önem derecelerini desteklemiyor")
        return None, None
    
    # Özellik isimlerinin uzunluğunu kontrol et
    if len(feature_names) != len(importances):
        logger.warning(f"Özellik isimleri ({len(feature_names)}) ve önem dereceleri ({len(importances)}) boyutları uyumsuz")
        feature_names = [f"feature_{i}" for i in range(len(importances))]
    
    # Özellik önem derecelerini DataFrame'e dönüştür
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Önem derecesine göre sırala
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Çıktı dizinini belirle
    if output_dir is None:
        project_root = get_project_root()
        output_dir = os.path.join(project_root, 'reports', 'figures')
    
    # Klasörün var olduğundan emin ol
    os.makedirs(output_dir, exist_ok=True)
    
    # Görselleştir
    plt.figure(figsize=(12, 10))
    
    # En önemli 20 özellik
    top_n = min(20, len(feature_importance))
    top_features = feature_importance.head(top_n)
    
    plt.barh(range(top_n), top_features['Importance'])
    plt.yticks(range(top_n), top_features['Feature'])
    
    if importance_type == 'feature_importances':
        plt.xlabel('Özellik Önem Derecesi')
        plt.title('Özellik Önem Dereceleri (Top 20)')
    else:
        plt.xlabel('Katsayı Büyüklüğü (Mutlak Değer)')
        plt.title('Özellik Katsayıları (Top 20, Mutlak Değer)')
    
    plt.tight_layout()
    
    # Kaydet
    feature_imp_path = os.path.join(output_dir, 'detailed_feature_importance.png')
    plt.savefig(feature_imp_path)
    plt.close()
    
    logger.info(f"Özellik önem dereceleri değerlendirildi. En önemli özellik: {feature_importance.iloc[0]['Feature']}")
    return feature_importance, feature_imp_path


def compare_models(model_metrics, output_dir=None):
    """
    Farklı modellerin performansını karşılaştırır ve görselleştirir.
    
    Args:
        model_metrics (dict): Model metriklerini içeren sözlük {model_name: metrics}
        output_dir (str, optional): Çıktı dizini
        
    Returns:
        pandas.DataFrame: Karşılaştırma tablosu
        dict: Oluşturulan görselleştirmeler bilgileri
    """
    logger.info("Model karşılaştırması yapılıyor")
    
    # Karşılaştırma verilerini hazırla
    comparison_data = []
    
    for model_name, metrics in model_metrics.items():
        comparison_data.append({
            'Model': model_name,
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae'],
            'MAPE (%)': metrics['mape'],
            'R²': metrics['r2']
        })
    
    # DataFrame oluştur
    comparison_df = pd.DataFrame(comparison_data)
    
    # MAPE'ye göre sırala
    comparison_df = comparison_df.sort_values('MAPE (%)', ascending=True)
    
    # Çıktı dizinini belirle
    if output_dir is None:
        project_root = get_project_root()
        output_dir = os.path.join(project_root, 'reports', 'figures')
    
    # Klasörün var olduğundan emin ol
    os.makedirs(output_dir, exist_ok=True)
    
    # Görselleştirme bilgilerini tut
    visualizations = {}
    
    # 1. MAPE Karşılaştırması
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Model', y='MAPE (%)', data=comparison_df)
    plt.title('Model Karşılaştırması - MAPE (%)')
    plt.ylabel('MAPE (%)')
    plt.xticks(rotation=45)
    
    # Değerleri çubukların üzerine yaz
    for i, v in enumerate(comparison_df['MAPE (%)']):
        ax.text(i, v + 0.5, f"{v:.2f}%", ha='center')
    
    plt.tight_layout()
    
    # Kaydet
    mape_comparison_path = os.path.join(output_dir, 'model_comparison_mape.png')
    plt.savefig(mape_comparison_path)
    plt.close()
    
    visualizations['mape_comparison'] = mape_comparison_path
    
    # 2. R² Karşılaştırması
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Model', y='R²', data=comparison_df)
    plt.title('Model Karşılaştırması - R²')
    plt.ylabel('R²')
    plt.xticks(rotation=45)
    
    # Değerleri çubukların üzerine yaz
    for i, v in enumerate(comparison_df['R²']):
        ax.text(i, v - 0.05, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    
    # Kaydet
    r2_comparison_path = os.path.join(output_dir, 'model_comparison_r2.png')
    plt.savefig(r2_comparison_path)
    plt.close()
    
    visualizations['r2_comparison'] = r2_comparison_path
    
    logger.info("Model karşılaştırması tamamlandı")
    return comparison_df, visualizations


def analyze_error_distribution(y_true, y_pred, feature_data=None, output_dir=None):
    """
    Hata dağılımını analiz eder ve görselleştirir.
    
    Args:
        y_true (numpy.ndarray): Gerçek değerler
        y_pred (numpy.ndarray): Tahmin edilen değerler
        feature_data (pandas.DataFrame, optional): Özellik verisi
        output_dir (str, optional): Çıktı dizini
        
    Returns:
        dict: Hata analizi sonuçları
        dict: Oluşturulan görselleştirmeler bilgileri
    """
    logger.info("Hata dağılımı analiz ediliyor")
    
    # Hataları hesapla
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    percentage_errors = np.abs(errors / y_true) * 100
    
    # Hata istatistiklerini hesapla
    error_stats = {
        'mean_error': errors.mean(),
        'median_error': np.median(errors),
        'std_error': errors.std(),
        'mean_abs_error': abs_errors.mean(),
        'median_abs_error': np.median(abs_errors),
        'max_abs_error': abs_errors.max(),
        'mean_percentage_error': percentage_errors.mean(),
        'median_percentage_error': np.median(percentage_errors),
        'max_percentage_error': percentage_errors.max(),
        'error_within_5_percent': (percentage_errors <= 5).mean() * 100,
        'error_within_10_percent': (percentage_errors <= 10).mean() * 100,
        'error_within_20_percent': (percentage_errors <= 20).mean() * 100
    }
    
    # Çıktı dizinini belirle
    if output_dir is None:
        project_root = get_project_root()
        output_dir = os.path.join(project_root, 'reports', 'figures')
    
    # Klasörün var olduğundan emin ol
    os.makedirs(output_dir, exist_ok=True)
    
    # Görselleştirme bilgilerini tut
    visualizations = {}
    
    # 1. Hata Dağılımı Histogramı
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Hata (TL)')
    plt.ylabel('Frekans')
    plt.title('Hata Dağılımı')
    plt.tight_layout()
    
    # Kaydet
    error_dist_path = os.path.join(output_dir, 'error_distribution.png')
    plt.savefig(error_dist_path)
    plt.close()
    
    visualizations['error_distribution'] = error_dist_path
    
    # 2. Yüzdelik Hata Dağılımı
    plt.figure(figsize=(10, 6))
    sns.histplot(percentage_errors, kde=True)
    plt.axvline(x=5, color='g', linestyle='--', label='%5 Hata')
    plt.axvline(x=10, color='y', linestyle='--', label='%10 Hata')
    plt.axvline(x=20, color='r', linestyle='--', label='%20 Hata')
    plt.xlabel('Yüzdelik Hata (%)')
    plt.ylabel('Frekans')
    plt.title('Yüzdelik Hata Dağılımı')
    plt.legend()
    plt.tight_layout()
    
    # Kaydet
    pct_error_path = os.path.join(output_dir, 'percentage_error_distribution.png')
    plt.savefig(pct_error_path)
    plt.close()
    
    visualizations['percentage_error_distribution'] = pct_error_path
    
    # 3. Özelliklerle ilişkili hata analizi (isteğe bağlı)
    if feature_data is not None:
        # En önemli 5 sayısal özelliği seç
        numeric_features = feature_data.select_dtypes(include=['int64', 'float64']).columns[:5]
        
        for feature in numeric_features:
            plt.figure(figsize=(10, 6))
            plt.scatter(feature_data[feature], percentage_errors, alpha=0.5)
            plt.xlabel(feature)
            plt.ylabel('Yüzdelik Hata (%)')
            plt.title(f'{feature} vs Yüzdelik Hata')
            plt.tight_layout()
            
            # Kaydet
            feature_error_path = os.path.join(output_dir, f'error_vs_{feature}.png')
            plt.savefig(feature_error_path)
            plt.close()
            
            visualizations[f'error_vs_{feature}'] = feature_error_path
    
    logger.info(f"Hata dağılımı analizi tamamlandı. Ortalama hata: {error_stats['mean_error']:.2f} TL")
    return error_stats, visualizations


def create_evaluation_report(metrics, visualizations, model_name, output_dir=None):
    """
    Değerlendirme raporu oluşturur.
    
    Args:
        metrics (dict): Hesaplanan metrikler
        visualizations (dict): Görselleştirme bilgileri
        model_name (str): Model adı
        output_dir (str, optional): Çıktı dizini
        
    Returns:
        str: Oluşturulan rapor dosyasının yolu
    """
    logger.info(f"{model_name} modeli için değerlendirme raporu oluşturuluyor")
    
    # Çıktı dizinini belirle
    if output_dir is None:
        project_root = get_project_root()
        output_dir = os.path.join(project_root, 'reports')
    
    # Klasörün var olduğundan emin ol
    os.makedirs(output_dir, exist_ok=True)
    
    # Rapor içeriğini hazırla
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = {
        'model_name': model_name,
        'evaluation_time': timestamp,
        'metrics': metrics,
        'visualizations': visualizations
    }
    
    # Rapor dosyasını oluştur
    report_file = os.path.join(output_dir, f"{model_name}_evaluation_report.json")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    
    logger.info(f"Değerlendirme raporu oluşturuldu: {report_file}")
    return report_file


if __name__ == "__main__":
    # Test işlevselliği
    try:
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        
        # Yapay veri oluştur
        X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Test için özellik isimleri
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Model eğit
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Model performansını değerlendir
        metrics, visualizations = evaluate_model_performance(
            model, X_test, y_test, feature_names=feature_names
        )
        
        print(f"Hesaplanan metrikler: {metrics}")
        print(f"Oluşturulan görselleştirmeler: {list(visualizations.keys())}")
        
        # Özellik önem derecelerini değerlendir
        feature_importance, _ = evaluate_feature_importance(model, feature_names)
        
        print(f"En önemli 3 özellik:")
        print(feature_importance.head(3))
        
        # Hata dağılımını analiz et
        y_pred = model.predict(X_test)
        error_stats, _ = analyze_error_distribution(y_test, y_pred)

        print(f"Hata istatistikleri: {error_stats}")
        print("Model değerlendirme modülü başarıyla test edildi.")
    except Exception as e:
        logger.error(f"Test sırasında hata oluştu: {str(e)}")