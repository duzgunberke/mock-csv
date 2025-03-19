import pandas as pd
import numpy as np
import os
import pickle
import logging
from pathlib import Path
import sys
import json
from datetime import datetime

# src klasörünü Python yoluna ekle
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(src_dir)

from src.data.data_loader import get_project_root, load_test_employee
from src.data.preprocess import prepare_new_data
from src.models.evaluate_model import calculate_metrics

# Loglama yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path=None, model_name='rf'):
    """
    Eğitilmiş modeli yükler.
    
    Args:
        model_path (str, optional): Model dosyasının yolu
        model_name (str, optional): Model ismi (model_path belirtilmezse en son model yüklenir)
        
    Returns:
        object: Eğitilmiş model
    """
    if model_path is None:
        # En son modeli bul
        project_root = get_project_root()
        models_dir = os.path.join(project_root, 'models')
        
        if not os.path.exists(models_dir):
            logger.warning(f"Models dizini bulunamadı: {models_dir}. Varsayılan model oluşturuluyor.")
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=100, random_state=42)
        
        # model_name ile başlayan tüm dosyaları bul
        model_files = [f for f in os.listdir(models_dir) if f.startswith(f"{model_name}_") and f.endswith(".pkl")]
        
        if not model_files:
            logger.warning(f"{model_name} modeli bulunamadı. Varsayılan model oluşturuluyor.")
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Dosyaları tarihe göre sırala (en yeni en sonda)
        model_files.sort()
        model_path = os.path.join(models_dir, model_files[-1])
    
    try:
        logger.info(f"Model yükleniyor: {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info("Model başarıyla yüklendi")
        return model
    except Exception as e:
        logger.error(f"Model yüklenirken hata oluştu: {str(e)}")
        # Hata durumunda varsayılan bir model döndür
        logger.warning("Varsayılan model oluşturuluyor")
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=100, random_state=42)


def load_preprocessor(preprocessor_path=None):
    """
    Eğitilmiş önişleyiciyi yükler.
    
    Args:
        preprocessor_path (str, optional): Önişleyici dosyasının yolu
        
    Returns:
        object: Eğitilmiş önişleyici
    """
    if preprocessor_path is None:
        # En son önişleyiciyi bul
        project_root = get_project_root()
        preprocessing_dir = os.path.join(project_root, 'models', 'preprocessing')
        
        if not os.path.exists(preprocessing_dir):
            os.makedirs(preprocessing_dir, exist_ok=True)
            logger.warning(f"Preprocessing dizini bulunamadı: {preprocessing_dir}. Varsayılan önişleyici oluşturuluyor.")
            from sklearn.preprocessing import StandardScaler
            return StandardScaler()
        
        # preprocessor ile biten tüm dosyaları bul
        preprocessor_files = [f for f in os.listdir(preprocessing_dir) if f.endswith("_preprocessor.pkl")]
        
        if not preprocessor_files:
            logger.warning(f"Önişleyici bulunamadı. Varsayılan önişleyici oluşturuluyor.")
            from sklearn.preprocessing import StandardScaler
            return StandardScaler()
        
        # Dosyaları tarihe göre sırala (en yeni en sonda)
        preprocessor_files.sort()
        preprocessor_path = os.path.join(preprocessing_dir, preprocessor_files[-1])
    
    try:
        logger.info(f"Önişleyici yükleniyor: {preprocessor_path}")
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        logger.info("Önişleyici başarıyla yüklendi")
        return preprocessor
    except Exception as e:
        logger.error(f"Önişleyici yüklenirken hata oluştu: {str(e)}")
        # Hata durumunda varsayılan bir önişleyici döndür
        logger.warning("Varsayılan önişleyici oluşturuluyor")
        from sklearn.preprocessing import StandardScaler
        return StandardScaler()


def predict_salary(employee_data, model=None, preprocessor=None, return_features=False):
    """
    Çalışanın maaşını tahmin eder.
    
    Args:
        employee_data (pandas.DataFrame): Çalışan verisi
        model (object, optional): Eğitilmiş model
        preprocessor (object, optional): Eğitilmiş önişleyici
        return_features (bool): Özellik önem dereceleri de döndürülsün mü?
        
    Returns:
        float: Tahmin edilen maaş
        dict: İşlenmiş özellikler ve önem dereceleri (return_features=True ise)
    """
    logger.info("Maaş tahmini yapılıyor")
    
    # Model ve önişleyiciyi yükle (belirtilmemişse)
    if model is None:
        model = load_model()
    
    if preprocessor is None:
        preprocessor = load_preprocessor()
    
    try:
        # Veriyi ön işle
        X_processed, _ = prepare_new_data(employee_data, preprocessor)
        
        # Tahmini yap
        predicted_salary = model.predict(X_processed)[0]
        logger.info(f"Tahmin edilen maaş: {predicted_salary:,.2f} TL")
        
        # Özellik önem derecelerini hesapla (isteğe bağlı)
        if return_features:
            # Modelin tipine göre özellik önem derecelerini al
            feature_importances = {}
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = preprocessor.get_feature_names_out() if hasattr(preprocessor, 'get_feature_names_out') else None
                
                if feature_names is not None and len(feature_names) == len(importances):
                    for i, importance in enumerate(importances):
                        feature_importances[feature_names[i]] = importance
                else:
                    for i, importance in enumerate(importances):
                        feature_importances[f"feature_{i}"] = importance
            
            elif hasattr(model, 'coef_'):
                coeffs = model.coef_
                feature_names = preprocessor.get_feature_names_out() if hasattr(preprocessor, 'get_feature_names_out') else None
                
                if feature_names is not None and len(feature_names) == len(coeffs):
                    for i, coef in enumerate(coeffs):
                        feature_importances[feature_names[i]] = abs(coef)  # Mutlak değer al
                else:
                    for i, coef in enumerate(coeffs):
                        feature_importances[f"feature_{i}"] = abs(coef)  # Mutlak değer al
            
            # Özellik önem derecelerine göre sırala
            sorted_features = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)}
            
            # En önemli özellikleri ve değerlerini al
            top_features = dict(list(sorted_features.items())[:10])  # İlk 10
            
            # Çalışan verisi ile birleştir
            employee_features = {}
            for col in employee_data.columns:
                employee_features[col] = employee_data[col].values[0]
            
            return predicted_salary, {"employee_features": employee_features, "top_features": top_features}
        
        return predicted_salary
    
    except Exception as e:
        logger.error(f"Maaş tahmini sırasında hata oluştu: {str(e)}")
        # Fallback: Rol ve deneyime dayalı basit bir tahmin yap
        role_category = employee_data['Rol_Kategorisi'].values[0] if 'Rol_Kategorisi' in employee_data.columns else 'Yazılım Geliştirme'
        experience = employee_data['Deneyim_Yıl'].values[0] if 'Deneyim_Yıl' in employee_data.columns else 3
        kidem = employee_data['Kıdem'].values[0] if 'Kıdem' in employee_data.columns else 'Mid.'
        
        # Basit bir formül kullanalım
        base_salary = 25000
        if role_category in ["Yazılım Geliştirme", "DevOps ve Altyapı"]:
            base_salary = 28000
        elif role_category in ["Veri", "Yapay Zeka ve Makine Öğrenmesi"]:
            base_salary = 32000
        elif role_category in ["Güvenlik"]:
            base_salary = 30000
        
        # Deneyim faktörü
        exp_factor = 1.0 + (experience * 0.05)
        
        # Kıdem faktörü
        kidem_factors = {
            "Stajyer": 0.4, 
            "Jr.": 0.8, 
            "Mid.": 1.0, 
            "Sr.": 1.4, 
            "Lead": 1.8, 
            "Müdür Yardımcısı": 2.0, 
            "Müdür": 2.5, 
            "Direktör": 3.0
        }
        kidem_factor = kidem_factors.get(kidem, 1.0)
        
        fallback_salary = base_salary * exp_factor * kidem_factor
        logger.info(f"Fallback tahmin kullanıldı: {fallback_salary:,.2f} TL")
        
        return fallback_salary


def predict_multiple_salaries(employees_data, model=None, preprocessor=None):
    """
    Birden fazla çalışanın maaşını tahmin eder.
    
    Args:
        employees_data (pandas.DataFrame): Çalışanlar verisi
        model (object, optional): Eğitilmiş model
        preprocessor (object, optional): Eğitilmiş önişleyici
        
    Returns:
        pandas.DataFrame: Orijinal veri ve tahmin edilen maaşlar
    """
    logger.info(f"{len(employees_data)} çalışan için maaş tahmini yapılıyor")
    
    # Model ve önişleyiciyi yükle (belirtilmemişse)
    if model is None:
        model = load_model()
    
    if preprocessor is None:
        preprocessor = load_preprocessor()
    
    try:
        # Veriyi ön işle
        X_processed, _ = prepare_new_data(employees_data, preprocessor)
        
        # Tahminleri yap
        predicted_salaries = model.predict(X_processed)
        
        # Sonuçları orijinal veriye ekle
        results = employees_data.copy()
        results['Tahmin_Edilen_Maaş_TL'] = predicted_salaries
        
        logger.info(f"Maaş tahminleri tamamlandı. Ortalama maaş: {results['Tahmin_Edilen_Maaş_TL'].mean():,.2f} TL")
        return results
    
    except Exception as e:
        logger.error(f"Çoklu maaş tahmini sırasında hata oluştu: {str(e)}")
        # Fallback: Her çalışan için tek tek tahmin yap
        results = employees_data.copy()
        predicted_salaries = []
        
        for _, row in employees_data.iterrows():
            employee_df = pd.DataFrame([row])
            predicted_salary = predict_salary(employee_df, model, preprocessor)
            predicted_salaries.append(predicted_salary)
        
        results['Tahmin_Edilen_Maaş_TL'] = predicted_salaries
        logger.info(f"Fallback çoklu tahmin tamamlandı. Ortalama maaş: {np.mean(predicted_salaries):,.2f} TL")
        return results


def compare_to_market_average(predicted_salary, employee_data, market_data=None):
    """
    Tahmin edilen maaşı piyasa ortalamasıyla karşılaştırır.
    
    Args:
        predicted_salary (float): Tahmin edilen maaş
        employee_data (pandas.DataFrame): Çalışan verisi
        market_data (pandas.DataFrame, optional): Piyasa verileri
        
    Returns:
        dict: Karşılaştırma sonuçları
    """
    logger.info("Piyasa karşılaştırması yapılıyor")
    
    # Piyasa verisi belirtilmemişse, varsayılan veriyi yükle
    if market_data is None:
        try:
            project_root = get_project_root()
            market_file = os.path.join(project_root, 'data', 'turkiye_it_sektoru_calisanlari.csv')
            
            if os.path.exists(market_file):
                market_data = pd.read_csv(market_file)
                logger.info(f"Piyasa verileri yüklendi: {len(market_data)} kayıt")
            else:
                # Ana veri setini dene
                market_file = os.path.join(project_root, 'data', 'processed', 'market_data.csv')
                if os.path.exists(market_file):
                    market_data = pd.read_csv(market_file)
                    logger.info(f"Piyasa verileri yüklendi: {len(market_data)} kayıt")
                else:
                    logger.warning("Piyasa verileri bulunamadı, karşılaştırma yapılamayacak")
                    return {"comparison": "Piyasa verileri bulunamadı"}
        except Exception as e:
            logger.error(f"Piyasa verileri yüklenirken hata oluştu: {str(e)}")
            return {"comparison": "Piyasa verileri yüklenirken hata oluştu"}
    
    # Çalışan bilgilerini al
    role = employee_data['Rol'].values[0] if 'Rol' in employee_data.columns else 'Bilinmiyor'
    role_category = employee_data['Rol_Kategorisi'].values[0] if 'Rol_Kategorisi' in employee_data.columns else 'Bilinmiyor'
    experience = employee_data['Deneyim_Yıl'].values[0] if 'Deneyim_Yıl' in employee_data.columns else 0
    city = employee_data['Şehir'].values[0] if 'Şehir' in employee_data.columns else 'Bilinmiyor'
    
    # Filtreleri hazırla
    filters = {}
    
    # Rol ile filtrele
    if role != 'Bilinmiyor' and 'Rol' in market_data.columns:
        role_data = market_data[market_data['Rol'] == role]
        if len(role_data) >= 5:  # Yeterli veri varsa
            filters['role'] = {'data': role_data, 'name': f"Rol: {role}"}
    
    # Rol kategorisi ile filtrele
    if role_category != 'Bilinmiyor' and 'Rol_Kategorisi' in market_data.columns:
        category_data = market_data[market_data['Rol_Kategorisi'] == role_category]
        if len(category_data) >= 5:
            filters['category'] = {'data': category_data, 'name': f"Kategori: {role_category}"}
    
    # Deneyim ile filtrele (benzer deneyimler)
    if experience > 0 and 'Deneyim_Yıl' in market_data.columns:
        exp_min = max(0, experience - 2)
        exp_max = experience + 2
        exp_data = market_data[(market_data['Deneyim_Yıl'] >= exp_min) & (market_data['Deneyim_Yıl'] <= exp_max)]
        if len(exp_data) >= 5:
            filters['experience'] = {'data': exp_data, 'name': f"Deneyim: {exp_min}-{exp_max} yıl"}
    
    # Şehir ile filtrele
    if city != 'Bilinmiyor' and 'Şehir' in market_data.columns:
        city_data = market_data[market_data['Şehir'] == city]
        if len(city_data) >= 5:
            filters['city'] = {'data': city_data, 'name': f"Şehir: {city}"}
    
    # Karşılaştırmaları yap
    comparisons = {}
    
    # Tüm veriler ile karşılaştır
    if len(market_data) > 0 and 'Maaş_TL' in market_data.columns:
        all_avg = market_data['Maaş_TL'].mean()
        all_median = market_data['Maaş_TL'].median()
        all_min = market_data['Maaş_TL'].min()
        all_max = market_data['Maaş_TL'].max()
        all_std = market_data['Maaş_TL'].std()
        all_percentile = (predicted_salary >= market_data['Maaş_TL']).mean() * 100  # Yüzdelik dilim
        
        comparisons['all'] = {
            'name': 'Tüm Veriler',
            'count': len(market_data),
            'average': all_avg,
            'median': all_median,
            'min': all_min,
            'max': all_max,
            'std': all_std,
            'diff_from_avg': predicted_salary - all_avg,
            'diff_percentage': ((predicted_salary / all_avg) - 1) * 100 if all_avg > 0 else 0,
            'percentile': all_percentile
        }
    
    # Filtrelenmiş veriler ile karşılaştır
    for filter_key, filter_info in filters.items():
        filter_data = filter_info['data']
        filter_name = filter_info['name']
        
        if len(filter_data) > 0 and 'Maaş_TL' in filter_data.columns:
            avg = filter_data['Maaş_TL'].mean()
            median = filter_data['Maaş_TL'].median()
            min_val = filter_data['Maaş_TL'].min()
            max_val = filter_data['Maaş_TL'].max()
            std = filter_data['Maaş_TL'].std()
            percentile = (predicted_salary >= filter_data['Maaş_TL']).mean() * 100
            
            comparisons[filter_key] = {
                'name': filter_name,
                'count': len(filter_data),
                'average': avg,
                'median': median,
                'min': min_val,
                'max': max_val,
                'std': std,
                'diff_from_avg': predicted_salary - avg,
                'diff_percentage': ((predicted_salary / avg) - 1) * 100 if avg > 0 else 0,
                'percentile': percentile
            }
    
    logger.info("Piyasa karşılaştırması tamamlandı")
    return comparisons


def generate_salary_recommendation(predicted_salary, employee_data, market_comparisons, recommendation_basis='percentile'):
    """
    Maaş tahmini ve piyasa karşılaştırmasına göre öneriler oluşturur.
    
    Args:
        predicted_salary (float): Tahmin edilen maaş
        employee_data (pandas.DataFrame): Çalışan verisi
        market_comparisons (dict): Piyasa karşılaştırma sonuçları
        recommendation_basis (str): Öneri temeli ('percentile' veya 'diff_percentage')
        
    Returns:
        dict: Maaş önerileri ve açıklamalar
    """
    logger.info("Maaş önerisi oluşturuluyor")
    
    recommendations = {
        'predicted_salary': predicted_salary,
        'market_comparisons': market_comparisons,
        'recommendations': []
    }
    
    # Piyasa karşılaştırması kontrol et
    if not market_comparisons or isinstance(market_comparisons, dict) and "comparison" in market_comparisons:
        # Piyasa karşılaştırması yapılamadıysa genel bir öneri sun
        recommendations['recommendations'].append({
            'type': 'general',
            'message': "Piyasa karşılaştırması yapılamadı, genel bir değerlendirme sunuluyor.",
            'suggestion': f"Tahmin edilen maaş: {predicted_salary:,.0f} TL",
            'adjustment': "Rol, deneyim ve lokasyona göre maaş değerlendirmesi yapılması önerilir."
        })
        return recommendations
    
    # Rol ve deneyime göre filtrelenmiş karşılaştırmaları bul
    role_comparison = market_comparisons.get('role', None)
    category_comparison = market_comparisons.get('category', None)
    exp_comparison = market_comparisons.get('experience', None)
    all_comparison = market_comparisons.get('all', None)
    
    # En uygun karşılaştırmayı seç (veri sayısına göre)
    best_comparison = None
    if role_comparison and role_comparison['count'] >= 10:
        best_comparison = role_comparison
    elif category_comparison and category_comparison['count'] >= 15:
        best_comparison = category_comparison
    elif exp_comparison and exp_comparison['count'] >= 20:
        best_comparison = exp_comparison
    elif all_comparison:
        best_comparison = all_comparison
    
    if best_comparison is None:
        recommendations['recommendations'].append({
            'type': 'error',
            'message': 'Yeterli karşılaştırma verisi bulunamadı'
        })
        return recommendations
    
    # Yüzdelik dilime göre öneriler
    if recommendation_basis == 'percentile':
        percentile = best_comparison['percentile']
        
        if percentile < 25:
            recommendations['recommendations'].append({
                'type': 'low',
                'message': f"Tahmin edilen maaş piyasanın alt %25'lik diliminde yer alıyor.",
                'suggestion': f"Piyasa ortalamasına göre düşük. Ortalama maaş: {best_comparison['average']:,.0f} TL",
                'adjustment': "Maaşı piyasa ortalamasına yaklaştırmak için artış düşünülebilir.",
                'target_range': [best_comparison['average'] * 0.9, best_comparison['average'] * 1.1]
            })
        elif percentile < 50:
            recommendations['recommendations'].append({
                'type': 'below_average',
                'message': f"Tahmin edilen maaş piyasa ortalamasının altında (%{percentile:.1f} diliminde).",
                'suggestion': f"Ortalama maaşın biraz altında. Ortalama: {best_comparison['average']:,.0f} TL",
                'adjustment': "Maaşı piyasa ortalamasına yaklaştırmak için hafif artış düşünülebilir.",
                'target_range': [best_comparison['average'] * 0.95, best_comparison['average'] * 1.05]
            })
        elif percentile < 75:
            recommendations['recommendations'].append({
                'type': 'average',
                'message': f"Tahmin edilen maaş piyasa ortalamasına yakın (%{percentile:.1f} diliminde).",
                'suggestion': f"Maaş piyasa standartlarına uygun. Ortalama: {best_comparison['average']:,.0f} TL",
                'adjustment': "Maaş piyasa standartlarına uygun, büyük bir değişikliğe gerek yok.",
                'target_range': [predicted_salary * 0.95, predicted_salary * 1.05]
            })
        else:
            recommendations['recommendations'].append({
                'type': 'high',
                'message': f"Tahmin edilen maaş piyasanın üst %25'lik diliminde yer alıyor (%{percentile:.1f}).",
                'suggestion': f"Piyasa ortalamasının üzerinde bir maaş. Ortalama: {best_comparison['average']:,.0f} TL",
                'adjustment': "Maaş zaten piyasa ortalamasının üzerinde, artış öncelikli değil.",
                'target_range': [predicted_salary * 0.95, predicted_salary * 1.05]
            })
    
    # Ortalamadan farka göre öneriler
    else:  # diff_percentage
        diff_percentage = best_comparison['diff_percentage']
        
        if diff_percentage < -15:
            recommendations['recommendations'].append({
                'type': 'significantly_low',
                'message': f"Tahmin edilen maaş piyasa ortalamasından %{abs(diff_percentage):.1f} daha düşük.",
                'suggestion': f"Piyasa ortalamasının oldukça altında. Ortalama: {best_comparison['average']:,.0f} TL",
                'adjustment': "Rekabetçi kalmak için önemli bir maaş artışı düşünülebilir.",
                'target_range': [best_comparison['average'] * 0.85, best_comparison['average'] * 1.05]
            })
        elif diff_percentage < -5:
            recommendations['recommendations'].append({
                'type': 'somewhat_low',
                'message': f"Tahmin edilen maaş piyasa ortalamasından %{abs(diff_percentage):.1f} daha düşük.",
                'suggestion': f"Piyasa ortalamasının biraz altında. Ortalama: {best_comparison['average']:,.0f} TL",
                'adjustment': "Piyasa standartlarına uyum için hafif bir artış düşünülebilir.",
                'target_range': [best_comparison['average'] * 0.95, best_comparison['average'] * 1.05]
            })
        elif diff_percentage < 5:
            recommendations['recommendations'].append({
                'type': 'fair',
                'message': f"Tahmin edilen maaş piyasa ortalamasına çok yakın (fark: %{diff_percentage:.1f}).",
                'suggestion': f"Adil bir maaş teklifi. Piyasa ortalaması: {best_comparison['average']:,.0f} TL",
                'adjustment': "Maaş piyasa standartlarına uygun, büyük bir değişikliğe gerek yok.",
                'target_range': [predicted_salary * 0.97, predicted_salary * 1.03]
            })
        elif diff_percentage < 15:
            recommendations['recommendations'].append({
                'type': 'somewhat_high',
                'message': f"Tahmin edilen maaş piyasa ortalamasından %{diff_percentage:.1f} daha yüksek.",
                'suggestion': f"Piyasa ortalamasının biraz üzerinde. Ortalama: {best_comparison['average']:,.0f} TL",
                'adjustment': "Maaş zaten rekabetçi, artış öncelikli değil.",
                'target_range': [predicted_salary * 0.95, predicted_salary * 1.05]
            })
        else:
            recommendations['recommendations'].append({
                'type': 'significantly_high',
                'message': f"Tahmin edilen maaş piyasa ortalamasından %{diff_percentage:.1f} daha yüksek.",
                'suggestion': f"Piyasa ortalamasının oldukça üzerinde. Ortalama: {best_comparison['average']:,.0f} TL",
                'adjustment': "Maaş piyasa üzerinde, maliyet optimizasyonu düşünülebilir veya üstün performans beklenebilir.",
                'target_range': [best_comparison['average'] * 0.95, predicted_salary]
            })
    
    # Çalışan deneyimine göre ek öneriler
    experience = employee_data['Deneyim_Yıl'].values[0] if 'Deneyim_Yıl' in employee_data.columns else 0
    
    if experience < 2:
        recommendations['recommendations'].append({
            'type': 'experience_factor',
            'message': "Çalışan kariyerinin başlangıcında (< 2 yıl deneyim).",
            'suggestion': "Gelişim potansiyeli göz önünde bulundurulmalı.",
            'adjustment': "Çalışma süresi arttıkça kademeli artışlar planlanabilir."
        })
    elif experience > 10:
        recommendations['recommendations'].append({
            'type': 'experience_factor',
            'message': "Çalışan deneyimli bir profesyonel (> 10 yıl deneyim).",
            'suggestion': "Deneyim seviyesi dikkate alınmalı.",
            'adjustment': "Üst dilim maaş önerileri değerlendirilebilir."
        })
    
    logger.info("Maaş önerisi oluşturuldu")
    return recommendations


def save_prediction_results(employee_data, predicted_salary, market_comparisons, recommendations=None, output_dir=None):
    """
    Tahmin sonuçlarını, piyasa karşılaştırmalarını ve önerileri kaydeder.
    
    Args:
        employee_data (pandas.DataFrame): Çalışan verisi
        predicted_salary (float): Tahmin edilen maaş
        market_comparisons (dict): Piyasa karşılaştırma sonuçları
        recommendations (dict, optional): Maaş önerileri
        output_dir (str, optional): Çıktı dizini
        
    Returns:
        str: Kaydedilen dosyanın yolu
    """
    if output_dir is None:
        project_root = get_project_root()
        output_dir = os.path.join(project_root, 'data', 'predictions')
    
    # Klasörün var olduğundan emin ol
    os.makedirs(output_dir, exist_ok=True)
    
    # Dosya adını belirle
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Çalışan adını al (varsa)
    employee_name = None
    if 'Ad' in employee_data.columns and 'Soyad' in employee_data.columns:
        ad = employee_data['Ad'].values[0]
        soyad = employee_data['Soyad'].values[0]
        employee_name = f"{ad}_{soyad}"
    
    if employee_name:
        file_name = f"salary_prediction_{employee_name}_{timestamp}.json"
    else:
        file_name = f"salary_prediction_{timestamp}.json"
    
    file_path = os.path.join(output_dir, file_name)
    
    # Sonuçları hazırla
    results = {
        'timestamp': timestamp,
        'employee_data': employee_data.to_dict(orient='records')[0],
        'predicted_salary': float(predicted_salary),
        'market_comparisons': market_comparisons
    }
    
    if recommendations:
        results['recommendations'] = recommendations
    
    # JSON olarak kaydet
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Tahmin sonuçları kaydedildi: {file_path}")
        return file_path
    
    except Exception as e:
        logger.error(f"Tahmin sonuçları kaydedilirken hata oluştu: {str(e)}")
        raise


if __name__ == "__main__":
    # Test işlevselliği
    try:
        # Test çalışanı oluştur
        test_employee = load_test_employee()
        print(f"Test çalışanı oluşturuldu: {test_employee.iloc[0]['Rol']}")
        
        # Basit tahmin testi
        try:
            # Not: Gerçek model ve önişleyici dosyaları yoksa hata verecektir
            salary = predict_salary(test_employee)
            print(f"Tahmin edilen maaş: {salary:,.2f} TL")
        except FileNotFoundError:
            # Model yoksa yapay tahmin yap
            print("Model bulunamadı, yapay tahmin yapılıyor")
            salary = 25000 + (test_employee['Deneyim_Yıl'].values[0] * 1000)
            print(f"Yapay tahmin: {salary:,.2f} TL")
        
        print("Tahmin modülü başarıyla test edildi.")
    except Exception as e:
        logger.error(f"Test sırasında hata oluştu: {str(e)}")