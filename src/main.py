import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# src klasörünü Python yoluna ekle
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.append(src_dir)

from src.data.data_loader import (
    get_project_root, load_it_salary_data, 
    create_data_splits, save_data_splits, load_data_splits,
    load_test_employee, save_model_results
)
from src.data.preprocess import (
    prepare_it_salary_data, prepare_data_for_modeling, 
    identify_column_types, handle_missing_values, handle_outliers,
    feature_engineering_it_salary
)
from src.models.train_model import (
    train_model, evaluate_model, hyperparameter_tuning,
    train_and_evaluate_all_models, save_model
)
from src.models.predict_model import (
    predict_salary, predict_multiple_salaries, 
    compare_to_market_average, generate_salary_recommendation
)
from src.models.evaluate_model import (
    evaluate_model_performance, evaluate_feature_importance,
    create_evaluation_report
)
from src.models.ensemble_models import (
    train_all_ensembles, get_best_ensemble_model
)

# Loglama yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"salary_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_arg_parser():
    """Komut satırı argümanlarını ayarlar."""
    parser = argparse.ArgumentParser(description='IT Sektörü Maaş Tahmin Sistemi')
    
    # Ana işlem türü
    parser.add_argument('--mode', type=str, default='train', 
                      choices=['train', 'predict', 'evaluate', 'ensemble'],
                      help='İşlem türü: train, predict, evaluate, ensemble')
    
    # Veri işleme
    parser.add_argument('--data_file', type=str, default=None,
                      help='Veri dosyasının yolu')
    parser.add_argument('--test_size', type=float, default=0.2,
                      help='Test seti oranı')
    parser.add_argument('--val_size', type=float, default=0.25,
                      help='Doğrulama seti oranı (eğitim setinden)')
    parser.add_argument('--random_state', type=int, default=42,
                      help='Rastgele sayı üreteci tohum değeri')
    
    # Model eğitim parametreleri
    parser.add_argument('--model', type=str, default='rf',
                      choices=['mlr', 'dt', 'rf', 'ridge', 'adaboost', 'all'],
                      help='Kullanılacak model türü')
    parser.add_argument('--tune_hyperparams', action='store_true',
                      help='Hiperparametre optimizasyonu yap')
    parser.add_argument('--n_iter', type=int, default=10,
                      help='Hiperparametre optimizasyonu için iterasyon sayısı')
    
    # Tahmin parametreleri
    parser.add_argument('--employee_file', type=str, default=None,
                      help='Tahmin için çalışan verisinin yolu')
    parser.add_argument('--model_file', type=str, default=None,
                      help='Kullanılacak model dosyasının yolu')
    parser.add_argument('--preprocessor_file', type=str, default=None,
                      help='Kullanılacak önişleyici dosyasının yolu')
    
    # Değerlendirme parametreleri
    parser.add_argument('--splits_folder', type=str, default=None,
                      help='Değerlendirme için veri bölmelerinin yolu')
    
    # Topluluk model parametreleri
    parser.add_argument('--ensemble_type', type=str, default='all',
                      choices=['voting', 'stacking', 'bagging', 'boosting', 'all'],
                      help='Oluşturulacak topluluk modeli türü')
    
    # Çıktı parametreleri
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Çıktı dizini')
    parser.add_argument('--save_results', action='store_true',
                      help='Sonuçları kaydet')
    
    return parser


def train_salary_model(args):
    """Maaş tahmin modelini eğitir."""
    logger.info("Maaş tahmin modeli eğitiliyor")
    
    # Veriyi yükle
    if args.data_file:
        logger.info(f"Veri dosyası belirtildi: {args.data_file}")
        df = load_it_salary_data(args.data_file)
    else:
        logger.info("Varsayılan veri dosyası kullanılıyor")
        df = load_it_salary_data()
    
    # Veriyi ön işle
    logger.info("Veri ön işleniyor")
    df_processed, column_types = prepare_it_salary_data(df, feature_eng=True)
    
    # Veriyi böl
    logger.info("Veri eğitim/doğrulama/test olarak bölünüyor")
    splits = create_data_splits(
        df_processed, 
        target_column='Maaş_TL', 
        test_size=args.test_size, 
        val_size=args.val_size, 
        random_state=args.random_state
    )
    
    # Veri bölmelerini kaydet
    if args.save_results:
        splits_folder = save_data_splits(splits)
        logger.info(f"Veri bölmeleri kaydedildi: {splits_folder}")
    
    # Modelleme için veriyi hazırla
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data_for_modeling(
        df_processed, target_column='Maaş_TL', test_size=args.test_size, random_state=args.random_state
    )
    
    # Doğrulama seti
    X_val = splits['X_val'].values
    y_val = splits['y_val'].values
    
    # Hiperparametre optimizasyonu
    if args.tune_hyperparams:
        logger.info(f"Hiperparametre optimizasyonu yapılıyor: {args.model}")
        
        if args.model == 'all':
            logger.warning("Tüm modeller için hiperparametre optimizasyonu uzun sürebilir")
            models, all_metrics, comparison = train_and_evaluate_all_models(
                X_train, y_train, X_test, y_test, 
                feature_names=feature_names,
                do_hyperparameter_tuning=True,
                X_val=X_val, y_val=y_val,
                save_models=args.save_results,
                random_state=args.random_state
            )
            
            logger.info("Tüm modeller eğitildi ve değerlendirildi")
            logger.info(f"En iyi model: {comparison.iloc[0]['Model']}, MAPE: {comparison.iloc[0]['Test MAPE (%)']:.2f}%, R²: {comparison.iloc[0]['Test R²']:.4f}")
            
            # En iyi modeli seç
            best_model_name = comparison.iloc[0]['Model'].lower()
            best_model = models[best_model_name]
            
            # En iyi modeli değerlendir
            metrics, visualizations = evaluate_model_performance(
                best_model, X_test, y_test, feature_names=feature_names
            )
            
            # Özellik önem derecelerini değerlendir
            feature_importance, _ = evaluate_feature_importance(
                best_model, feature_names
            )
            
            logger.info(f"En önemli 5 özellik: {', '.join(feature_importance['Feature'].head(5).tolist())}")
        else:
            # Tek bir modeli eğit ve optimize et
            best_model, best_params, _ = hyperparameter_tuning(
                X_train, y_train, X_val, y_val, 
                model_name=args.model, 
                n_iter=args.n_iter,
                random_state=args.random_state
            )
            
            # Modeli değerlendir
            metrics, feature_importance = evaluate_model(
                best_model, X_test, y_test, X_train, y_train, feature_names
            )
            
            logger.info(f"Model eğitildi ve değerlendirildi")
            logger.info(f"Test metrikleri: MAPE={metrics['test_mape']:.2f}%, R²={metrics['test_r2']:.4f}")
            
            # Modeli kaydet
            if args.save_results:
                model_file = save_model(
                    best_model, args.model, metrics, feature_importance, best_params
                )
                logger.info(f"Model kaydedildi: {model_file}")
    else:
        # Hiperparametre optimizasyonu olmadan eğit
        if args.model == 'all':
            # Tüm modelleri eğit
            models, all_metrics, comparison = train_and_evaluate_all_models(
                X_train, y_train, X_test, y_test, 
                feature_names=feature_names,
                do_hyperparameter_tuning=False,
                save_models=args.save_results,
                random_state=args.random_state
            )
            
            logger.info("Tüm modeller eğitildi ve değerlendirildi")
            logger.info(f"En iyi model: {comparison.iloc[0]['Model']}, MAPE: {comparison.iloc[0]['Test MAPE (%)']:.2f}%, R²: {comparison.iloc[0]['Test R²']:.4f}")
            
            # Çıktı dizinini belirle
            if args.output_dir:
                output_dir = args.output_dir
            else:
                project_root = get_project_root()
                output_dir = os.path.join(project_root, 'reports')
                
            os.makedirs(output_dir, exist_ok=True)
            
            # Karşılaştırma sonuçlarını kaydet
            comparison_file = os.path.join(output_dir, f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            comparison.to_csv(comparison_file, index=False)
            logger.info(f"Model karşılaştırma sonuçları kaydedildi: {comparison_file}")
        else:
            # Tek bir modeli eğit
            model = train_model(
                X_train, y_train, model_name=args.model, random_state=args.random_state
            )
            
            # Modeli değerlendir
            metrics, feature_importance = evaluate_model(
                model, X_test, y_test, X_train, y_train, feature_names
            )
            
            logger.info(f"Model eğitildi ve değerlendirildi")
            logger.info(f"Test metrikleri: MAPE={metrics['test_mape']:.2f}%, R²={metrics['test_r2']:.4f}")
            
            # Modeli kaydet
            if args.save_results:
                model_file = save_model(
                    model, args.model, metrics, feature_importance
                )
                logger.info(f"Model kaydedildi: {model_file}")
    
    logger.info("Maaş tahmin modeli eğitimi tamamlandı")


def predict_employee_salary(args):
    """Çalışanların maaşını tahmin eder."""
    logger.info("Çalışan maaşı tahmin ediliyor")
    
    # Çalışan verisini yükle
    if args.employee_file:
        logger.info(f"Çalışan dosyası belirtildi: {args.employee_file}")
        employee_data = load_test_employee(args.employee_file)
    else:
        logger.info("Örnek çalışan verisi oluşturuluyor")
        employee_data = load_test_employee()
    
    # Model ve önişleyiciyi yükle
    model = None
    preprocessor = None
    
    if args.model_file:
        from src.models.train_model import load_trained_model
        model = load_trained_model(args.model_file)
    
    if args.preprocessor_file:
        from src.data.preprocess import load_preprocessor
        preprocessor = load_preprocessor(args.preprocessor_file)
    
    # Maaş tahmini yap
    try:
        if employee_data.shape[0] == 1:
            # Tek çalışan için tahmin
            predicted_salary = predict_salary(employee_data, model, preprocessor)
            
            logger.info(f"Tahmin edilen maaş: {predicted_salary:,.2f} TL")
            
            # Piyasa karşılaştırması yap
            market_comparisons = compare_to_market_average(predicted_salary, employee_data)
            
            # Maaş önerisi oluştur
            recommendations = generate_salary_recommendation(
                predicted_salary, employee_data, market_comparisons
            )
            
            # Sonuçları göster
            print(f"\nTahmin Edilen Maaş: {predicted_salary:,.2f} TL")
            
            if market_comparisons:
                print("\nPiyasa Karşılaştırması:")
                for key, comparison in market_comparisons.items():
                    print(f"  {comparison['name']} ({comparison['count']} kayıt)")
                    print(f"    Ortalama: {comparison['average']:,.2f} TL")
                    print(f"    Fark: {comparison['diff_from_avg']:,.2f} TL (%{comparison['diff_percentage']:.1f})")
                    print(f"    Yüzdelik dilim: %{comparison['percentile']:.1f}")
            
            if recommendations and 'recommendations' in recommendations:
                print("\nÖneriler:")
                for rec in recommendations['recommendations']:
                    print(f"  {rec['message']}")
                    if 'suggestion' in rec:
                        print(f"  {rec['suggestion']}")
                    if 'adjustment' in rec:
                        print(f"  {rec['adjustment']}")
            
            # Sonuçları kaydet
            if args.save_results:
                from src.models.predict_model import save_prediction_results
                results_file = save_prediction_results(
                    employee_data, predicted_salary, market_comparisons, recommendations, args.output_dir
                )
                logger.info(f"Tahmin sonuçları kaydedildi: {results_file}")
        else:
            # Birden fazla çalışan için tahmin
            results = predict_multiple_salaries(employee_data, model, preprocessor)
            
            logger.info(f"{len(results)} çalışan için maaş tahmini yapıldı")
            logger.info(f"Ortalama tahmini maaş: {results['Tahmin_Edilen_Maaş_TL'].mean():,.2f} TL")
            
            # Sonuçları göster
            print(f"\n{len(results)} çalışan için maaş tahmini yapıldı")
            print(f"Ortalama tahmini maaş: {results['Tahmin_Edilen_Maaş_TL'].mean():,.2f} TL")
            
            # İlk 5 tahmini göster
            print("\nİlk 5 tahmin:")
            for i, row in results.head().iterrows():
                print(f"  {row['Ad']} {row['Soyad']}: {row['Tahmin_Edilen_Maaş_TL']:,.2f} TL")
            
            # Sonuçları kaydet
            if args.save_results:
                if args.output_dir:
                    output_dir = args.output_dir
                else:
                    project_root = get_project_root()
                    output_dir = os.path.join(project_root, 'data', 'predictions')
                
                os.makedirs(output_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = os.path.join(output_dir, f"salary_predictions_{timestamp}.csv")
                results.to_csv(results_file, index=False)
                logger.info(f"Tahmin sonuçları kaydedildi: {results_file}")
    
    except Exception as e:
        logger.error(f"Maaş tahmini sırasında hata oluştu: {str(e)}")
        raise
    
    logger.info("Maaş tahmin işlemi tamamlandı")


def evaluate_salary_model(args):
    """Eğitilmiş modelin performansını değerlendirir."""
    logger.info("Model performansı değerlendiriliyor")
    
    # Veri bölmelerini yükle
    if args.splits_folder:
        logger.info(f"Veri bölmeleri belirtildi: {args.splits_folder}")
        splits = load_data_splits(args.splits_folder)
    else:
        # Veriyi yükle
        if args.data_file:
            logger.info(f"Veri dosyası belirtildi: {args.data_file}")
            df = load_it_salary_data(args.data_file)
        else:
            logger.info("Varsayılan veri dosyası kullanılıyor")
            df = load_it_salary_data()
        
        # Veriyi ön işle
        logger.info("Veri ön işleniyor")
        df_processed, column_types = prepare_it_salary_data(df, feature_eng=True)
        
        # Veriyi böl
        logger.info("Veri eğitim/doğrulama/test olarak bölünüyor")
        splits = create_data_splits(
            df_processed, 
            target_column='Maaş_TL', 
            test_size=args.test_size, 
            val_size=args.val_size, 
            random_state=args.random_state
        )
    
    # Model ve önişleyiciyi yükle
    model = None
    
    if args.model_file:
        from src.models.train_model import load_trained_model
        model = load_trained_model(args.model_file)
    else:
        # En son bir model eğit
        logger.info("Model dosyası belirtilmedi, yeni model eğitiliyor")
        X_train = splits['X_train'].values
        y_train = splits['y_train'].values
        
        model = train_model(
            X_train, y_train, model_name=args.model, random_state=args.random_state
        )
    
    # Model performansını değerlendir
    X_test = splits['X_test'].values
    y_test = splits['y_test'].values
    
    feature_names = splits['metadata']['feature_names'] if 'metadata' in splits else None
    
    # Değerlendirme
    metrics, visualizations = evaluate_model_performance(
        model, X_test, y_test, feature_names=feature_names, output_dir=args.output_dir
    )
    
    logger.info(f"Model değerlendirildi")
    logger.info(f"Test metrikleri: MAPE={metrics['mape']:.2f}%, R²={metrics['r2']:.4f}")
    
    # Özellik önem derecelerini değerlendir
    feature_importance, feature_imp_path = evaluate_feature_importance(
        model, feature_names, output_dir=args.output_dir
    )
    
    # Sonuçları göster
    print(f"\nModel Değerlendirme Sonuçları:")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  MAE: {metrics['mae']:.2f}")
    
    print("\nEn Önemli 5 Özellik:")
    for i, row in feature_importance.head(5).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    # Grafikler hakkında bilgi
    print("\nOluşturulan Grafikler:")
    for vis_name, vis_path in visualizations.items():
        print(f"  {vis_name}: {vis_path}")
    
    # Değerlendirme raporu oluştur
    if args.save_results:
        model_name = args.model if args.model_file is None else os.path.basename(args.model_file).split('_')[0]
        report_file = create_evaluation_report(
            metrics, visualizations, model_name, output_dir=args.output_dir
        )
        logger.info(f"Değerlendirme raporu oluşturuldu: {report_file}")
    
    logger.info("Model değerlendirme işlemi tamamlandı")


def train_ensemble_models(args):
    """Topluluk modellerini eğitir."""
    logger.info("Topluluk modelleri eğitiliyor")
    
    # Veriyi yükle
    if args.data_file:
        logger.info(f"Veri dosyası belirtildi: {args.data_file}")
        df = load_it_salary_data(args.data_file)
    else:
        logger.info("Varsayılan veri dosyası kullanılıyor")
        df = load_it_salary_data()
    
    # Veriyi ön işle
    logger.info("Veri ön işleniyor")
    df_processed, column_types = prepare_it_salary_data(df, feature_eng=True)
    
    # Veriyi böl
    logger.info("Veri eğitim/doğrulama/test olarak bölünüyor")
    splits = create_data_splits(
        df_processed, 
        target_column='Maaş_TL', 
        test_size=args.test_size, 
        val_size=args.val_size, 
        random_state=args.random_state
    )
    
    # Modelleme için veriyi hazırla
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data_for_modeling(
        df_processed, target_column='Maaş_TL', test_size=args.test_size, random_state=args.random_state
    )
    
    # Doğrulama seti
    X_val = splits['X_val'].values
    y_val = splits['y_val'].values
    
    # Temel modelleri eğit
    logger.info("Temel modeller eğitiliyor")
    
    base_models = ['mlr', 'dt', 'rf', 'ridge']
    models_dict = {}
    
    for model_name in base_models:
        model = train_model(
            X_train, y_train, model_name=model_name, random_state=args.random_state
        )
        models_dict[model_name] = model
    
    # Topluluk modellerini eğit
    if args.ensemble_type == 'all':
        # Tüm topluluk modellerini eğit
        ensembles, all_metrics, comparison = train_all_ensembles(
            X_train, y_train, X_test, y_test, models_dict=models_dict,
            save_models_flag=args.save_results, random_state=args.random_state
        )
        
        logger.info("Tüm topluluk modelleri eğitildi ve değerlendirildi")
        logger.info(f"En iyi topluluk modeli: {comparison.iloc[0]['Model']}, MAPE: {comparison.iloc[0]['MAPE (%)']:.2f}%, R²: {comparison.iloc[0]['R²']:.4f}")
        
        # En iyi topluluk modelini seç
        best_name, best_model, best_value = get_best_ensemble_model(ensembles, all_metrics)
        
        # Sonuçları göster
        print(f"\nTopluluk Modelleri Karşılaştırması:")
        print(comparison)
        
        print(f"\nEn İyi Topluluk Modeli: {best_name}")
        print(f"  MAPE: {all_metrics[best_name]['mape']:.2f}%")
        print(f"  R²: {all_metrics[best_name]['r2']:.4f}")
    else:
        # Tek bir topluluk modeli eğit
        from src.models.ensemble_models import (
            create_ensemble_from_trained_models,
            create_bagging_ensemble,
            create_boosting_ensemble
        )
        
        if args.ensemble_type == 'voting':
            # Oylama modeli
            ensemble = create_ensemble_from_trained_models(
                X_train, y_train, models_dict, ensemble_type='voting'
            )
        elif args.ensemble_type == 'stacking':
            # Yığınlama modeli
            ensemble = create_ensemble_from_trained_models(
                X_train, y_train, models_dict, ensemble_type='stacking'
            )
        elif args.ensemble_type == 'bagging':
            # Torbalama modeli (Decision Tree tabanlı)
            dt_base = DecisionTreeRegressor(random_state=args.random_state)
            ensemble = create_bagging_ensemble(
                dt_base, n_estimators=50, random_state=args.random_state
            )
            ensemble.fit(X_train, y_train)
        elif args.ensemble_type == 'boosting':
            # Güçlendirme modeli
            ensemble = create_boosting_ensemble(
                X_train, y_train, base_estimator='dt', random_state=args.random_state
            )
        
        # Modeli değerlendir
        y_pred = ensemble.predict(X_test)
        
        from src.models.evaluate_model import calculate_metrics
        metrics = calculate_metrics(y_test, y_pred)
        
        logger.info(f"{args.ensemble_type} topluluk modeli eğitildi ve değerlendirildi")
        logger.info(f"Test metrikleri: MAPE={metrics['mape']:.2f}%, R²={metrics['r2']:.4f}")
        
        # Modeli kaydet
        if args.save_results:
            from src.models.ensemble_models import save_ensemble_model
            model_file = save_ensemble_model(ensemble, args.ensemble_type)
            logger.info(f"Topluluk modeli kaydedildi: {model_file}")
        
        # Sonuçları göster
        print(f"\n{args.ensemble_type.capitalize()} Topluluk Modeli Değerlendirme Sonuçları:")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        print(f"  R²: {metrics['r2']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  MAE: {metrics['mae']:.2f}")
    
    logger.info("Topluluk modelleri eğitimi tamamlandı")


def main():
    """Ana program işlevi."""
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    logger.info(f"IT Sektörü Maaş Tahmin Sistemi başlatılıyor. Mod: {args.mode}")
    
    try:
        if args.mode == 'train':
            train_salary_model(args)
        elif args.mode == 'predict':
            predict_employee_salary(args)
        elif args.mode == 'evaluate':
            evaluate_salary_model(args)
        elif args.mode == 'ensemble':
            train_ensemble_models(args)
        else:
            logger.error(f"Bilinmeyen mod: {args.mode}")
            parser.print_help()
    except Exception as e:
        logger.error(f"Program çalışırken hata oluştu: {str(e)}")
        raise
    
    logger.info("Program tamamlandı")


if __name__ == "__main__":
    main()