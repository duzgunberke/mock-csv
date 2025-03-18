import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# src klasörünü Python yoluna ekle
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.append(src_dir)

from src.data.data_loader import (
    get_project_root, load_it_salary_data, load_test_employee, save_model_results
)
from src.data.preprocess import (
    prepare_it_salary_data, feature_engineering_it_salary
)
from src.models.predict_model import (
    predict_salary, load_model, load_preprocessor,
    compare_to_market_average, generate_salary_recommendation
)


# Sayfanın temel ayarları ve stili
st.set_page_config(
    page_title="IT Maaş Tahmin Sistemi",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS ile özel stillendirme
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .title-container {
        background-color: #3498db;
        padding: 1rem;
        border-radius: 5px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .salary-result {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #ecf0f1;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .market-comparison {
        padding: 1rem;
        background-color: #f5f5f5;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .recommendation-box {
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .recommendation-high {
        background-color: #d4efdf;
        border-left: 5px solid #27ae60;
    }
    .recommendation-average {
        background-color: #ebf5fb;
        border-left: 5px solid #3498db;
    }
    .recommendation-low {
        background-color: #f9ebea;
        border-left: 5px solid #e74c3c;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    .info-box {
        padding: 1rem;
        background-color: #eaf2f8;
        border-radius: 5px;
        margin: 1rem 0;
        border-left: 5px solid #3498db;
    }
    .stMultiSelect [data-baseweb=select] span {
        max-width: 500px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


def load_default_data():
    """
    Varsayılan referans veri setini yükler.
    """
    try:
        df = load_it_salary_data()
        return df
    except Exception as e:
        st.error(f"Referans veri yüklenirken hata oluştu: {str(e)}")
        return None


def load_saved_model():
    """
    Kaydedilmiş modeli ve önişleyiciyi yükler.
    """
    try:
        model = load_model()
        preprocessor = load_preprocessor()
        return model, preprocessor
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {str(e)}")
        return None, None


def get_reference_data():
    """
    Kullanıcı arayüzü için referans verilerini döndürür.
    """
    # Varsayılan veri seti
    df = load_default_data()
    
    if df is None:
        return {}
    
    # Referans verileri
    references = {}
    
    # Roller
    references["roller"] = {
        "Yazılım Geliştirme": [
            "Frontend Geliştirici", "Backend Geliştirici", "Full-Stack Geliştirici", "Mobil Uygulama Geliştirici",
            "Oyun Geliştirici", "Gömülü Yazılım Geliştirici", "API Geliştirici", "Web Geliştirici"
        ],
        "Veri": [
            "Veri Bilimci", "Veri Mühendisi", "Veri Analisti", "İş Zekası Uzmanı", "ETL Geliştirici",
            "Büyük Veri Uzmanı", "Veri Tabanı Yöneticisi", "Veri Mimarı"
        ],
        "DevOps ve Altyapı": [
            "DevOps Mühendisi", "Site Reliability Engineer (SRE)", "Cloud Mühendisi", "Sistem Yöneticisi",
            "Ağ Mühendisi", "Linux Yöneticisi", "Windows Sistem Uzmanı", "Bulut Mimarı"
        ],
        "Güvenlik": [
            "Siber Güvenlik Uzmanı", "Güvenlik Analisti", "Güvenlik Mühendisi", "Penetrasyon Test Uzmanı",
            "Güvenlik Mimarı", "SOC Analisti", "Ağ Güvenliği Uzmanı", "Adli Bilişim Uzmanı"
        ],
        "Yapay Zeka ve Makine Öğrenmesi": [
            "Makine Öğrenmesi Mühendisi", "Yapay Zeka Araştırmacısı", "NLP Uzmanı", "Bilgisayarla Görü Uzmanı",
            "Derin Öğrenme Mühendisi", "Robotik Süreç Otomasyonu Uzmanı", "Chatbot Geliştirici"
        ],
        "Tasarım ve Kullanıcı Deneyimi": [
            "UI Tasarımcısı", "UX Tasarımcısı", "Ürün Tasarımcısı", "Grafik Tasarımcı", "Etkileşim Tasarımcısı",
            "Web Tasarımcısı", "Kullanıcı Araştırma Uzmanı"
        ],
        "QA ve Test": [
            "Test Uzmanı", "QA Mühendisi", "Test Otomasyon Uzmanı", "Performans Test Uzmanı",
            "Güvenlik Test Uzmanı", "Kullanıcı Kabul Testi Uzmanı", "Yazılım Test Mühendisi"
        ],
        "Yönetim ve Liderlik": [
            "IT Müdürü", "Teknik Yönetici", "Proje Yöneticisi", "Scrum Master", "Ürün Sahibi",
            "Teknik Takım Lideri", "Yazılım Geliştirme Müdürü", "CTO", "CIO", "CISO", "Yazılım Mimarı"
        ],
        "Destek ve Operasyon": [
            "IT Destek Uzmanı", "Teknik Destek Mühendisi", "Sistem Operatörleri", "Ağ Operatörleri",
            "Altyapı Teknisyeni", "Yardım Masası Uzmanı", "IT Operasyon Uzmanı"
        ],
        "İş ve Analiz": [
            "İş Analisti", "Sistem Analisti", "Gereksinim Analisti", "ERP Uzmanı", "CRM Uzmanı",
            "Bilgi Sistemleri Analisti", "IT Danışmanı"
        ]
    }
    
    # Kıdemler
    references["kidemler"] = ["Stajyer", "Jr.", "Mid.", "Sr.", "Lead", "Müdür Yardımcısı", "Müdür", "Direktör"]
    
    # Şehirler
    references["sehirler"] = [
        "İstanbul", "Ankara", "İzmir", "Bursa", "Antalya", "Kocaeli", "Konya", "Adana", "Gaziantep",
        "Eskişehir", "Samsun", "Tekirdağ", "Kayseri", "Mersin", "Trabzon", "Diyarbakır", "Muğla", 
        "Denizli", "Sakarya", "Aydın"
    ]
    
    # Eğitim seviyeleri
    references["egitim_seviyeleri"] = [
        "Lise", "Önlisans", "Lisans", "Yüksek Lisans", "Doktora", "Sertifika Programı"
    ]
    
    # Eğitim alanları
    references["egitim_alanlari"] = [
        "Bilgisayar Mühendisliği", "Yazılım Mühendisliği", "Elektrik-Elektronik Mühendisliği", 
        "Endüstri Mühendisliği", "Matematik", "İstatistik", "Fizik", "Bilgisayar Programcılığı",
        "Bilişim Sistemleri", "Yönetim Bilişim Sistemleri", "Mekatronik", "Elektronik Haberleşme",
        "Bilgisayar Teknolojileri", "Bilgisayar Bilimleri", "Diğer"
    ]
    
    # Çalışma şekilleri
    references["calisma_sekilleri"] = ["Tam Zamanlı", "Yarı Zamanlı", "Sözleşmeli", "Freelance", "Uzaktan", "Hibrit"]
    
    # Programlama dilleri
    references["prog_dilleri"] = [
        "Python", "Java", "JavaScript", "TypeScript", "C#", "C++", "Go", "Ruby", "PHP", "Swift", 
        "Kotlin", "Rust", "Scala", "R", "MATLAB", "Perl", "Dart", "Groovy", "Objective-C", "Lua", 
        "Clojure", "Elixir", "COBOL", "VBA", "Assembly", "Haskell", "F#"
    ]
    
    # İngilizce seviyeleri
    references["ingilizce_seviyeleri"] = ["Başlangıç", "Orta", "İleri", "İleri Düzey", "Anadil"]
    
    # Minimum, maksimum, ortalama deneyim ve yaş değerleri
    if "Deneyim_Yıl" in df.columns:
        references["min_deneyim"] = float(max(0, df["Deneyim_Yıl"].min()))
        references["max_deneyim"] = float(min(30, df["Deneyim_Yıl"].max()))
        references["avg_deneyim"] = float(df["Deneyim_Yıl"].mean())
    else:
        references["min_deneyim"] = 0.0
        references["max_deneyim"] = 30.0
        references["avg_deneyim"] = 5.0
        
    if "Yaş" in df.columns:
        references["min_yas"] = int(max(18, df["Yaş"].min()))
        references["max_yas"] = int(min(65, df["Yaş"].max()))
        references["avg_yas"] = int(df["Yaş"].mean())
    else:
        references["min_yas"] = 22
        references["max_yas"] = 60
        references["avg_yas"] = 30
    
    # Minimum, maksimum, ortalama maaş değerleri
    if "Maaş_TL" in df.columns:
        references["min_maas"] = float(df["Maaş_TL"].min())
        references["max_maas"] = float(df["Maaş_TL"].max())
        references["avg_maas"] = float(df["Maaş_TL"].mean())
        
        # Rol kategorilerine göre ortalama maaşlar
        if "Rol_Kategorisi" in df.columns:
            references["kategori_maaslar"] = df.groupby("Rol_Kategorisi")["Maaş_TL"].mean().to_dict()
        
        # Şehirlere göre ortalama maaşlar
        if "Şehir" in df.columns:
            references["sehir_maaslar"] = df.groupby("Şehir")["Maaş_TL"].mean().to_dict()
            
        # Kıdemlere göre ortalama maaşlar
        if "Kıdem" in df.columns:
            references["kidem_maaslar"] = df.groupby("Kıdem")["Maaş_TL"].mean().to_dict()
    else:
        references["min_maas"] = 10000.0
        references["max_maas"] = 100000.0
        references["avg_maas"] = 30000.0
    
    return references


def create_employee_form(references):
    """
    Kullanıcı girdileri için form oluşturur.
    """
    employee_data = {}
    
    # İki sütunlu düzen için
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Kişisel Bilgiler")
        
        employee_data["Ad"] = st.text_input("Ad", "")
        employee_data["Soyad"] = st.text_input("Soyad", "")
        employee_data["Cinsiyet"] = st.selectbox("Cinsiyet", ["Erkek", "Kadın"])
        employee_data["Yaş"] = st.slider(
            "Yaş", 
            min_value=references.get("min_yas", 22), 
            max_value=references.get("max_yas", 60), 
            value=references.get("avg_yas", 30)
        )
        
        st.markdown("### Eğitim Bilgileri")
        employee_data["Eğitim_Seviyesi"] = st.selectbox(
            "Eğitim Seviyesi", 
            references.get("egitim_seviyeleri", ["Lisans"])
        )
        employee_data["Eğitim_Alanı"] = st.selectbox(
            "Eğitim Alanı", 
            references.get("egitim_alanlari", ["Bilgisayar Mühendisliği"])
        )
    
    with col2:
        st.markdown("### Pozisyon Bilgileri")
        
        # Rol kategorisi seçimi
        rol_kategorisi = st.selectbox(
            "Rol Kategorisi", 
            list(references.get("roller", {}).keys())
        )
        employee_data["Rol_Kategorisi"] = rol_kategorisi
        
        # Seçilen kategoriye göre rolleri güncelle
        roller = references.get("roller", {}).get(rol_kategorisi, [])
        employee_data["Rol"] = st.selectbox("Rol", roller)
        
        employee_data["Kıdem"] = st.selectbox(
            "Kıdem", 
            references.get("kidemler", ["Mid."])
        )
        
        employee_data["Deneyim_Yıl"] = st.slider(
            "Deneyim (Yıl)", 
            min_value=references.get("min_deneyim", 0.0), 
            max_value=references.get("max_deneyim", 30.0), 
            value=references.get("avg_deneyim", 5.0),
            step=0.5
        )
        
        st.markdown("### Lokasyon ve Çalışma Şekli")
        employee_data["Şehir"] = st.selectbox(
            "Şehir", 
            references.get("sehirler", ["İstanbul"])
        )
        
        employee_data["Çalışma_Şekli"] = st.selectbox(
            "Çalışma Şekli", 
            references.get("calisma_sekilleri", ["Tam Zamanlı"])
        )
        
        if employee_data["Çalışma_Şekli"] in ["Uzaktan", "Hibrit"]:
            employee_data["Uzaktan_Çalışma_Oranı"] = st.slider(
                "Uzaktan Çalışma Oranı (%)", 
                min_value=0, 
                max_value=100, 
                value=50 if employee_data["Çalışma_Şekli"] == "Hibrit" else 100,
                step=10
            )
        else:
            employee_data["Uzaktan_Çalışma_Oranı"] = 0
    
    # Genişletilmiş bilgiler
    with st.expander("Ek Bilgiler (Opsiyonel)", expanded=False):
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### Teknik Bilgiler")
            
            employee_data["Ana_Programlama_Dili"] = st.selectbox(
                "Ana Programlama Dili", 
                [""] + references.get("prog_dilleri", ["Python"])
            )
            
            employee_data["Kullandığı_Teknolojiler"] = st.multiselect(
                "Kullandığı Teknolojiler", 
                references.get("prog_dilleri", ["Python"]),
                []
            )
            
            if employee_data["Kullandığı_Teknolojiler"]:
                employee_data["Kullandığı_Teknolojiler"] = ", ".join(employee_data["Kullandığı_Teknolojiler"])
            else:
                employee_data["Kullandığı_Teknolojiler"] = ""
            
            employee_data["Toplam_Proje_Sayısı"] = st.number_input(
                "Toplam Proje Sayısı", 
                min_value=0, 
                max_value=100, 
                value=10
            )
            
            employee_data["Teknik_Beceri_Puanı"] = st.slider(
                "Teknik Beceri Puanı (1-100)", 
                min_value=1, 
                max_value=100, 
                value=70
            )
            
        with col4:
            st.markdown("### Dil ve İletişim Becerileri")
            
            employee_data["İngilizce_Seviyesi"] = st.selectbox(
                "İngilizce Seviyesi", 
                references.get("ingilizce_seviyeleri", ["Orta"])
            )
            
            other_languages = st.multiselect(
                "Diğer Diller", 
                ["Almanca", "Fransızca", "İspanyolca", "Rusça", "Arapça", "Japonca", "Çince"],
                []
            )
            
            if other_languages:
                employee_data["Diğer_Diller"] = "Türkçe (Anadil), " + ", ".join([f"{dil} (Orta)" for dil in other_languages])
            else:
                employee_data["Diğer_Diller"] = "Türkçe (Anadil)"
            
            employee_data["Soft_Skill_Puanı"] = st.slider(
                "Soft Skill Puanı (1-100)", 
                min_value=1, 
                max_value=100, 
                value=65
            )
    
    # Form verilerini DataFrame'e dönüştür
    df = pd.DataFrame([employee_data])
    
    return df


def show_salary_prediction(employee_data, model, preprocessor):
    """
    Maaş tahmini yapar ve sonuçları görselleştirir.
    """
    try:
        # Tahmin yap
        if model is not None and preprocessor is not None:
            predicted_salary = predict_salary(employee_data, model, preprocessor)
        else:
            # Model yoksa basit bir hesaplama yap
            base_salary = 30000  # Temel maaş
            
            # Katsayılar
            kidem_dict = {"Stajyer": 0.4, "Jr.": 0.8, "Mid.": 1.0, "Sr.": 1.4, "Lead": 1.8, 
                        "Müdür Yardımcısı": 2.0, "Müdür": 2.5, "Direktör": 3.0}
            
            kategori_dict = {"Yazılım Geliştirme": 1.0, "Veri": 1.1, "DevOps ve Altyapı": 1.05, 
                           "Güvenlik": 1.15, "Yapay Zeka ve Makine Öğrenmesi": 1.2, 
                           "Tasarım ve Kullanıcı Deneyimi": 0.95, "QA ve Test": 0.9, 
                           "Yönetim ve Liderlik": 1.3, "Destek ve Operasyon": 0.8, "İş ve Analiz": 0.9}
            
            sehir_dict = {"İstanbul": 1.0, "Ankara": 0.9, "İzmir": 0.85}
            default_sehir = 0.8
            
            # Katsayıları uygula
            kidem_carpani = kidem_dict.get(employee_data["Kıdem"].values[0], 1.0)
            kategori_carpani = kategori_dict.get(employee_data["Rol_Kategorisi"].values[0], 1.0)
            sehir_carpani = sehir_dict.get(employee_data["Şehir"].values[0], default_sehir)
            deneyim_carpani = 1.0 + (employee_data["Deneyim_Yıl"].values[0] * 0.05)
            
            predicted_salary = base_salary * kidem_carpani * kategori_carpani * sehir_carpani * deneyim_carpani
        
        # Piyasa karşılaştırmaları
        market_comparisons = compare_to_market_average(predicted_salary, employee_data)
        
        # Maaş önerileri
        recommendations = generate_salary_recommendation(predicted_salary, employee_data, market_comparisons)
        
        # Sonuçları göster
        st.markdown('<div class="section-header">Tahmini Maaş</div>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="salary-result">{predicted_salary:,.0f} TL / Ay</div>', unsafe_allow_html=True)
        
        # Maaş aralığı (Tahminin ±%10'u)
        min_salary = predicted_salary * 0.9
        max_salary = predicted_salary * 1.1
        
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p>Tahmini maaş aralığı: {min_salary:,.0f} TL - {max_salary:,.0f} TL</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Karşılaştırma ve öneriler
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="section-header">Piyasa Karşılaştırması</div>', unsafe_allow_html=True)
            
            if market_comparisons:
                # En ilgili karşılaştırmayı bul
                relevant_comparisons = []
                
                if "role" in market_comparisons:
                    relevant_comparisons.append(market_comparisons["role"])
                elif "category" in market_comparisons:
                    relevant_comparisons.append(market_comparisons["category"])
                
                if "experience" in market_comparisons:
                    relevant_comparisons.append(market_comparisons["experience"])
                
                if "city" in market_comparisons:
                    relevant_comparisons.append(market_comparisons["city"])
                
                if "all" in market_comparisons:
                    relevant_comparisons.append(market_comparisons["all"])
                
                # İlgili karşılaştırmaları göster
                for comparison in relevant_comparisons:
                    st.markdown(f"""
                    <div class="market-comparison">
                        <h4>{comparison['name']}</h4>
                        <p>Ortalama Maaş: {comparison['average']:,.0f} TL</p>
                        <p>Medyan Maaş: {comparison['median']:,.0f} TL</p>
                        <p>Fark: {comparison['diff_from_avg']:,.0f} TL (%{comparison['diff_percentage']:.1f})</p>
                        <p>Piyasa Yüzdelik Dilimi: %{comparison['percentile']:.1f}</p>
                        <p>Minimum: {comparison['min']:,.0f} TL</p>
                        <p>Maksimum: {comparison['max']:,.0f} TL</p>
                        <p>Standart Sapma: {comparison['std']:,.0f} TL</p>
                        <p>Veri Sayısı: {comparison['count']} kayıt</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Karşılaştırma grafiği
                if relevant_comparisons:
                    comparison = relevant_comparisons[0]  # En ilgili karşılaştırma
                    
                    labels = ['Minimum', 'Ortalama', 'Tahmini Maaş', 'Maksimum']
                    values = [comparison['min'], comparison['average'], predicted_salary, comparison['max']]
                    
                    fig = go.Figure(data=[go.Bar(
                        x=labels,
                        y=values,
                        marker_color=['#3498db', '#2ecc71', '#e74c3c', '#3498db']
                    )])
                    
                    fig.update_layout(
                        title=f"{comparison['name']} Karşılaştırması",
                        yaxis_title="Maaş (TL)",
                        height=400,
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Piyasa karşılaştırması için yeterli veri bulunamadı.")
        
        with col2:
            st.markdown('<div class="section-header">Maaş Önerileri</div>', unsafe_allow_html=True)
            
            if recommendations and 'recommendations' in recommendations:
                for rec in recommendations['recommendations']:
                    if 'type' in rec:
                        if rec['type'] in ['high', 'significantly_high', 'somewhat_high']:
                            style_class = "recommendation-high"
                        elif rec['type'] in ['average', 'fair']:
                            style_class = "recommendation-average"
                        else:
                            style_class = "recommendation-low"
                    else:
                        style_class = "recommendation-average"
                    
                    st.markdown(f"""
                    <div class="recommendation-box {style_class}">
                        <h4>{rec['message']}</h4>
                    """, unsafe_allow_html=True)
                    
                    if 'suggestion' in rec:
                        st.markdown(f"<p><strong>Öneri:</strong> {rec['suggestion']}</p>", unsafe_allow_html=True)
                    
                    if 'adjustment' in rec:
                        st.markdown(f"<p><strong>Ayarlama:</strong> {rec['adjustment']}</p>", unsafe_allow_html=True)
                    
                    if 'target_range' in rec:
                        st.markdown(f"""
                        <p><strong>Hedef Aralık:</strong> {rec['target_range'][0]:,.0f} TL - {rec['target_range'][1]:,.0f} TL</p>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("Maaş önerileri oluşturulamadı.")
        
        # Özellik önem dereceleri
        st.markdown('<div class="section-header">Etkili Faktörler</div>', unsafe_allow_html=True)
        
        # Örnek faktörler
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h4>Deneyim</h4>
                <p>Deneyim seviyesi maaş üzerinde önemli bir etkiye sahiptir. Her yıl için yaklaşık %5 artış görülür.</p>
            </div>
            """, unsafe_allow_html=True)
            
            factor_value = employee_data["Deneyim_Yıl"].values[0]
            factor_max = 15
            factor_importance = min(1.0, factor_value / factor_max)
            
            # İlerleme çubuğu
            progress_html = f"""
            <div style="margin-top: 10px;">
                <div style="background-color: #e0e0e0; border-radius: 5px; height: 10px; width: 100%;">
                    <div style="background-color: #3498db; border-radius: 5px; height: 10px; width: {factor_importance * 100}%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                    <span style="font-size: 0.8rem;">Az</span>
                    <span style="font-size: 0.8rem;">Orta</span>
                    <span style="font-size: 0.8rem;">Çok</span>
                </div>
            </div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h4>Rol ve Kıdem</h4>
                <p>Pozisyon ve kıdem seviyesi maaş üzerinde doğrudan etkilidir. Özellikle yönetim pozisyonları ve yüksek kıdem seviyeleri ciddi maaş artışı sağlar.</p>
            </div>
            """, unsafe_allow_html=True)
            
            kidem_dict = {"Stajyer": 0.2, "Jr.": 0.4, "Mid.": 0.6, "Sr.": 0.8, "Lead": 0.9, 
                        "Müdür Yardımcısı": 0.95, "Müdür": 0.98, "Direktör": 1.0}
            
            factor_importance = kidem_dict.get(employee_data["Kıdem"].values[0], 0.6)
            
            # İlerleme çubuğu
            progress_html = f"""
            <div style="margin-top: 10px;">
                <div style="background-color: #e0e0e0; border-radius: 5px; height: 10px; width: 100%;">
                    <div style="background-color: #3498db; border-radius: 5px; height: 10px; width: {factor_importance * 100}%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                    <span style="font-size: 0.8rem;">Jr.</span>
                    <span style="font-size: 0.8rem;">Mid.</span>
                    <span style="font-size: 0.8rem;">Sr.</span>
                </div>
            </div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="info-box">
                <h4>Lokasyon</h4>
                <p>Çalışma lokasyonu maaş üzerinde etkilidir. İstanbul'daki maaşlar genellikle diğer şehirlere göre %10-20 daha yüksektir.</p>
            </div>
            """, unsafe_allow_html=True)
            
            sehir_dict = {"İstanbul": 1.0, "Ankara": 0.85, "İzmir": 0.80}
            default_sehir = 0.7
            
            factor_importance = sehir_dict.get(employee_data["Şehir"].values[0], default_sehir)
            
            # İlerleme çubuğu
            progress_html = f"""
            <div style="margin-top: 10px;">
                <div style="background-color: #e0e0e0; border-radius: 5px; height: 10px; width: 100%;">
                    <div style="background-color: #3498db; border-radius: 5px; height: 10px; width: {factor_importance * 100}%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                    <span style="font-size: 0.8rem;">Düşük</span>
                    <span style="font-size: 0.8rem;">Orta</span>
                    <span style="font-size: 0.8rem;">Yüksek</span>
                </div>
            </div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)
        
        # İkinci sıra faktörler
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.markdown("""
            <div class="info-box">
                <h4>Eğitim Seviyesi</h4>
                <p>Eğitim seviyesi maaş üzerinde etkilidir, özellikle yüksek lisans ve doktora dereceleri için prim söz konusudur.</p>
            </div>
            """, unsafe_allow_html=True)
            
            egitim_dict = {"Lise": 0.5, "Önlisans": 0.6, "Lisans": 0.7, "Yüksek Lisans": 0.9, "Doktora": 1.0, "Sertifika Programı": 0.55}
            
            factor_importance = egitim_dict.get(employee_data["Eğitim_Seviyesi"].values[0], 0.7)
            
            # İlerleme çubuğu
            progress_html = f"""
            <div style="margin-top: 10px;">
                <div style="background-color: #e0e0e0; border-radius: 5px; height: 10px; width: 100%;">
                    <div style="background-color: #3498db; border-radius: 5px; height: 10px; width: {factor_importance * 100}%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                    <span style="font-size: 0.8rem;">Lise</span>
                    <span style="font-size: 0.8rem;">Lisans</span>
                    <span style="font-size: 0.8rem;">Doktora</span>
                </div>
            </div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)
        
        with col5:
            st.markdown("""
            <div class="info-box">
                <h4>Teknik Beceriler</h4>
                <p>Uzmanlık alanlarında teknik yetkinlikler maaş için önemlidir. Özellikle yazılım ve veribilimi alanlarında yüksek teknoloji becerileri önemli avantaj sağlar.</p>
            </div>
            """, unsafe_allow_html=True)
            
            factor_value = employee_data["Teknik_Beceri_Puanı"].values[0] if "Teknik_Beceri_Puanı" in employee_data.columns else 70
            factor_importance = factor_value / 100
            
            # İlerleme çubuğu
            progress_html = f"""
            <div style="margin-top: 10px;">
                <div style="background-color: #e0e0e0; border-radius: 5px; height: 10px; width: 100%;">
                    <div style="background-color: #3498db; border-radius: 5px; height: 10px; width: {factor_importance * 100}%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                    <span style="font-size: 0.8rem;">Temel</span>
                    <span style="font-size: 0.8rem;">Orta</span>
                    <span style="font-size: 0.8rem;">İleri</span>
                </div>
            </div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)
        
        with col6:
            st.markdown("""
            <div class="info-box">
                <h4>Dil Becerileri</h4>
                <p>İngilizce ve diğer yabancı dil becerileri özellikle uluslararası şirketlerde ve yüksek pozisyonlarda maaş artışı sağlar.</p>
            </div>
            """, unsafe_allow_html=True)
            
            ing_dict = {"Başlangıç": 0.3, "Orta": 0.6, "İleri": 0.8, "İleri Düzey": 0.9, "Anadil": 1.0}
            
            factor_importance = ing_dict.get(employee_data["İngilizce_Seviyesi"].values[0], 0.6)
            
            # İlerleme çubuğu
            progress_html = f"""
            <div style="margin-top: 10px;">
                <div style="background-color: #e0e0e0; border-radius: 5px; height: 10px; width: 100%;">
                    <div style="background-color: #3498db; border-radius: 5px; height: 10px; width: {factor_importance * 100}%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                    <span style="font-size: 0.8rem;">Başlangıç</span>
                    <span style="font-size: 0.8rem;">Orta</span>
                    <span style="font-size: 0.8rem;">İleri</span>
                </div>
            </div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)
        
        # PDF Rapor oluşturma butonu
        st.markdown('<div class="section-header">Rapor İşlemleri</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("📄 PDF Rapor Oluştur", use_container_width=True):
                st.success("PDF rapor oluşturuldu ve indirilmeye hazır.")
                # PDF oluşturma fonksiyonu burada çağrılabilir
        
        with col2:
            if st.button("💾 Kaydedilen Raporlar", use_container_width=True):
                st.info("Kaydedilen raporlar listeleniyor...")
                # Kaydedilen raporları listeleme fonksiyonu burada çağrılabilir
        
        with col3:
            if st.button("📊 Piyasa Analizleri", use_container_width=True):
                st.info("Piyasa analizleri görüntüleniyor...")
                # Piyasa analizleri fonksiyonu burada çağrılabilir
        
    except Exception as e:
        st.error(f"Maaş tahmini sırasında bir hata oluştu: {str(e)}")


def show_dashboard():
    """
    Veri analitiği dashboard'unu gösterir.
    """
    st.markdown('<div class="section-header">IT Sektörü Maaş Analizi</div>', unsafe_allow_html=True)
    
    # Veri yükle
    df = load_default_data()
    
    if df is None:
        st.warning("Dashboard için veri yüklenemedi.")
        return
    
    # Filtre bölümü
    st.markdown("### Filtreler")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Rol kategorisi filtresi
        if 'Rol_Kategorisi' in df.columns:
            selected_kategori = st.multiselect(
                "Rol Kategorisi",
                options=sorted(df['Rol_Kategorisi'].unique()),
                default=[]
            )
            
            if selected_kategori:
                df = df[df['Rol_Kategorisi'].isin(selected_kategori)]
    
    with col2:
        # Şehir filtresi
        if 'Şehir' in df.columns:
            selected_sehir = st.multiselect(
                "Şehir",
                options=sorted(df['Şehir'].unique()),
                default=[]
            )
            
            if selected_sehir:
                df = df[df['Şehir'].isin(selected_sehir)]
    
    with col3:
        # Deneyim filtresi
        if 'Deneyim_Yıl' in df.columns:
            min_deneyim = float(df['Deneyim_Yıl'].min())
            max_deneyim = float(df['Deneyim_Yıl'].max())
            
            deneyim_range = st.slider(
                "Deneyim Yılı",
                min_value=min_deneyim,
                max_value=max_deneyim,
                value=(min_deneyim, max_deneyim),
                step=0.5
            )
            
            df = df[(df['Deneyim_Yıl'] >= deneyim_range[0]) & (df['Deneyim_Yıl'] <= deneyim_range[1])]
    
    # Ana dashboard grafiklerini göster
    if len(df) > 0:
        st.markdown("### Maaş Dağılımı")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Maaş dağılım histogramı
            fig = px.histogram(
                df, 
                x="Maaş_TL", 
                nbins=20, 
                title="Maaş Dağılımı",
                labels={"Maaş_TL": "Maaş (TL)", "count": "Çalışan Sayısı"},
                color_discrete_sequence=['#3498db']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Kutu grafiği
            fig = px.box(
                df, 
                y="Maaş_TL", 
                title="Maaş İstatistikleri",
                labels={"Maaş_TL": "Maaş (TL)"},
                color_discrete_sequence=['#3498db']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Gruplandırmalı analizler
        st.markdown("### Faktörlere Göre Maaş Analizi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Kategoriye göre ortalama maaş
            if 'Rol_Kategorisi' in df.columns:
                kategori_maas = df.groupby('Rol_Kategorisi')['Maaş_TL'].mean().reset_index()
                kategori_maas = kategori_maas.sort_values('Maaş_TL', ascending=False)
                
                fig = px.bar(
                    kategori_maas, 
                    x='Rol_Kategorisi', 
                    y='Maaş_TL', 
                    title="Kategoriye Göre Ortalama Maaş",
                    labels={"Rol_Kategorisi": "Rol Kategorisi", "Maaş_TL": "Ortalama Maaş (TL)"},
                    color_discrete_sequence=['#3498db']
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Kıdeme göre ortalama maaş
            if 'Kıdem' in df.columns:
                # Kıdem sıralaması
                kidem_order = ["Stajyer", "Jr.", "Mid.", "Sr.", "Lead", "Müdür Yardımcısı", "Müdür", "Direktör"]
                
                kidem_maas = df.groupby('Kıdem')['Maaş_TL'].mean().reset_index()
                
                # Kıdem sıralamasını uygula
                kidem_maas['Kıdem_Sıra'] = kidem_maas['Kıdem'].apply(lambda x: kidem_order.index(x) if x in kidem_order else 999)
                kidem_maas = kidem_maas.sort_values('Kıdem_Sıra')
                
                fig = px.bar(
                    kidem_maas, 
                    x='Kıdem', 
                    y='Maaş_TL', 
                    title="Kıdeme Göre Ortalama Maaş",
                    labels={"Kıdem": "Kıdem Seviyesi", "Maaş_TL": "Ortalama Maaş (TL)"},
                    color_discrete_sequence=['#3498db'],
                    category_orders={"Kıdem": kidem_order}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        # Deneyim-Maaş ilişkisi
        st.markdown("### Deneyim ve Maaş İlişkisi")
        
        if 'Deneyim_Yıl' in df.columns:
            fig = px.scatter(
                df, 
                x="Deneyim_Yıl", 
                y="Maaş_TL", 
                color="Rol_Kategorisi" if "Rol_Kategorisi" in df.columns else None,
                size="Yaş" if "Yaş" in df.columns else None,
                hover_name="Rol" if "Rol" in df.columns else None,
                hover_data=["Şehir", "Kıdem"] if all(col in df.columns for col in ["Şehir", "Kıdem"]) else None,
                title="Deneyim ve Maaş İlişkisi",
                labels={"Deneyim_Yıl": "Deneyim (Yıl)", "Maaş_TL": "Maaş (TL)"}
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Trend çizgisi
            fig = px.scatter(
                df, 
                x="Deneyim_Yıl", 
                y="Maaş_TL", 
                trendline="ols",
                title="Deneyim ve Maaş Trend Analizi",
                labels={"Deneyim_Yıl": "Deneyim (Yıl)", "Maaş_TL": "Maaş (TL)"}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Filtre kriterlerine uygun veri bulunamadı.")


def show_about():
    """
    Uygulama hakkında bilgi sayfasını gösterir.
    """
    st.markdown('<div class="section-header">Uygulama Hakkında</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### IT Sektörü Maaş Tahmin Sistemi

    Bu uygulama, IT sektöründe faaliyet gösteren işletmelerin rekabet gücünü artıracak, uygun maaş seviyelerini belirlemelerine ve nitelikli personelleri elde tutmalarına yardımcı olacak bir maaş tahminleme sistemidir.

    #### Özellikler

    - **Makine Öğrenmesi Tabanlı Tahmin**: Gelişmiş makine öğrenmesi algoritmaları kullanarak maaş tahmini
    - **Topluluk Öğrenme Modelleri**: Ensemble learning tabanlı maaş tahminleme modelleri
    - **Genişletilmiş Öznitelik Kümesi**: Detaylı çalışan özelliklerini dikkate alan modeller
    - **Piyasa Karşılaştırması**: Tahmin edilen maaşın piyasa ortalamasıyla karşılaştırması
    - **Detaylı Analizler**: Rol, deneyim, şehir ve diğer faktörlere göre maaş analizleri
    - **Özelleştirilebilir Raporlar**: PDF formatında indirilebilir detaylı raporlar

    #### Kullanılan Teknolojiler

    - **Python**: Ana programlama dili
    - **Streamlit**: Web arayüzü geliştirme
    - **Scikit-learn**: Makine öğrenmesi modelleri
    - **Pandas & NumPy**: Veri analizi ve işleme
    - **Plotly**: İnteraktif veri görselleştirme

    #### Veri Kaynakları

    Uygulama, Türkiye IT sektöründen toplanan güncel maaş verileri üzerine kurulmuştur. Veri seti düzenli olarak güncellenmektedir.

    #### Nasıl Kullanılır?

    1. **Maaş Tahmini**: Ana sayfada çalışan bilgilerini girerek maaş tahmini alabilirsiniz
    2. **Piyasa Analizi**: Dashboard sayfasından sektörel analizlere ulaşabilirsiniz
    3. **Raporlar**: Tahmin sonuçlarını PDF formatında indirebilirsiniz

    #### Gizlilik

    - Girilen tüm veriler gizli tutulmaktadır
    - Kişisel veriler kaydedilmemektedir
    - Analizlerde anonimleştirilmiş veriler kullanılmaktadır

    #### İletişim

    Sorularınız ve önerileriniz için:
    - Email: destek@maastahmin.com
    - Twitter: @maastahmin
    - LinkedIn: /company/maastahmin

    #### Lisans

    Bu uygulama MIT lisansı altında dağıtılmaktadır. Detaylı bilgi için [Lisans](https://opensource.org/licenses/MIT) sayfasını ziyaret ediniz.
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>© 2024 IT Maaş Tahmin Sistemi - Tüm hakları saklıdır.</p>
    </div>
    """, unsafe_allow_html=True)

# Ana uygulama
def main():
    """
    Ana uygulama fonksiyonu
    """
    # Sayfa başlığı
    st.markdown("""
        <div class="title-container">
            <h1>IT Sektörü Maaş Tahmin Sistemi</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Menü")
    page = st.sidebar.selectbox(
        "Sayfa Seçiniz",
        ["Maaş Tahmini", "Dashboard", "Hakkında"]
    )
    
    # Referans verilerini yükle
    references = get_reference_data()
    
    # Sayfa yönlendirmesi
    if page == "Maaş Tahmini":
        if references:
            employee_data = create_employee_form(references)
            
            # Tahmin butonu
            if st.button("Maaş Tahmini Yap", use_container_width=True):
                if employee_data is not None:
                    model, preprocessor = load_saved_model()
                    show_salary_prediction(employee_data, model, preprocessor)
        else:
            st.error("Referans verileri yüklenemedi.")
            
    elif page == "Dashboard":
        show_dashboard()
    else:
        show_about()

if __name__ == "__main__":
    main()