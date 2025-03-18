import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import re

# Faker'ı Türkçe dilini destekleyecek şekilde ayarlayalım
fake = Faker(['tr_TR'])
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# Kapsamlı rol ve kıdem bilgilerini tanımlayalım
roller = {
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

# Roller için kıdem seviyeleri
kidem_seviyeleri = ["Stajyer", "Jr.", "Mid.", "Sr.", "Lead", "Müdür Yardımcısı", "Müdür", "Direktör"]
kidem_seviyesi_yil_araliklari = {
    "Stajyer": (0, 0.5),
    "Jr.": (0.5, 2),
    "Mid.": (2, 5),
    "Sr.": (5, 8),
    "Lead": (8, 12),
    "Müdür Yardımcısı": (10, 15),
    "Müdür": (12, 20),
    "Direktör": (15, 30)
}

# Şehirler ve teknoloji kümelenmesi
turkiye_sehirleri = [
    ("İstanbul", 45),  # IT sektörünün %45'i İstanbul'da
    ("Ankara", 20),
    ("İzmir", 10),  
    ("Bursa", 3),
    ("Antalya", 3),
    ("Kocaeli", 3),
    ("Konya", 2),
    ("Adana", 2),
    ("Gaziantep", 1),
    ("Eskişehir", 1),
    ("Samsun", 1),
    ("Tekirdağ", 1),
    ("Kayseri", 1),
    ("Mersin", 1),
    ("Trabzon", 1),
    ("Diyarbakır", 1),
    ("Muğla", 1),
    ("Denizli", 1),
    ("Sakarya", 1),
    ("Aydın", 1)
]

# İlçeler - şehirlere göre
turkiye_ilceler = {
    "İstanbul": ["Kadıköy", "Beşiktaş", "Şişli", "Ataşehir", "Beyoğlu", "Bakırköy", "Maltepe", "Üsküdar", "Sarıyer", "Levent", "Maslak"],
    "Ankara": ["Çankaya", "Yenimahalle", "Keçiören", "Etimesgut", "Gölbaşı", "Mamak", "Sincan", "Altındağ"],
    "İzmir": ["Bornova", "Konak", "Karşıyaka", "Bayraklı", "Çiğli", "Karabağlar", "Buca", "Gaziemir"],
    "Bursa": ["Nilüfer", "Osmangazi", "Yıldırım", "Mudanya", "Gürsu"],
    "Antalya": ["Muratpaşa", "Konyaaltı", "Kepez", "Lara", "Döşemealtı"],
    "Kocaeli": ["İzmit", "Gebze", "Darıca", "Çayırova", "Derince"],
}
# Diğer şehirler için varsayılan ilçeler
default_ilceler = ["Merkez", "Yeni Mahalle", "Atatürk", "Cumhuriyet", "Fatih", "Yıldız", "Bahçelievler"]

cinsiyet = [("Erkek", 70), ("Kadın", 30)]  # IT sektöründeki cinsiyet dağılımı yaklaşık olarak

egitim_seviyeleri = [
    "Lise", "Önlisans", "Lisans", "Yüksek Lisans", "Doktora", "Sertifika Programı"
]

# Çalışma durumları
calisma_durumlari = ["Tam Zamanlı", "Yarı Zamanlı", "Sözleşmeli", "Freelance", "Uzaktan", "Hibrit"]

# Eğitim alanları
egitim_alanlari = [
    "Bilgisayar Mühendisliği", "Yazılım Mühendisliği", "Elektrik-Elektronik Mühendisliği", 
    "Endüstri Mühendisliği", "Matematik", "İstatistik", "Fizik", "Bilgisayar Programcılığı",
    "Bilişim Sistemleri", "Yönetim Bilişim Sistemleri", "Mekatronik", "Elektronik Haberleşme",
    "Bilgisayar Teknolojileri", "Bilgisayar Bilimleri", "Diğer"
]

# Şirket büyüklükleri
sirket_buyuklukleri = [
    "Startup (1-10)", "Küçük (11-50)", "Orta (51-200)", "Büyük (201-1000)", "Kurumsal (1000+)"
]

# Diller
konusulan_diller = [
    ("Türkçe", 100), 
    ("İngilizce", 80), 
    ("Almanca", 10), 
    ("Fransızca", 8), 
    ("İspanyolca", 7), 
    ("Rusça", 5), 
    ("Arapça", 5),
    ("Japonca", 3), 
    ("Çince", 3), 
    ("Korece", 2)
]

# Dil seviyeleri
dil_seviyeleri = ["Başlangıç", "Orta", "İleri", "İleri Düzey", "Anadil"]

# Role göre teknolojiler
teknolojiler = {
    "Programlama Dilleri": [
        "Python", "Java", "JavaScript", "TypeScript", "C#", "C++", "Go", "Ruby", "PHP", "Swift", 
        "Kotlin", "Rust", "Scala", "R", "MATLAB", "Perl", "Dart", "Groovy", "Objective-C", "Lua", 
        "Clojure", "Elixir", "COBOL", "VBA", "Assembly", "Haskell", "F#"
    ],
    "Web Frontend": [
        "HTML", "CSS", "JavaScript", "TypeScript", "React", "Vue.js", "Angular", "Svelte", "jQuery", 
        "Bootstrap", "Tailwind CSS", "Material UI", "Redux", "Next.js", "Nuxt.js", "Ember.js", 
        "Webpack", "Babel", "Sass/SCSS", "Less", "GraphQL", "Apollo Client", "PWA", "WebAssembly"
    ],
    "Web Backend": [
        "Node.js", "Express.js", "Django", "Flask", "Spring Boot", "Laravel", "ASP.NET Core", "Ruby on Rails", 
        "FastAPI", "NestJS", "Phoenix", "Play Framework", "Symfony", "CodeIgniter", "Gin", "Echo", 
        "Fiber", "Strapi", "Adonis.js", "Koa.js", "Deno"
    ],
    "Mobil": [
        "Android", "iOS", "React Native", "Flutter", "Xamarin", "Ionic", "Kotlin", "Swift", "Objective-C", 
        "Java (Android)", "Cordova", "PhoneGap", "Native Script", "SwiftUI", "Jetpack Compose", "Kotlin Multiplatform"
    ],
    "Veritabanları": [
        "MySQL", "PostgreSQL", "MongoDB", "SQL Server", "Oracle", "SQLite", "Redis", "Cassandra", 
        "DynamoDB", "Firebase", "Couchbase", "MariaDB", "ElasticSearch", "Neo4j", "InfluxDB", "CockroachDB", 
        "Firestore", "FaunaDB", "Supabase"
    ],
    "Bulut Platformlar": [
        "AWS", "Azure", "Google Cloud", "IBM Cloud", "Oracle Cloud", "DigitalOcean", "Heroku", "Alibaba Cloud", 
        "Tencent Cloud", "Linode", "Vultr", "Cloudflare", "OVH", "Scaleway", "Vercel", "Netlify"
    ],
    "DevOps Araçlar": [
        "Docker", "Kubernetes", "Jenkins", "Git", "Terraform", "Ansible", "Prometheus", "Grafana", "CircleCI", 
        "GitHub Actions", "GitLab CI/CD", "TeamCity", "ArgoCD", "Helm", "Puppet", "Chef", "SonarQube", "Istio", 
        "Vault", "ELK Stack", "Sentry"
    ],
    "Veri & AI": [
        "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy", "Apache Spark", "Hadoop", "NLTK", "OpenCV", 
        "Keras", "Dask", "Databricks", "Power BI", "Tableau", "Apache Kafka", "Luigi", "Airflow", "MLflow", 
        "Ray", "Hugging Face", "OpenAI API", "CUDA", "SageMaker", "Apache Beam"
    ],
    "Test Araçları": [
        "Selenium", "JUnit", "TestNG", "Cypress", "Playwright", "Jest", "Mocha", "Chai", "Cucumber", "Postman", 
        "SoapUI", "JMeter", "Gatling", "Appium", "Robot Framework", "PyTest", "PHPUnit", "xUnit", "Jasmine", 
        "Mockito", "WireMock"
    ],
    "Güvenlik Araçları": [
        "Burp Suite", "Nmap", "Metasploit", "Wireshark", "OWASP ZAP", "Kali Linux", "Snort", "Nessus", 
        "OpenVAS", "Aircrack-ng", "John the Ripper", "Hydra", "Splunk", "HashiCorp Vault", "CrowdStrike", 
        "Symantec", "Trend Micro", "SonarQube", "Veracode", "Checkmarx"
    ],
    "Tasarım Araçları": [
        "Figma", "Adobe XD", "Sketch", "InVision", "Zeplin", "Adobe Photoshop", "Adobe Illustrator", 
        "Axure RP", "Framer", "Balsamiq", "Principle", "ProtoPie", "Marvel", "UXPin", "Webflow"
    ]
}

# Şirketler listesi (sektörlerine göre)
sirketler = {
    "Büyük Teknoloji": [
        "Trendyol Tech", "Hepsiburada Tech", "Turkcell Teknoloji", "Vodafone Teknoloji", "Türk Telekom Teknoloji",
        "Yemeksepeti Tech", "Aselsan", "Havelsan", "Roketsan", "TÜBİTAK", "Getir Tech", "Migros Teknoloji",
        "Arçelik A.Ş.", "İş Bankası Teknoloji", "Akbank Teknoloji"
    ],
    "Yazılım Firmaları": [
        "Logo Yazılım", "Softtech", "Etiya", "Intertech", "Innova", "KoçSistem", "Cybersoft", "Netaş", 
        "Bilge Adam Teknoloji", "Veripark", "Insider", "Bizim Yazılım", "Akinon", "Testinium", "Siemens Türkiye"
    ],
    "Startuplar": [
        "Modanisa Tech", "Obilet Technology", "Armut.com", "Iyzico", "Martı Tech", "BiTaksi", "Evidea", 
        "Enpara Teknoloji", "Papara", "Paraşüt", "Sipay", "Teknasyon", "Mobiroller", "Apsiyon", "Tapu.com"
    ],
    "Şube/Danışmanlık": [
        "Deloitte Digital", "Accenture Türkiye", "IBM Türkiye", "Microsoft Türkiye", "Oracle Türkiye", 
        "Bimsa", "Ericsson Türkiye", "PwC Teknoloji", "HP Türkiye", "ThoughtWorks", "OBSS", "Lostar",
        "Huawei Türkiye", "Amazon Türkiye", "Hitachi Türkiye"
    ],
    "Savunma Sanayi": [
        "STM", "TUSAŞ", "BAYKAR", "FNSS", "ALTINAY Robot", "VESTEL Savunma", "C2TECH", "BİTES", 
        "MILSOFT", "METEKSAN Savunma", "KAREL", "SDT Uzay", "ROKETSAN Teknoloji", "TRtest", "ISBIR"
    ]
}

# Sektörler
sektorler = [
    "E-Ticaret", "Finans/Bankacılık", "Telekom", "Savunma Sanayi", "Sağlık", "Eğitim", "Perakende", 
    "Sigorta", "Otomotiv", "Üretim", "Enerji", "Turizm", "Lojistik", "Medya", "Kamu", "Danışmanlık", 
    "Yazılım", "Oyun", "İlaç", "Gıda", "Diğer"
]

# Popüler mesleki sertifikalar
mesleki_sertifikalar = {
    "Bulut": [
        "AWS Certified Solutions Architect", "AWS Certified Developer", "AWS Certified DevOps Engineer",
        "Microsoft Certified: Azure Administrator", "Microsoft Certified: Azure Developer", "Microsoft Certified: Azure Solutions Architect",
        "Google Cloud Professional Cloud Architect", "Google Cloud Professional Data Engineer", "Google Cloud Professional Cloud Developer",
        "Oracle Cloud Infrastructure Architect", "Oracle Cloud Infrastructure Developer"
    ],
    "Yazılım Geliştirme": [
        "Oracle Certified Professional, Java SE Programmer", "Oracle Certified Professional, Java EE Developer",
        "Microsoft Certified: .NET Developer", "Certified Kubernetes Developer (CKD)", "Certified Kubernetes Administrator (CKA)",
        "Zend Certified PHP Engineer", "Node.js Application Developer", "React Developer Certificate", "MongoDB Certified Developer",
        "Certified Spring Professional", "Professional Scrum Developer"
    ],
    "Proje Yönetimi": [
        "Project Management Professional (PMP)", "Certified ScrumMaster (CSM)", "Certified Scrum Product Owner (CSPO)",
        "PMI Agile Certified Practitioner (PMI-ACP)", "PRINCE2 Foundation/Practitioner", "Lean Six Sigma Green Belt/Black Belt",
        "Disciplined Agile Senior Scrum Master (DASSM)", "SAFe Agilist", "ITIL Foundation", "Professional Scrum Master (PSM)"
    ],
    "Veri": [
        "Microsoft Certified: Data Analyst Associate", "Microsoft Certified: Data Engineer", "Microsoft Certified: Data Scientist",
        "Certified Data Management Professional (CDMP)", "Cloudera Certified Associate Data Analyst", "MongoDB Certified DBA",
        "Databricks Certified Data Engineer", "Tableau Desktop Certified Professional", "Snowflake SnowPro Core Certification",
        "IBM Data Science Professional Certificate", "SAS Certified Data Scientist"
    ],
    "Ağ": [
        "Cisco Certified Network Associate (CCNA)", "Cisco Certified Network Professional (CCNP)", "Cisco Certified Internetwork Expert (CCIE)",
        "CompTIA Network+", "Juniper Networks Certified Internet Associate (JNCIA)", "Certified Network Professional", 
        "VMware Certified Network Virtualization", "Aruba Certified Switching Associate", "Fortinet Network Security Associate",
        "Alcatel-Lucent Network Routing Specialist"
    ],
    "Güvenlik": [
        "Certified Information Systems Security Professional (CISSP)", "Certified Ethical Hacker (CEH)", "Offensive Security Certified Professional (OSCP)",
        "CompTIA Security+", "Certified Information Security Manager (CISM)", "GIAC Security Essentials (GSEC)",
        "Certified Cloud Security Professional (CCSP)", "Systems Security Certified Practitioner (SSCP)", "Certified Information Systems Auditor (CISA)",  
        "CyberArk Certified Defender", "Palo Alto Networks Certified Cybersecurity Associate (PCCSA)"
    ],
    "Sistem": [
        "Red Hat Certified System Administrator (RHCSA)", "Red Hat Certified Engineer (RHCE)", "Microsoft Certified: Windows Server",
        "CompTIA Server+", "Linux Professional Institute Certification (LPIC)", "VMware Certified Professional - Data Center Virtualization",
        "Certified OpenStack Administrator", "Oracle Linux System Administrator", "SUSE Certified Administrator", 
        "IBM Certified System Administrator", "HPE ASE - Server Solutions Architect"
    ],
    "DevOps": [
        "Docker Certified Associate", "Kubernetes Administrator", "AWS Certified DevOps Engineer", "Azure DevOps Engineer Expert",
        "Google Professional Cloud DevOps Engineer", "Certified Jenkins Engineer", "Terraform Associate", "GitLab Certified",
        "Chef Certified Developer", "Puppet Professional", "Red Hat Certified Specialist in Ansible Automation"
    ],
    "UI/UX": [
        "Certified User Experience Professional (CUXP)", "Certified Professional in User Experience (CPUX)", "Interaction Design Foundation UX Certificate",
        "Adobe XD Certification", "Figma Certification", "Nielsen Norman Group UX Certification", "UX Design Institute Professional Diploma",
        "Human Factors International Certified Usability Analyst", "Google UX Design Certificate", "Sketch Certified"
    ]
}

# Maaş aralıkları - Deneyim ve şehir göre (TL)
def maas_hesapla(rol_kategori, kidem, deneyim_yil, sehir):
    # Baz maaş belirleme (role ve kıdeme göre)
    if rol_kategori in ["Yönetim ve Liderlik", "Yapay Zeka ve Makine Öğrenmesi"]:
        baz_maas = 35000
    elif rol_kategori in ["Güvenlik", "DevOps ve Altyapı", "Veri"]:
        baz_maas = 30000
    elif rol_kategori in ["Yazılım Geliştirme"]:
        baz_maas = 28000
    else:
        baz_maas = 25000
    
    # Kıdem çarpanı
    if kidem == "Stajyer":
        kidem_carpani = 0.4
    elif kidem == "Jr.":
        kidem_carpani = 0.8
    elif kidem == "Mid.":
        kidem_carpani = 1.0
    elif kidem == "Sr.":
        kidem_carpani = 1.4
    elif kidem == "Lead":
        kidem_carpani = 1.8
    elif kidem == "Müdür Yardımcısı":
        kidem_carpani = 2.0
    elif kidem == "Müdür":
        kidem_carpani = 2.5
    elif kidem == "Direktör":
        kidem_carpani = 3.0
    
    # Deneyim çarpanı (kıdem içinde zaten var, ama ince ayar için)
    deneyim_carpani = 1.0 + (deneyim_yil * 0.01)  # Her yıl için %1 artış
    
    # Şehir çarpanı
    if sehir == "İstanbul":
        sehir_carpani = 1.0
    elif sehir in ["Ankara", "İzmir"]:
        sehir_carpani = 0.9
    elif sehir in ["Bursa", "Kocaeli", "Antalya", "Tekirdağ"]:
        sehir_carpani = 0.85
    else:
        sehir_carpani = 0.8
    
    # Son maaş
    maas = int(baz_maas * kidem_carpani * deneyim_carpani * sehir_carpani)
    
    # Rastgele varyasyon ekleyelim (%10 aşağı veya yukarı)
    maas = int(maas * random.uniform(0.9, 1.1))
    
    return maas

# Veri setini oluşturan ana fonksiyon
def veri_seti_olustur(n=10000):
    veri = {
        "Kimlik_No": list(range(1, n+1)),
        "Ad": [],
        "Soyad": [],
        "Cinsiyet": [],
        "Yaş": [],
        "Email": [],
        "Telefon": [],
        "Şehir": [],
        "İlçe": [],
        "Rol_Kategorisi": [],
        "Rol": [],
        "Kıdem": [],
        "Ünvan": [],
        "Deneyim_Yıl": [],
        "Eğitim_Seviyesi": [],
        "Eğitim_Alanı": [],
        "Mezuniyet_Yılı": [],
        "Doğum_Yılı": [],
        "İşe_Başlama_Tarihi": [],
        "Maaş_TL": [],
        "Şirket": [],
        "Şirket_Büyüklüğü": [],
        "Sektör": [],
        "Çalışma_Şekli": [],
        "Uzaktan_Çalışma_Oranı": [],
        "Kullandığı_Teknolojiler": [],
        "Ana_Programlama_Dili": [],
        "İngilizce_Seviyesi": [],
        "Diğer_Diller": [],
        "Sertifikalar": [],
        "LinkedIn_Profili": [],
        "GitHub_Profili": [],
        "Toplam_Proje_Sayısı": [],
        "Teknik_Beceri_Puanı": [],  # 1-100 arasında
        "Soft_Skill_Puanı": [],     # 1-100 arasında
        "Yıllık_İzin_Gün": [],
        "Son_Terfi_Tarihi": []
    }
    
    bugun = datetime.now()
    
    # Her bir kişi için veri oluştur
    for i in range(n):
        # Temel kişisel bilgiler
        cinsiyet_deger = random.choices([c[0] for c in cinsiyet], weights=[c[1] for c in cinsiyet])[0]
        veri["Cinsiyet"].append(cinsiyet_deger)
        
        if cinsiyet_deger == "Erkek":
            ad = fake.first_name_male()
        else:
            ad = fake.first_name_female()
            
        soyad = fake.last_name()
        veri["Ad"].append(ad)
        veri["Soyad"].append(soyad)
        
        # Yaş belirleme (22-60 arası)
        yas = random.randint(22, 60)
        veri["Yaş"].append(yas)
        dogum_yili = bugun.year - yas
        veri["Doğum_Yılı"].append(dogum_yili)
        
        # İletişim bilgileri
        email_format = random.choice([
            f"{ad.lower()}.{soyad.lower()}@gmail.com",
            f"{ad.lower()}{soyad.lower()}@gmail.com",
            f"{ad.lower()}_{soyad.lower()}@hotmail.com",
            f"{soyad.lower()}.{ad.lower()}@yahoo.com",
            f"{ad.lower()[0]}{soyad.lower()}@gmail.com",
            f"{ad.lower()}.{soyad.lower()}@outlook.com"
        ])
        # Türkçe karakterleri düzelt
        email_format = re.sub(r'[çÇ]', 'c', email_format)
        email_format = re.sub(r'[ğĞ]', 'g', email_format)
        email_format = re.sub(r'[ıİ]', 'i', email_format)
        email_format = re.sub(r'[öÖ]', 'o', email_format)
        email_format = re.sub(r'[şŞ]', 's', email_format)
        email_format = re.sub(r'[üÜ]', 'u', email_format)
        veri["Email"].append(email_format)
        
        veri["Telefon"].append(fake.phone_number())
        
        # Şehir belirleme
        sehir = random.choices([s[0] for s in turkiye_sehirleri], weights=[s[1] for s in turkiye_sehirleri])[0]
        veri["Şehir"].append(sehir)
        
        # İlçe belirleme (city_part metodu yerine ilçe listesinden seçim yapıyoruz)
        if sehir in turkiye_ilceler:
            ilce = random.choice(turkiye_ilceler[sehir])
        else:
            ilce = random.choice(default_ilceler)
        veri["İlçe"].append(ilce)
        
        # Rol ve kıdem belirleme
        rol_kategori = random.choice(list(roller.keys()))
        veri["Rol_Kategorisi"].append(rol_kategori)
        rol = random.choice(roller[rol_kategori])
        veri["Rol"].append(rol)
        
        # Kıdem ve deneyim
        uygun_kidemler = []
        max_deneyim = min(yas - 22 + 2, 35)  # Üniversiteyi 22 yaşında bitirdiğini varsayıyoruz, +2 staj vb. için
        
        # Yaşa ve role uygun kıdemler
        for k, (min_yil, max_yil) in kidem_seviyesi_yil_araliklari.items():
            if min_yil <= max_deneyim and max_yil <= max_deneyim:
                uygun_kidemler.append(k)
                
        if not uygun_kidemler:
            uygun_kidemler = ["Stajyer", "Jr."]
            
        kidem = random.choice(uygun_kidemler)
        veri["Kıdem"].append(kidem)
        
        # Deneyim (kıdeme uygun)
        min_yil, max_yil = kidem_seviyesi_yil_araliklari[kidem]
        if max_yil > max_deneyim:
            max_yil = max_deneyim
            
        deneyim = round(random.uniform(min_yil, max_yil), 1)
        veri["Deneyim_Yıl"].append(deneyim)
        
        # Ünvan oluştur
        unvan = f"{kidem} {rol}"
        veri["Ünvan"].append(unvan)
        
        # Eğitim bilgileri
        # Yaşa göre eğitim seviyesi ağırlıklandırma
        if yas < 25:
            egitim_agirliklari = [0.05, 0.15, 0.75, 0.05, 0.0, 0.0]
        elif yas < 30:
            egitim_agirliklari = [0.05, 0.10, 0.65, 0.15, 0.02, 0.03]
        elif yas < 40:
            egitim_agirliklari = [0.05, 0.10, 0.55, 0.25, 0.03, 0.02]
        else:
            egitim_agirliklari = [0.10, 0.15, 0.50, 0.15, 0.05, 0.05]
            
        egitim_seviyesi = random.choices(egitim_seviyeleri, weights=egitim_agirliklari)[0]
        veri["Eğitim_Seviyesi"].append(egitim_seviyesi)
        
        # Eğitim alanı
        if egitim_seviyesi in ["Lise", "Sertifika Programı"]:
            egitim_alan_agirliklari = [0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.30, 0.20, 0.10, 0.01, 0.01, 0.20, 0.05, 0.04]
        else:
            # Üniversite mezunları için
            if rol_kategori in ["Yazılım Geliştirme", "Veri", "Yapay Zeka ve Makine Öğrenmesi"]:
                egitim_alan_agirliklari = [0.40, 0.30, 0.10, 0.05, 0.05, 0.03, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01]
            elif rol_kategori in ["DevOps ve Altyapı", "Güvenlik"]:
                egitim_alan_agirliklari = [0.35, 0.20, 0.20, 0.05, 0, 0.01, 0.01, 0.05, 0.05, 0.02, 0.01, 0.02, 0.01, 0.01, 0.01]
            else:
                egitim_alan_agirliklari = [0.25, 0.15, 0.10, 0.10, 0.05, 0.05, 0.02, 0.05, 0.05, 0.10, 0.01, 0.01, 0.01, 0.01, 0.04]
                
        egitim_alani_secim = random.choices(egitim_alanlari, weights=egitim_alan_agirliklari)[0]
        veri["Eğitim_Alanı"].append(egitim_alani_secim)
        
        # Mezuniyet yılı
        if egitim_seviyesi == "Lise":
            mezuniyet_yasi = 18
        elif egitim_seviyesi == "Önlisans":
            mezuniyet_yasi = 20
        elif egitim_seviyesi == "Lisans":
            mezuniyet_yasi = 22
        elif egitim_seviyesi == "Yüksek Lisans":
            mezuniyet_yasi = 24
        elif egitim_seviyesi == "Doktora":
            mezuniyet_yasi = 28
        else:  # Sertifika
            mezuniyet_yasi = random.randint(20, 30)
            
        mezuniyet_yili = bugun.year - (yas - mezuniyet_yasi)
        # Mezuniyetin çok eski olmaması için kontrol
        if mezuniyet_yili < 1980:
            mezuniyet_yili = random.randint(1980, 1990)
            
        veri["Mezuniyet_Yılı"].append(mezuniyet_yili)
        
        # İşe başlama tarihi
        # İşe başlama tarihi
        min_is_baslama_yili = max(mezuniyet_yili, bugun.year - int(deneyim))
        max_is_baslama_yili = bugun.year
        
        # İki değeri karşılaştırarak sorunları önle
        if min_is_baslama_yili > max_is_baslama_yili:
            # Eğer minimum değer maksimum değerden büyükse, minimum değeri kullan
            is_baslama_yili = min_is_baslama_yili
        else:
            # Normal durumda random.randint kullan
            is_baslama_yili = random.randint(min_is_baslama_yili, max_is_baslama_yili)
            
        is_baslama_ayi = random.randint(1, 12)
        is_baslama_gunu = random.randint(1, 28)  # Ay sonu problemlerinden kaçınmak için
        
        is_baslama_tarihi = datetime(is_baslama_yili, is_baslama_ayi, is_baslama_gunu)
        veri["İşe_Başlama_Tarihi"].append(is_baslama_tarihi.strftime("%Y-%m-%d"))
        
        # Maaş hesaplama
        maas = maas_hesapla(rol_kategori, kidem, deneyim, sehir)
        veri["Maaş_TL"].append(maas)
        
        # Şirket ve sektör
        sirket_kategorisi = random.choice(list(sirketler.keys()))
        sirket = random.choice(sirketler[sirket_kategorisi])
        veri["Şirket"].append(sirket)
        
        if sirket_kategorisi == "Büyük Teknoloji":
            sirket_buyuklugu = random.choice(["Büyük (201-1000)", "Kurumsal (1000+)"])
        elif sirket_kategorisi == "Yazılım Firmaları":
            sirket_buyuklugu = random.choice(["Orta (51-200)", "Büyük (201-1000)"])
        elif sirket_kategorisi == "Startuplar":
            sirket_buyuklugu = random.choice(["Startup (1-10)", "Küçük (11-50)"])
        elif sirket_kategorisi == "Şube/Danışmanlık":
            sirket_buyuklugu = random.choice(["Büyük (201-1000)", "Kurumsal (1000+)"])
        else:  # Savunma
            sirket_buyuklugu = random.choice(["Orta (51-200)", "Büyük (201-1000)", "Kurumsal (1000+)"])
            
        veri["Şirket_Büyüklüğü"].append(sirket_buyuklugu)
        
        # Şirkete göre sektör
        if sirket_kategorisi == "Büyük Teknoloji":
            uygun_sektorler = ["E-Ticaret", "Finans/Bankacılık", "Telekom", "Perakende"]
        elif sirket_kategorisi == "Yazılım Firmaları":
            uygun_sektorler = ["Yazılım", "Finans/Bankacılık", "Danışmanlık"]
        elif sirket_kategorisi == "Startuplar":
            uygun_sektorler = ["E-Ticaret", "Yazılım", "Finans/Bankacılık", "Oyun"]
        elif sirket_kategorisi == "Şube/Danışmanlık":
            uygun_sektorler = ["Danışmanlık", "Yazılım"]
        else:  # Savunma
            uygun_sektorler = ["Savunma Sanayi", "Kamu"]
        
        sektor = random.choice(uygun_sektorler)
        veri["Sektör"].append(sektor)
        
        # Çalışma şekli
        if sirket_buyuklugu in ["Startup (1-10)", "Küçük (11-50)"]:
            calisma_sekli_agirliklari = [0.60, 0.05, 0.10, 0.10, 0.10, 0.05]
        else:
            calisma_sekli_agirliklari = [0.80, 0.05, 0.05, 0.02, 0.03, 0.05]
            
        calisma_sekli = random.choices(calisma_durumlari, weights=calisma_sekli_agirliklari)[0]
        veri["Çalışma_Şekli"].append(calisma_sekli)
        
        # Uzaktan çalışma oranı
        if calisma_sekli == "Uzaktan":
            uzaktan_calisma = 100
        elif calisma_sekli == "Hibrit":
            uzaktan_calisma = random.choice([20, 40, 60, 80])
        elif calisma_sekli == "Freelance":
            uzaktan_calisma = random.choice([80, 100])
        else:
            uzaktan_calisma = random.choice([0, 0, 0, 0, 20])
            
        veri["Uzaktan_Çalışma_Oranı"].append(uzaktan_calisma)
        
        # Teknolojiler ve programlama dili
        kullanilan_teknolojiler = []
        
        # Role göre teknoloji kategorileri seçimi
        if rol_kategori == "Yazılım Geliştirme":
            tech_kategorileri = ["Programlama Dilleri", "Web Frontend", "Web Backend", "Veritabanları"]
            if "Mobil" in rol:
                tech_kategorileri.append("Mobil")
        elif rol_kategori == "Veri":
            tech_kategorileri = ["Programlama Dilleri", "Veritabanları", "Veri & AI"]
        elif rol_kategori == "DevOps ve Altyapı":
            tech_kategorileri = ["Programlama Dilleri", "DevOps Araçlar", "Veritabanları", "Bulut Platformlar"]
        elif rol_kategori == "Güvenlik":
            tech_kategorileri = ["Programlama Dilleri", "Güvenlik Araçları", "DevOps Araçlar"]
        elif rol_kategori == "Yapay Zeka ve Makine Öğrenmesi":
            tech_kategorileri = ["Programlama Dilleri", "Veri & AI", "Veritabanları"]
        elif rol_kategori == "Tasarım ve Kullanıcı Deneyimi":
            tech_kategorileri = ["Tasarım Araçları", "Web Frontend"]
        elif rol_kategori == "QA ve Test":
            tech_kategorileri = ["Programlama Dilleri", "Test Araçları", "DevOps Araçlar"]
        else:
            tech_kategorileri = ["Programlama Dilleri", "Veritabanları", "DevOps Araçlar"]
            
        # Her kategoriden birkaç teknoloji seç
        for kategori in tech_kategorileri:
            num_techs = random.randint(2, 4)
            secilen_techs = random.sample(teknolojiler[kategori], min(num_techs, len(teknolojiler[kategori])))
            kullanilan_teknolojiler.extend(secilen_techs)
            
        # Programlama dili belirleme (ana uzmanlık)
        prog_dilleri = [tech for tech in kullanilan_teknolojiler if tech in teknolojiler["Programlama Dilleri"]]
        
        if not prog_dilleri and ("Yazılım" in rol_kategori or "Veri" in rol_kategori or "Yapay Zeka" in rol_kategori):
            # Eğer hiç dil seçilmediyse, role uygun birkaç dil ekle
            if "Frontend" in rol or "Web" in rol:
                prog_dilleri = ["JavaScript", "TypeScript"]
            elif "Backend" in rol:
                prog_dilleri = ["Java", "Python", "C#"]
            elif "Mobil" in rol:
                prog_dilleri = ["Swift", "Kotlin", "Java"]
            elif "Veri" in rol_kategori or "Yapay Zeka" in rol_kategori:
                prog_dilleri = ["Python", "R"]
            else:
                prog_dilleri = random.sample(["Python", "Java", "JavaScript", "C#", "C++"], 2)
                
            kullanilan_teknolojiler.extend(prog_dilleri)
            
        # Ana dil seçimi
        if prog_dilleri:
            ana_prog_dili = random.choice(prog_dilleri)
        else:
            ana_prog_dili = ""
            
        veri["Kullandığı_Teknolojiler"].append(", ".join(kullanilan_teknolojiler))
        veri["Ana_Programlama_Dili"].append(ana_prog_dili)
        
        # Dil bilgisi
        # İngilizce seviyesi (sektörde çoğunlukla yüksek)
        ing_agirliklari = [0.02, 0.25, 0.40, 0.30, 0.03]
        ing_seviyesi = random.choices(dil_seviyeleri, weights=ing_agirliklari)[0]
        veri["İngilizce_Seviyesi"].append(ing_seviyesi)
        
        # Diğer diller
        diger_diller = ["Türkçe (Anadil)"]
        diger_dil_sayisi = random.choices([0, 1, 2], weights=[0.7, 0.25, 0.05])[0]
        
        if diger_dil_sayisi > 0:
            secilen_diller = random.choices(
                [d[0] for d in konusulan_diller if d[0] not in ["Türkçe", "İngilizce"]], 
                weights=[d[1] for d in konusulan_diller if d[0] not in ["Türkçe", "İngilizce"]],
                k=diger_dil_sayisi
            )
            
            for dil in secilen_diller:
                seviye = random.choices(dil_seviyeleri[:3], weights=[0.5, 0.4, 0.1])[0]
                diger_diller.append(f"{dil} ({seviye})")
                
        veri["Diğer_Diller"].append(", ".join(diger_diller))
        
        # Sertifikalar
        sertifika_kategorileri = []
        
        if "Bulut" in rol or "Cloud" in rol or "DevOps" in rol_kategori:
            sertifika_kategorileri.append("Bulut")
            sertifika_kategorileri.append("DevOps")
        
        if "Yazılım" in rol_kategori:
            sertifika_kategorileri.append("Yazılım Geliştirme")
            
        if "Proje" in rol or "Yönetici" in rol or "Liderlik" in rol_kategori:
            sertifika_kategorileri.append("Proje Yönetimi")
            
        if "Veri" in rol_kategori or "Yapay Zeka" in rol_kategori:
            sertifika_kategorileri.append("Veri")
            
        if "Ağ" in rol or "Network" in rol or "Sistem" in rol:
            sertifika_kategorileri.append("Ağ")
            sertifika_kategorileri.append("Sistem")
            
        if "Güvenlik" in rol_kategori or "Siber" in rol:
            sertifika_kategorileri.append("Güvenlik")
            
        if "Tasarım" in rol_kategori or "UX" in rol or "UI" in rol:
            sertifika_kategorileri.append("UI/UX")
            
        if not sertifika_kategorileri:
            # Eğer hiç kategori seçilmediyse, rastgele seç
            sertifika_kategorileri = random.sample(list(mesleki_sertifikalar.keys()), 1)
            
        # Sertifika sayısı
        if kidem in ["Stajyer", "Jr."]:
            sertifika_sayisi_agirliklari = [0.7, 0.2, 0.1, 0.0, 0.0]
        elif kidem in ["Mid."]:
            sertifika_sayisi_agirliklari = [0.3, 0.4, 0.2, 0.1, 0.0]
        elif kidem in ["Sr.", "Lead"]:
            sertifika_sayisi_agirliklari = [0.1, 0.2, 0.4, 0.2, 0.1]
        else:  # Yönetici seviyesi
            sertifika_sayisi_agirliklari = [0.1, 0.1, 0.3, 0.3, 0.2]
            
        sertifika_sayisi = random.choices(range(0, 5), weights=sertifika_sayisi_agirliklari)[0]
        
        secilen_sertifikalar = []
        for _ in range(sertifika_sayisi):
            if sertifika_kategorileri:
                kategori = random.choice(sertifika_kategorileri)
                sertifika = random.choice(mesleki_sertifikalar[kategori])
                secilen_sertifikalar.append(sertifika)
                
        veri["Sertifikalar"].append(", ".join(secilen_sertifikalar))
        
        # LinkedIn ve GitHub profilleri
        linkedin_varlik_orani = 0.9
        if random.random() < linkedin_varlik_orani:
            linkedin = f"linkedin.com/in/{ad.lower()}-{soyad.lower()}-{random.randint(100, 999)}"
            # Türkçe karakterleri düzelt
            linkedin = re.sub(r'[çÇ]', 'c', linkedin)
            linkedin = re.sub(r'[ğĞ]', 'g', linkedin)
            linkedin = re.sub(r'[ıİ]', 'i', linkedin)
            linkedin = re.sub(r'[öÖ]', 'o', linkedin)
            linkedin = re.sub(r'[şŞ]', 's', linkedin)
            linkedin = re.sub(r'[üÜ]', 'u', linkedin)
        else:
            linkedin = ""
            
        veri["LinkedIn_Profili"].append(linkedin)
        
        # GitHub profili (yazılımcılar için daha olası)
        if "Yazılım" in rol_kategori or "Geliştirici" in rol or "Developer" in rol:
            github_varlik_orani = 0.8
        elif "Veri" in rol_kategori or "Yapay Zeka" in rol_kategori:
            github_varlik_orani = 0.7
        else:
            github_varlik_orani = 0.3
            
        if random.random() < github_varlik_orani:
            username_tipleri = [
                f"{ad.lower()}{soyad.lower()}",
                f"{ad.lower()}_{soyad.lower()}",
                f"{ad.lower()}{soyad.lower()}{random.randint(1, 99)}",
                f"{ad.lower()[0]}{soyad.lower()}",
                f"{ad.lower()}.{soyad.lower()}",
                f"{soyad.lower()}{ad.lower()[0]}"
            ]
            github_username = random.choice(username_tipleri)
            # Türkçe karakterleri düzelt
            github_username = re.sub(r'[çÇ]', 'c', github_username)
            github_username = re.sub(r'[ğĞ]', 'g', github_username)
            github_username = re.sub(r'[ıİ]', 'i', github_username)
            github_username = re.sub(r'[öÖ]', 'o', github_username)
            github_username = re.sub(r'[şŞ]', 's', github_username)
            github_username = re.sub(r'[üÜ]', 'u', github_username)
            
            github = f"github.com/{github_username}"
        else:
            github = ""
            
        veri["GitHub_Profili"].append(github)
        
        # Proje sayısı
        if kidem == "Stajyer":
            proje_sayisi = random.randint(1, 3)
        elif kidem == "Jr.":
            proje_sayisi = random.randint(2, 8)
        elif kidem == "Mid.":
            proje_sayisi = random.randint(5, 15)
        elif kidem == "Sr.":
            proje_sayisi = random.randint(10, 25)
        elif kidem == "Lead":
            proje_sayisi = random.randint(15, 35)
        else:  # Yönetici
            proje_sayisi = random.randint(20, 50)
            
        veri["Toplam_Proje_Sayısı"].append(proje_sayisi)
        
        # Beceri puanları
        # Teknik beceri (deneyim, sertifikalar, projelere göre)
        teknik_baz = 40 + min(deneyim * 4, 40)  # Deneyim ağırlıklı
        teknik_sertifika_etkisi = min(len(secilen_sertifikalar) * 3, 15)
        teknik_proje_etkisi = min(proje_sayisi * 0.5, 10)
        
        teknik_beceri = int(teknik_baz + teknik_sertifika_etkisi + teknik_proje_etkisi)
        # Rastgele biraz değişiklik
        teknik_beceri += random.randint(-5, 5)
        # 0-100 arası olmasını sağla
        teknik_beceri = max(min(teknik_beceri, 100), 1)
        
        veri["Teknik_Beceri_Puanı"].append(teknik_beceri)
        
        # Soft skill (yöneticilik pozisyonlarında daha yüksek)
        if "Yönetim" in rol_kategori or "Manager" in rol or "Scrum" in rol or "Direktör" in kidem or "Müdür" in kidem:
            soft_skill_baz = 65
        else:
            soft_skill_baz = 50
            
        soft_skill_deneyim_etkisi = min(deneyim * 1.5, 30)
        
        soft_skill = int(soft_skill_baz + soft_skill_deneyim_etkisi)
        # Rastgele biraz değişiklik
        soft_skill += random.randint(-10, 10)
        # 0-100 arası olmasını sağla
        soft_skill = max(min(soft_skill, 100), 1)
        
        veri["Soft_Skill_Puanı"].append(soft_skill)
        
        # Yıllık izin günü
        if deneyim < 1:
            izin_gun = 14
        elif deneyim < 5:
            izin_gun = 14 + int(deneyim)  # Her yıl +1
        elif deneyim < 15:
            izin_gun = 20 + int((deneyim - 5) * 0.5)  # 5 yıl sonrası her 2 yılda +1
        else:
            izin_gun = 26  # Maksimum
            
        veri["Yıllık_İzin_Gün"].append(izin_gun)
        
        # Son terfi tarihi
        if kidem in ["Stajyer", "Jr."]:
            terfi_olasiliği = 0.3
        else:
            terfi_olasiliği = 0.8
            
        if random.random() < terfi_olasiliği:
            # Deneyimin yarısı kadar zaman önce terfi almış olabilir
            max_terfi_ay = min(int(deneyim * 12 * 0.7), 60)  # En fazla 5 yıl önce
            
            if max_terfi_ay > 3:  # En az 3 ay önce başlamış olmalı
                terfi_ay_once = random.randint(1, max_terfi_ay)
                terfi_tarihi = bugun - timedelta(days=terfi_ay_once*30)
                terfi_tarihi_str = terfi_tarihi.strftime("%Y-%m-%d")
            else:
                terfi_tarihi_str = ""
        else:
            terfi_tarihi_str = ""
            
        veri["Son_Terfi_Tarihi"].append(terfi_tarihi_str)
    
    # DataFrame oluştur ve CSV'ye kaydet
    df = pd.DataFrame(veri)
    return df

# Veri setini oluştur ve kaydet
if __name__ == "__main__":
    print("Veri seti oluşturuluyor...")
    veri_df = veri_seti_olustur(n=10000)
    
    # CSV'ye kaydet
    veri_df.to_csv("turkiye_it_sektoru_calisanlari.csv", index=False)
    print("Veri seti oluşturuldu ve kaydedildi: turkiye_it_sektoru_calisanlari.csv")
    
    # Temel istatistikler
    print("\nVeri Seti İstatistikleri:")
    print(f"Toplam Çalışan Sayısı: {len(veri_df)}")
    print(f"Cinsiyet Dağılımı: \n{veri_df['Cinsiyet'].value_counts(normalize=True).apply(lambda x: f'%{x*100:.1f}')}")
    print(f"Şehir Dağılımı (Top 5): \n{veri_df['Şehir'].value_counts().head().to_string()}")
    print(f"Ortalama Yaş: {veri_df['Yaş'].mean():.1f}")
    print(f"Ortalama Deneyim (Yıl): {veri_df['Deneyim_Yıl'].mean():.1f}")
    print(f"Ortalama Maaş (TL): {veri_df['Maaş_TL'].mean():.0f}")
    print(f"Rol Kategorileri: \n{veri_df['Rol_Kategorisi'].value_counts().to_string()}")
    
    # Maaş istatistikleri
    print("\nMaaş İstatistikleri (Rol Kategorilerine Göre):")
    print(veri_df.groupby('Rol_Kategorisi')['Maaş_TL'].agg(['mean', 'min', 'max']).applymap(lambda x: f"{x:,.0f} TL").to_string())
    
    # Kıdem dağılımı
    print("\nKıdem Dağılımı:")
    print(veri_df['Kıdem'].value_counts(normalize=True).apply(lambda x: f'%{x*100:.1f}').to_string())
    
    # Veri setinden örnek veriler
    print("\nVeri Setinden Rastgele 3 Örnek:")
    ornek_kolonlar = ['Ad', 'Soyad', 'Yaş', 'Şehir', 'Rol', 'Kıdem', 'Deneyim_Yıl', 'Maaş_TL', 'İngilizce_Seviyesi']
    print(veri_df[ornek_kolonlar].sample(3).to_string(index=False))