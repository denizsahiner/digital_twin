# ML Modelleri ve Veri Setleri Tutarlılık Sağlama Raporu (Validation Report)

Bu rapor, projenizdeki makine öğrenmesi modellerinin (Windkessel ve Cardio Risk) ve bunları eğitmek için üretilen/işlenen veri setlerinin fizyolojik ve istatistiksel tutarlılığını test etmek amacıyla hazırlanan doğrulama scripti ([validate_consistency.py](file:///Users/denizsahiner/.gemini/antigravity-cli/brain/b713fcc5-e3ef-4193-9514-cb39e29d83d4/scratch/validate_consistency.py)) çıktılarının analizidir.

Yazılan doğrulama scripti ile modeller üzerinde **Partial Dependence (Kısmi Bağımlılık) Analizi** ve **Fizyolojik Doğruluk Testleri** gerçekleştirilmiş ve projenin güvenilirliğini etkileyen **çok kritik iki tutarsızlık** tespit edilmiştir.

---

## 1. Windkessel Modeli Tutarlılık Analizi

**Windkessel Modeli (`realistic_model.pkl`), gözlemlenen vitallerden (SBP, DBP, HR, SV) gerçek hemodinamik parametreleri ($R$, $C$, $Z_c$) tahmin etmeye çalışır.**

### Yapılan Test:
Modelin eğitim seti olan `realistic_windkessel_dataset.csv` dosyasını üreten [generaete_dataset_3.py](file:///Users/denizsahiner/Desktop/digital_twin/data_set_generator/generaete_dataset_3.py) scripti incelendiğinde, 3-Element Windkessel diferansiyel denklemi çözülürken **aort karakteristik empedansı ($Z_c$) basınç düşüşü teriminin ($Q_{in} \times Z_c$) eklenmesinin unutulduğu** görülmüştür. 

Bunu doğrulamak için model iki ayrı veri setiyle test edilmiştir:
1.  **Dataset A (Doğru Fizyoloji):** $Z_c$ basınç düşüşü eklenmiş, gerçek 3-Element Windkessel verisi.
2.  **Dataset B (Hatalı/Eksik Fizyoloji):** $Z_c$ basınç düşüşü eklenmemiş (modelin eğitildiği veri setinin mantığı).

### İstatistiksel Çıktılar:
| Tahmin Edilen Parametre | Dataset B (Hatalı Fizyoloji - Eğitim Setiyle Aynı) | Dataset A (Doğru Fizyoloji - Gerçek Hayat) | Gözlem / Yorum |
| :--- | :--- | :--- | :--- |
| **Sistemik Direnç ($R$)** | $R^2 = 0.9928$ | $R^2 = 0.9633$ | Direnç tahmini her iki durumda da oldukça başarılı ve kararlı. |
| **Damar Esnekliği ($C$)** | $R^2 = 0.9351$ | **$R^2 = 0.3268$** | **Kritik Hata:** Model gerçek fizyolojideki veriyi gördüğünde damar uyumu (compliance) tahmini neredeyse tamamen çöküyor! |
| **Aort Empedansı ($Z_c$)** | $R^2 = 0.7152$ | $R^2 = 0.6924$ | Kabul edilebilir bir seviyede fakat korelasyon zayıf. |

### Fizyolojik Çelişki:
*   Fizyolojide damarın gevşeme süresi sabiti $\tau = R \times C$ bağıntısıyla korunur.
*   Model Dataset B (hatalı veri) üzerinde bu kuralı mükemmel öğrenmişken ($R^2 = 0.93$), gerçek fizyolojik Dataset A üzerinde test edildiğinde tahmin edilen $R \times C$ bağıntısının gerçek $\tau$ ile ilişkisi **$R^2 = -1.3715$** (anlamsız/rastgeleden daha kötü) çıkmaktadır.
*   **Sebebi:** Eğitim setinde $Z_c$ etkisi basınç eğrisine eklenmediği için, model sistolik kan basıncındaki ani akış yükselmelerini yanlış anlamlandırmış ve bunu yanlış compliance tahminlerine yansıtmıştır.

---

## 2. Cardio Risk Modeli Tutarlılık Analizi

**Cardio Modeli (`cardio_model_nolabs.pkl`), hastanın demografik ve vital verilerine dayanarak kalp hastalığı olasılığını tahmin eder.**

Yapılan Kısmi Bağımlılık (Sensitivity) analizi sonuçlarında **klinik ve biyolojik açıdan kabul edilemez tutarsızlıklar** tespit edilmiştir.

### A. Yaşam Tarzı Alışkanlıkları (Sigara ve Alkol) Çelişkisi
Script üzerinden diğer tüm değişkenleri sabit tutulan 45 yaşında erkek bir hastanın sadece yaşam alışkanlıkları değiştirilerek risk oranları karşılaştırılmıştır:

*   **Temiz Yaşam (Sigara Yok, Alkol Yok):** Risk = **%27.26**
*   **Sadece Sigara İçen:** Risk = **%26.97** (Fark: **-0.28%**) ➔ *Sigara içmek kalp hastalığı riskini neredeyse hiç etkilemiyor, hatta çok az düşürüyor!*
*   **Sadece Alkol Alan:** Risk = **%14.66** (Fark: **-12.60%**) ➔ *Alkol içmek kalp hastalığı riskini yarı yarıya düşürüyor!*
*   **Sigara + Alkol Kullanan:** Risk = **%13.31** (Fark: **-13.95%**) ➔ *Hem sigara içip hem alkol kullanan birinin riski, temiz yaşayan birine göre %14 daha düşük çıkıyor!*

> [!CAUTION]
> **Klinik Değerlendirme:** Bu durum tıbbi gerçeklerle tamamen çelişmektedir. Sigara ve alkolün kalp sağlığı için "koruyucu faktör" olarak modellenmesi, eğitim seti olan Kaggle `cardio_train.csv` dosyasındaki **seçim yanlılığından (selection bias)** veya **karıştırıcı değişkenlerden (confounding variables)** kaynaklanmaktadır. Model bu hatalı korelasyonu doğrudan ezberlemiştir.

### B. Kan Basıncı (Tansiyon) Hassasiyet Testi
Nabız sabit tutularak Sistolik Kan Basıncı (SBP) 90'dan 200 mmHg'ye kadar taranmıştır:
*   $SBP \le 120$ iken risk **%20 - %27** arasındadır (Normal).
*   $SBP = 130$ mmHg olduğunda risk birden **%52.64**'e fırlamaktadır (Klinik olarak Stage 1 Hipertansiyon sınırıdır, model bu sınırı iyi yakalamış).
*   $SBP \ge 140$ mmHg olduğunda risk **%85.77**'ye ulaşmaktadır (Stage 2 Hipertansiyon sınırı).
*   **Hata:** Ancak tansiyon kritik seviyelere çıkmaya devam ettikçe risk artmamakta, aksine hafifçe düşmektedir: $SBP = 200$ mmHg iken risk **%80.73** olarak tahmin edilmektedir. Bu durum, modelin aşırı uçlardaki verilerde (extreme values) genelleme yapamadığını (overfitting/under-representation) göstermektedir.

### C. Yaş Hassasiyet Testi
Yaş 20'den 80'e taranmıştır:
*   20 - 40 Yaş: Risk stabil **%13 - %14** bandındadır.
*   50 Yaş: Risk **%27.30**'a yükselmektedir.
*   70 - 80 Yaş: Risk **%60.76**'ya ulaşmaktadır.
*   **Klinik Değerlendirme:** Yaş hassasiyeti fizyolojik beklentilerle tamamen tutarlıdır.

---

## 3. Tutarlılığı Düzeltmek İçin Yol Haritası

Bu tutarsızlıkları gidermeden projenin klinik veya simülasyon amaçlı kullanılması yanıltıcı olacaktır. Düzeltme için yapılması gerekenler:

1.  **Windkessel Simülatörü ve Veri Kümesi Düzeltmesi:**
    *   [generaete_dataset_3.py](file:///Users/denizsahiner/Desktop/digital_twin/data_set_generator/generaete_dataset_3.py) scripti güncellenmeli ve basınç profiline $Q(t) \times Z_c$ terimi eklenmelidir.
    *   Bu doğru fizik formülüyle veri seti yeniden oluşturularak model (`realistic_model.pkl`) yeniden eğitilmelidir.
2.  **Cardio Risk Modeli Düzeltmesi:**
    *   Kaggle veri setindeki sigara ve alkol gibi dengesiz dağılmış özellikler için **oversampling/undersampling** veya feature weighting uygulanmalıdır.
    *   Gerekirse XGBoost'un ağaç derinliği (`max_depth`) azaltılmalı, sigara ve alkol özelliklerinin katsayıları veya kararları üzerinde monotonluk kısıtlamaları (`monotone_constraints`) uygulanarak tansiyon ve alışkanlıkların riski sadece artırıcı yönde etkilemesi zorlanmalıdır.

---

## 4. Downloads Klasöründeki Yeni Veri Setlerinin İncelemesi

Downloads klasörünüze indirdiğiniz iki yeni veri seti analiz edilmiştir:

### A. `windkessel3_dataset.csv`
*   **Boyut ve Yapı:** 8.28 MB, 120,000 satır, 8 sütun.
*   **İçerik:** Çeşitli $R$, $C$, $Z_c$ kombinasyonlarına karşılık gelen zaman serisi akış (`Q_ml_per_s`), giriş basıncı (`P_inlet_mmHg`) ve 2-element basınç (`P_across_R_C_mmHg`) değerleri.
*   **Fiziksel Tutarlılık Testi:** 
    *   Bu veri setinin 3-element Windkessel fizik kuralı ($P_{inlet} = P_{across\_RC} + Q \cdot Z_c$) uyumu kontrol edilmiş ve **%99.99** oranında ($R^2 = 0.999933$) mükemmel bir uyum sağladığı görülmüştür.
    *   **Klinik Değerlendirme:** Bu veri seti, Windkessel modelindeki diferansiyel denklemlerin zamana bağlı basınç dalga formunu (pulse wave contour) eğitmek için **kusursuz ve tutarlı bir kaynaktır**.

### B. `windkessel_parameters_dataset.csv`
*   **Boyut ve Yapı:** 13.4 KB, 73 satır, 17 sütun.
*   **İçerik:** Nabız dalga analizinde (PWA) kullanılan dalga formu özellikleri (`max_upstroke_slope`, `upstroke_time`, `pressure_std` vb.) ile bu dalga formunu oluşturan fiziksel parametrelerin ($Z_0$, $R_p$, $C$) ve klinik basınçların (SBP, DBP, MAP, PP) özet tabloları.
*   **Klinik Değerlendirme:** Çok küçük bir veri kümesidir (73 örnek). Waveform özelliklerinden Windkessel parametre tahmini yapan daha basit regresyon modellerini hızlıca test etmek veya doğrulamak (validation) için kullanılabilir.

---

## 5. Git Branch Karşılaştırma Analizi

Depodaki (Repository) git branch'leri sorgulanarak karşılaştırılmıştır:

*   **Mevcut Aktif Branch:** `master_2` (Yerel ve `origin/master_2` ile güncel).
*   **Diğer Branch:** `main` (Yerel ve `origin/main` ile güncel).

### Karşılaştırma Bulguları:
1.  **Güncellik Durumu:** `master_2` branch'i, `main` branch'inde bulunmayan en güncel commit olan `274888d` ("Add files via upload") commit'ine sahiptir. Dolayısıyla **`master_2` en güncel ve en doğru koda sahip olan branch'tir**.
2.  **Dosya Yapısı & Farklar:**
    *   `main` branch'inde projedeki tüm dosyalar düzenli klasör yapısı (`data_engineering/`, `data_set_generator/`, `ml_model_cardio/`, `ml_model_windkessel/`) içinde yer almaktadır.
    *   `master_2` branch'inde ise hem bu klasör yapısı bulunmakta hem de projenin ana (root) dizininde aynı dosyaların kopyaları (`extract_cardio.py`, `train_cardio.py`, `visualizer.py` vb.) yer almaktadır.
    *   `diff` kontrolü yapılmış ve kökte duran dosyaların, klasör altındaki karşılıklarıyla **%100 aynı (birebir kopya)** olduğu doğrulanmıştır.
    *   **Tavsiye:** `master_2` branch'ini kullanmaya devam etmeli ancak ana dizinde biriken ve dağınıklık yaratan kopyaları silerek sadece klasör yapılarını korumalısınız.

---

## 6. bitirme1 Klasöründeki Veri Setlerinin Analizi

Downloads klasörü altındaki `bitirme1` dizininde yer alan CSV dosyaları incelenmiştir:

### A. `sanal_kalp_veriseti.csv`
*   **Boyut ve Yapı:** 296.95 KB, 2,000 satır, 9 sütun.
*   **İçerik:** Fizyolojik parametreler (`R`, `C`, `Zc`, `HR`, `SV`) ile bunlara karşılık gelen kan basıncı değerlerini (`Systolic_BP`, `Diastolic_BP`, `MAP`, `Pulse_Pressure`) içeren küçük ölçekli sentetik bir veri setidir.

### B. `tahmin_ve_gercek.csv`
*   **Boyut ve Yapı:** 44.70 KB, 400 satır, 6 sütun.
*   **İçerik:** `sanal_kalp_veriseti.csv` veri setindeki $R$, $C$ ve $Z_c$ gerçek değerleri ile bir makine öğrenmesi modelinin bunlara karşılık yaptığı tahminleri (`_pred`) kıyaslayan test setidir. (2,000 satırlık veri setinin %20 test spliti tam olarak bu 400 satıra denk gelmektedir).

### Performans ve Doğruluk Analizi (Validation Metrics):
Bu test tahminlerinin başarısını ölçmek için koşturulan doğrulama scripti sonuçları şöyledir:
*   **Sistemik Direnç ($R$):** $R^2 = 0.3457$ | $MAE = 0.1121$
*   **Damar Esnekliği ($C$):** $R^2 = 0.5853$ | $MAE = 0.1417$
*   **Aort Empedansı ($Z_c$):** **$R^2 = -0.0427$** | $MAE = 0.0090$

### Önemli Bulgular ve Kod İlişkisi:
1.  **Kodlarda Kullanım Durumu:** Bu veri setleri şu anda aktif kod tabanınızdaki ([app.py](file:///Users/denizsahiner/Desktop/digital_twin/app.py) veya diğer `.py` eğitim dosyaları) hiçbir scriptte **kullanılmamaktadır**. Kod tabanında bu dosyalara yapılan herhangi bir referans bulunmamaktadır.
2.  **Eski Sürüm / Pilot Çalışma Kanıtı:** Tahmin başarı oranlarının (özellikle $Z_c$ için negatif $R^2$ değerinin) çok düşük olması, bu veri setinin projenin başlarında (muhtemelen bitirme tezi hazırlık aşamasında) yapılan **eski bir pilot çalışmaya veya test aşamasına ait olduğunu** göstermektedir.
3.  **Mevcut Modellerle İlişkisi:** Aktif olarak kullanılan `realistic_model.pkl` modeli 60,000 satırlık çok daha gelişmiş bir veri setiyle eğitilmiş olup, burmadaki tahminlerden çok daha yüksek doğruluğa sahiptir. Dolayısıyla `bitirme1` altındaki bu dosyalar tamamen **geçmişe ait yedek verilerdir**.


