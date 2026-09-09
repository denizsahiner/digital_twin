# Windkessel Dijital İkiz — ML Katmanı Derinlemesine Analiz Raporu

**Tarih:** 2026-09-09
**Kapsam:** 3-element Windkessel ODE tabanlı dijital ikizde, manşet/akıllı-saat vitallerinden
art-yük parametrelerini (`R`, `C`, `Zc`) tahmin eden XGBoost modelinin cross-validation,
feature engineering, SHAP açıklanabilirlik, ablation ve hata analizi ile derinleştirilmesi.

> Mevcut çıktılar **değiştirilmedi**. Tüm yeni kod ve sonuçlar `ml_analysis/` altında;
> tek istisna, eğitim betiğinin beklediği ama repoda olmayan
> `datasets/realistic_windkessel_dataset.csv` dosyasının projenin kendi üreticisiyle
> (`generaete_dataset_3.py`, seed=42, 60k örnek) yeniden üretilmesidir.

---

## 0. Yöntem Özeti

| Bileşen | Değer |
|---|---|
| Veri seti | `datasets/realistic_windkessel_dataset.csv` — 60 000 örnek, 3-element Windkessel ODE ileri simülasyonu + ölçüm gürültüsü (SBP ±3, DBP ±2, MAP ±1.5 mmHg) |
| Girdi (ham vital) | `HR, SV, Sys_Obs, Dia_Obs, MAP_Obs, PP_Obs` |
| Girdi (fizik-türevli) | `Shape_Index, Stiffness_Obs (PP/SV), R_Obs (MAP/CO), C_Physics (log-decay)` |
| Hedef | `R_True` (mmHg·s/mL), `C_True` (mL/mmHg), `Zc_True` (mmHg·s/mL) |
| Model | `MultiOutputRegressor(XGBRegressor)` — 1000 ağaç, lr 0.02, depth 6, subsample/colsample 0.8 (üretimdeki hiper-parametrelerle birebir) |
| Ölçekleme | `StandardScaler`, her fold içinde yalnız eğitim kısmına fit (sızıntısız) |

**Çalıştırma sırası:**
```
python ml_analysis/00_regenerate_dataset.py      # ~4 dk, sadece dosya yoksa
python ml_analysis/01_cross_validation.py
python ml_analysis/02_feature_engineering.py
python ml_analysis/03_shap_analysis.py
python ml_analysis/04_ablation_study.py
python ml_analysis/05_error_analysis.py
```
Ham çıktılar: `ml_analysis/outputs/` (JSON + CSV + PNG).

---

## 1. Proje Keşfi

- **Veri:** Tamamen sentetik. Her satır bir Windkessel ODE çözümünün son atımından çıkarılan
  SBP/DBP/MAP + eklenen ölçüm gürültüsü. Gerçek hasta yok → **yaş / hastalık / durum etiketi
  yok**, yalnızca vital aralıkları var (HR 50–129, SV 40–120 mL, SBP 80–210 mmHg).
- **Feature'lar:** 6 ham vital + 4 fizik-türevli (toplam 10). Fizik-türevliler analitik
  Windkessel formülleridir; `R_Obs = MAP / CO` ve `C_Physics` diyastolik basınç oranının
  logaritmik sönümünden.
- **Hedefler:** `R_True, C_True, Zc_True`. `Zc_true = U(0.03, 0.08) · R_true` olarak üretiliyor
  (yani Zc'nin R dışında bağımsız bir gözlemlenebilir sinyali **yok**).
- **Train/val/test:** Üretim betiğinde **yalnızca tek bir `train_test_split(test_size=0.2,
  random_state=42)`** var. **Cross-validation yok**, validation seti ayrılmıyor, erken
  durdurma yok.

---

## 2. Gerçek 5-Fold Cross-Validation

`KFold(5, shuffle=True, random_state=42)`, 60k örnek, üretim feature seti ve hiper-parametreleri.
(`outputs/01_cv_per_fold.csv`, `outputs/01_cv_summary.json`)

| Hedef | MAE (mean ± std) | RMSE (mean ± std) | R² (mean ± std) |
|---|---|---|---|
| **R_True**  | **0.0214 ± 0.0001** | 0.0274 ± 0.0001 | **0.9967 ± 0.0000** |
| **C_True**  | **0.0940 ± 0.0008** | 0.1397 ± 0.0008 | **0.9296 ± 0.0008** |
| **Zc_True** | **0.0181 ± 0.0001** | 0.0223 ± 0.0000 | **0.5715 ± 0.0063** |

**Yorum:**
- Fold'lar arası std neredeyse sıfır → model varyansı çok düşük; tek-split sonucu tekrarlanabilir.
- Ancak bu "mükemmel genelleme" değil, **sentetik verinin homojenliğinin** işaretidir
  (aynı ODE, düzgün dağılımlı parametreler, i.i.d. gürültü). Gerçek hastada bu std'ler büyür.
- `R` mükemmele yakın, `C` iyi, **`Zc` şanstan biraz iyi** (R² ≈ 0.57).

---

## 3. Feature Engineering İncelemesi

**Şu an kullanılan feature'lar:** yukarıdaki 6 ham + 4 fizik-türevli.

**Eklenen 7 yeni klinik-motivasyonlu türev feature** (`common.add_engineered_features`,
hepsi aynı manşet vitallerinden türetilebilir — invaziv ölçüm gerekmez):

| Feature | Tanım | Hipotez |
|---|---|---|
| `Ejection_Rate` | PP / sistol süresi | dP/dt vekili → sert aort / yüksek Zc |
| `Dia_Decay_Rate` | (SBP−DBP) / diyastol süresi | RC zaman sabiti → C |
| `Tau_Est` | t_dia / ln(SBP/DBP) | doğrudan R·C tahmini |
| `Systolic_Fraction` | sistol süresi / kalp döngüsü | görev çevrimi / morfoloji |
| `Form_Factor` | (MAP−DBP)/(SBP−DBP) | dalga yansıması / Zc'ye duyarlı şekil |
| `Impedance_Proxy` | PP / tepe sistolik akış | giriş empedansı vekili → Zc |
| `CO` | SV·HR/60 | kardiyak çıktı |

**3-fold CV ile A/B/C testi** (`outputs/02_feature_engineering.json`):

| Konfigürasyon | # feat | R² (R / C / Zc) | R MAE |
|---|---|---|---|
| A — sadece ham vital | 6 | 0.9957 / 0.9271 / 0.5662 | 0.0242 |
| **B — ham + fizik (üretim)** | 10 | 0.9967 / 0.9294 / 0.5695 | 0.0215 |
| C — ham + fizik + 7 yeni | 17 | 0.9967 / 0.9293 / 0.5700 | 0.0215 |

**Bulgu:**
- Fizik-türevli feature'lar **gerçek ama küçük** bir katkı veriyor: R'de MAE %11 düşüyor
  (0.0242 → 0.0215), R² 0.9957 → 0.9967.
- **7 yeni feature hiçbir şey katmadı** (ΔR² ≤ 0.0005, ΔMAE ≤ 0.0003). Tek tek eklendiğinde
  de en iyi kazanç Zc'de +0.0009 R² (`Impedance_Proxy` / `Ejection_Rate`) — gürültü seviyesinde.
- Sebep: gradyan-artırmalı ağaçlar bu etkileşimleri (PP/SV, τ, görev çevrimi) zaten ham
  vitallerden çıkarıyor. **Feature engineering doygunluğa ulaşmış**; darboğaz feature değil,
  **bilgi** (Zc için manşet verisi yetersiz).

---

## 4. SHAP Açıklanabilirlik

Hedef başına tek `XGBRegressor`, `shap.TreeExplainer`, 4000 örneklik test alt-kümesi.
(`outputs/03_shap_summary_all.png`, `outputs/03_shap_bar_all.png`, `outputs/03_shap_top5.json`)

![SHAP summary](outputs/03_shap_summary_all.png)

**Her parametre için en etkili 5 feature (ortalama |SHAP|):**

| Sıra | R_True | C_True | Zc_True |
|---|---|---|---|
| 1 | `R_Obs` — 0.358 | `C_Physics` — 0.248 | `R_Obs` — 0.018 |
| 2 | `Stiffness_Obs` — 0.034 | `Stiffness_Obs` — 0.114 | `Stiffness_Obs` — 0.003 |
| 3 | `SV` — 0.013 | `R_Obs` — 0.064 | `SV` — 0.001 |
| 4 | `HR` — 0.012 | `Shape_Index` — 0.008 | `C_Physics` — 0.001 |
| 5 | `Dia_Obs` — 0.012 | `PP_Obs` — 0.006 | `HR` — 0.001 |

**Klinik yorum:**
- **R:** Model neredeyse tümüyle `R_Obs = MAP/CO`'ya dayanıyor (2. sıranın 10 katı) ve onu hafif
  düzeltiyor. `R = MAP/CO` zaten sistemik vasküler direncin **tanımı**; klinik olarak doğru.
  SHAP eğilimi monotonik ve pozitif (yüksek R_Obs → yüksek R tahmini). Model fiziği ihlal etmiyor.
- **C:** `C_Physics` (diyastolik log-decay) + `Stiffness_Obs = PP/SV` birlikte belirleyici.
  `Stiffness_Obs` yüksekken SHAP negatif → **sert damar = düşük kompliyans**, fizyolojik olarak
  tutarlı. `R_Obs`'un da katkısı var çünkü diyastolik sönüm τ = R·C üzerinden R ile eşleşik.
- **Zc:** Hiçbir feature anlamlı sinyal taşımıyor (en yükseği 0.018, ±0.05 aralığında). Model,
  `Zc ≈ 0.055·R` koşullu ortalamasına çöküyor. **Dürüst sınır:** manşet SBP/DBP/MAP karakteristik
  empedansı kısıtlamıyor; Zc için nabız dalga formu / PPG kontur feature'ları şart.

---

## 5. Ablation Study — ML'in Fizik Baseline'a Gerçek Katkısı

Aynı hold-out test seti (80/20, seed 42), dört yaklaşım
(`outputs/04_ablation.json`, `outputs/04_ablation.png`):

- **(a) Sadece fizik:** öğrenme yok — `R = MAP/CO`, `C = t_dia/(R_obs·ln(SBP/DBP))`,
  `Zc = 0.055·R_obs` (fizikte kapalı form yok, 3–8% önselinin ortası).
- **(b) Sadece ML:** XGBoost, yalnız 6 ham vital.
- **(c) Hibrit:** XGBoost, ham + 4 fizik-türevli (= üretim modeli).
- **(c2) Hibrit-rezidü:** fizik `R,C` verir, XGBoost `(gerçek − fizik)` rezidüsünü öğrenir.

![Ablation](outputs/04_ablation.png)

| Yaklaşım | R² (R / C / Zc) | MAE (R / C / Zc) |
|---|---|---|
| (a) Sadece fizik | 0.970 / 0.878 / 0.546 | 0.0734 / 0.1244 / 0.0185 |
| (b) Sadece ML (ham) | 0.996 / 0.927 / 0.560 | 0.0241 / 0.0967 / 0.0183 |
| **(c) Hibrit (üretim)** | **0.997 / 0.929 / 0.565** | **0.0216 / 0.0947 / 0.0182** |
| (c2) Hibrit-rezidü | 0.997 / 0.928 / 0.561 | 0.0215 / 0.0956 / 0.0182 |

**Marjinal kazançlar:**

| Karşılaştırma | R MAE | C MAE | Zc MAE |
|---|---|---|---|
| ML-only − fizik-only | **−0.0493 (−%67)** | **−0.0277 (−%22)** | −0.0002 (~%0) |
| Hibrit − ML-only | −0.0025 (−%10 kalan hatanın) | −0.0019 (−%2) | −0.0001 (~%0) |
| Hibrit − fizik-only (toplam) | −0.0518 (−%71) | −0.0296 (−%24) | −0.0003 (~%2) |

**Bulgu — net rakamlarla:**
1. **ML'in asıl değeri fizik-analitik baseline'ın üzerine geliyor:** R hatasını %67, C hatasını
   %22 düşürüyor. Analitik `MAP/CO` ve log-decay formülleri gürültüye ve non-lineer etkileşimlere
   karşı kırılgan; ağaç modeli bunu düzeltiyor.
2. **Fizik feature'larını ML'e enjekte etmenin katkısı marjinal:** hibrit, sadece-ML'e göre R'de
   ekstra %10 (kalan hatanın), C'de %2 kazandırıyor. Yani "physics-informed" değeri feature
   enjeksiyonundan **değil**, hedefin fiziksel tanımından ve veri üreticisinin fiziğinden geliyor.
3. **Rezidü öğrenme ≈ doğrudan hibrit** — fiziği hard-code etmenin ek faydası yok.
4. **Zc'de hiçbir yaklaşım işe yaramıyor** (R² 0.55–0.57). Bu bir model/feature sorunu değil,
   **kimliklenebilirlik (identifiability)** sorunu.

---

## 6. Hata Analizi (Residual Analysis)

Üretim hibrit modeli, %20 hold-out. (`outputs/05_error_analysis.json`,
`outputs/05_error_analysis.png`, `outputs/05_worst20_*.csv`)

![Error analysis](outputs/05_error_analysis.png)

**Rezidü dağılımı — sistematik yanlılık yok:**

| Hedef | Bias (pred−true) | Std | MAE | p90 \|hata\| | p99 \|hata\| | max \|hata\| |
|---|---|---|---|---|---|---|
| R_True | −0.0001 | 0.0275 | 0.0216 | 0.045 | 0.073 | 0.149 |
| C_True | −0.0008 | 0.1409 | 0.0947 | 0.222 | **0.504** | **1.175** |
| Zc_True | +0.0002 | 0.0223 | 0.0182 | 0.036 | 0.055 | 0.073 |

**Hatalar rastgele değil — yapısal ve alt-grupta yoğunlaşıyor:**

- **C_True (en kritik):** ağır kuyruk. p99 hatası (0.50) ortalamanın **5.3 katı**.
  Bölmelere göre C MAE oranı: **PP ×5.5**, R ×3.9, HR ×2.4.
  En kötü 20 örnek profili (`05_worst20_C_True.csv`): `R_True ≈ 0.6` (aralığın alt sınırı),
  `PP ≈ 20–30 mmHg` (dar nabız basıncı), `C_True ≈ 2–3.6` (yüksek kompliyans).
  Korelasyonlar: `corr(|err_C|, C_True) = +0.53`, `corr(|err_C|, PP) = −0.41`,
  `corr(|err_C|, R_True) = −0.43`.
  → **Klinik grup:** düşük dirençli + elastik + dar nabız basınçlı damar profili (genç / sağlıklı
  vasküler yapı). Fiziksel sebep: düşük R'de diyastolik decay yavaş ve düz → C zayıf kısıtlanır.
- **R_True:** hata R büyüklüğüyle artıyor (R bölmeleri arası ×2.3, `corr = +0.39`) ve düşük SV /
  düşük HR'de kötüleşiyor (`corr(|err_R|, SV) = −0.25`). Yüksek dirençli (vazokonstriksiyon)
  hastalar en zoru.
- **Zc_True:** hata R büyüklüğünü takip ediyor (×2.6, `corr = +0.48`) — model Zc'yi ölçekli-R
  gibi tahmin ettiği için R büyüdükçe mutlak Zc hatası büyüyor.

**Özet:** yaş/durum etiketi olmadığından stratifikasyon vital aralıklarıyla sınırlı, ama hata
**belirli bir hemodinamik alt-grupta** (düşük R, dar PP, yüksek C) net biçimde yoğunlaşıyor;
rastgele dağılmıyor.

---

## 7. "Bu Projede Ne Öğrendim" (Özet)

1. **Gerçek cross-validation, tek-split'in söyleyemediğini söyler ama farklı bir şey söyler
   burada:** fold std'leri ~0 çıktı → model varyansı değil, **sentetik verinin homojenliği**
   ölçülüyor. Gerçek genelleme iddiası için dağılım-dışı (farklı HR/SV rejimi, gerçek MIMIC
   dalga formu) test şart.
2. **Ablation, "physics-informed ML" etiketini nicelleştirdi:** ML'in katkısı fizik-analitik
   baseline üzerine büyük (R MAE −%67, C −%22); ama fizik-türevli feature'ları XGBoost'a
   raw vitallerin üstüne eklemek neredeyse sıfır katkı (ΔR² ≤ 0.002). Değer feature
   enjeksiyonunda değil, hedefin fiziksel tanımında.
3. **Her parametre öğrenilebilir değil.** `Zc` üç yaklaşımda da R² ≈ 0.57'de takılı; SHAP hiçbir
   feature'ın sinyal taşımadığını gösteriyor. Manşet SBP/DBP/MAP'ten karakteristik empedans
   çıkarılamıyor — dürüst çözüm: ya nabız dalga formu feature'ları eklenmeli ya da Zc iddiadan
   çıkarılmalı.
4. **Hata yapısal, klinik olarak yorumlanabilir.** C hatasının %99 persentili ortalamanın 5
   katı ve tamamı düşük-direnç + dar-nabız-basıncı + yüksek-kompliyans alt-grubunda —
   ODE'de yavaş diyastolik sönümün C'yi zayıf kısıtlamasının doğrudan sonucu. Model iyi
   kalibre (bias ≈ 0), sorun belirsizlik, sapma değil.

---

## 8. CV'ye Eklenebilecek Teknik Özet

> **Kısa (1 cümle):**
> "3-element Windkessel dijital ikizde manşet vitallerinden (HR, SV, SBP/DBP/MAP) arteriyel
> art-yük parametrelerini tahmin eden çok-çıktılı XGBoost modeli geliştirdim; 60k örneklik
> sentetik ODE veri setinde **5-fold CV ile sistemik vasküler direnç R² = 0.997 (MAE 0.021
> mmHg·s/mL), arteriyel kompliyans R² = 0.93 (MAE 0.094 mL/mmHg)** elde ettim."

> **Uzun (2 cümle, ablation vurgulu):**
> "Physics-informed bir kardiyovasküler dijital ikizde Windkessel R/C/Zc parametrelerini
> non-invaziv vitallerden tahmin eden XGBoost modeli kurdum (60k sentetik örnek, 5-fold CV:
> R R² = 0.997, C R² = 0.93). SHAP ve ablation analizi ile ML'in fizik-analitik baseline'a
> kıyasla direnç tahmin hatasını %67, kompliyans hatasını %22 düşürdüğünü; karakteristik
> empedans Zc'nin ise manşet verisinden kimliklenebilir olmadığını (R² ≈ 0.57) gösterdim."

---

## Ek: Üretilen Dosyalar

| Dosya | İçerik |
|---|---|
| `ml_analysis/00_regenerate_dataset.py` | Eğitim veri setini projenin üreticisiyle yeniden üretir |
| `ml_analysis/common.py` | Ortak feature engineering + model fabrikası |
| `ml_analysis/01_..._05_*.py` | Beş analiz betiği |
| `outputs/01_cv_per_fold.csv`, `01_cv_summary.json` | 5-fold CV ham sonuçları |
| `outputs/02_feature_engineering.json` | A/B/C + leave-one-in feature testleri |
| `outputs/03_shap_summary_all.png`, `03_shap_bar_all.png`, `03_shap_top5.json` | SHAP |
| `outputs/04_ablation.png`, `04_ablation.json` | Ablation metrikleri + marjinal kazançlar |
| `outputs/05_error_analysis.png`, `05_error_analysis.json`, `05_worst20_*.csv` | Hata analizi |
