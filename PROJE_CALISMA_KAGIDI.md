# 🛰️ UZAY ENKAZI ALGILAMA, TAKİP VE ÇARPIŞMA KAÇINMA SİSTEMİ (SPACE-EDGE-AI)
## Kısıtlı Kaynaklı LEO Uyduları için Yerleşik (On-Board) Hafif Derin Öğrenme Tabanlı Algılama ve Takip Mimarisi

---

### 1. PROJE KİMLİĞİ VE TEMEL BİLGİLER

* **Proje Adı (TR):** Alçak Dünya Yörüngesi (LEO) Uyduları için Uç Yapay Zekâ (Edge AI) Tabanlı Gerçek Zamanlı Uzay Enkazı Algılama ve Takip Sistemi
* **Proje Adı (EN):** Edge-AI Powered Real-Time Space Debris Detection, Tracking, and Collision Avoidance for Resource-Constrained LEO Satellites
* **Uygulama Alanı:** Havacılık ve Uzay Mühendisliği, Bilgisayarlı Görü (Computer Vision), Gömülü Sistemler (Edge AI / On-Board Computing), Yörünge Mekaniği
* **Endüstriyel Vizyon Uyumu:** **Fergani Uzay**, TUA (Türkiye Uzay Ajansı), TÜBİTAK UZAY, NewSpace mikro/küp uydu takımyıldızları (constellation) vizyonu.

---

### 2. YÖNETİCİ ÖZETİ VE VİZYON (EXECUTIVE SUMMARY)

Alçak Dünya Yörüngesi'nde (LEO - 200 km ile 2000 km arası) insan yapımı nesnelerin, kullanım dışı uyduların ve roket kalıntılarının sayısı katlanarak artmaktadır. Bu durum, bir çarpışmanın zincirleme çarpışmaları tetikleyebileceği **Kessler Sendromu** riskini kritik seviyeye taşımıştır. Fergani Uzay gibi takımyıldızı (constellation) mimarisiyle çalışan yeni nesil ticari uydu operatörleri için yörünge güvenliği bir tercih değil, operasyonel bir zorunluluktur.

Geleneksel uzay gözetim yaklaşımları, yer tabanlı radar/optik teleskoplara veya uydunun çektiği görüntüleri yer istasyonuna indirmesine (downlink) dayanır. Ancak bu geleneksel yaklaşım iki büyük darboğaza takılır:
1. **İletişim ve İndirme Gecikmesi (Downlink Latency & Bandwidth Bottleneck):** Uydular yalnızca yer istasyonlarının görüş alanına girdiğinde yüksek hacimli optik görüntüleri aktarabilir. Yüksek çözünürlüklü görüntülerin indirilmesi saatler sürebilir; oysa bağıl hızı 7–10 km/s olan bir enkaz için karar süresi **saniyeler** mertebesindedir.
2. **Karasal Ağ Kapsama Boşlukları:** Okyanuslar veya kutup bölgeleri üzerinde yer istasyonu bağlantısı kurulamaz.

**Bu Projenin Temel Felsefesi:**
Görüntüyü yere indirmek yerine, **aklı uyduya taşımak (On-Board Edge AI Processing)**. Uydu üzerindeki yıldız izler (star tracker) veya görev optik kamerasından alınan ardışık kareler, uydu içindeki düşük güç tüketimli bir hızlandırıcıda (NVIDIA Jetson / NPU / FPGA) anlık işlenir. Arka plandaki yıldızlar elenir, uzay enkazları tespit edilip takip edilir (Multi-Object Tracking), olası çarpışma rotası belirlenir ve yere **gigabaytlarca ham görüntü yerine yalnızca birkaç baytlık "Çarpışma Uyarısı ve Kaçınma Manevra Vektörü (Telemetry Alert)"** iletilir.

```
[ Uydu Optik Sensörü / Star Tracker ] 
                  │ (Ham video/görüntü akışı)
                  ▼
┌─────────────────────────────────────────────────────────────┐
│             UYDU ÜZERİNDE ON-BOARD EDGE AI İŞLEMCİ          │
│                                                             │
│ 1. Uzay Gürültüsü & Yıldız Ayrıştırma (Star/Noise Removal)  │
│ 2. Hafif Nesne Tespiti (YOLO-Nano / P2 Tiny Debris Head)    │
│ 3. Çoklu Nesne Takibi (ByteTrack / SDT-Net / Kalman Filter) │
│ 4. Bağıl Yörünge ve Çarpışma Riski Analizi (TTC, Miss-Dist) │
└─────────────────────────────────────────────────────────────┘
                  │ (Sadece Kritik Uyarı & Manevra Vektörü: ~1 KB)
                  ▼
     [ Uydu Aviyonik / İtki Sistemi ] & [ Yer İstasyonu Telemetri ]
```

---

### 3. ÇÖZÜLEN PROBLEMLER (PROBLEM STATEMENT)

Bu bitirme projesi, uzay teknolojileri ve görüntü işlemenin kesişimindeki 3 temel problemi çözmeyi hedefler:

1. **Aşırı Düşük Güç ve Boyut Kısıtları (SWaP-C: Size, Weight, Power and Cost):**
   * Küçük uydularda (CubeSat / Micro-Sat) yapay zekâ donanımı için ayrılan güç bütçesi genellikle **5W – 15W** arasındadır.
   * Yüksek parametreli derin öğrenme modelleri (ör. büyük Transformer'lar, ağır CNN'ler) bu kısıtlar altında çalışamaz veya uydunun termal sınırlarını aşar.
   * *Çözüm:* Parametre sayısı minimuma indirilmiş, INT8/FP16 kuantize edilmiş, budanmış (pruned) hafif mimariler tasarlanmalıdır.

2. **Optik Zorluklar ve Küçük Nesne Problemi (Sub-Pixel / Tiny Objects):**
   * Uzay enkazları optik görüntülerde sadece birkaç piksel (hatta 2x2 veya 3x3 piksel) boyutunda zayıf bir ışık noktası veya çizgisel iz (streak) olarak görünür.
   * Arka planda milyonlarca sabit yıldız, kutup ışıkları (skylight / aurora), Dünya kenar aydınlığı (earth limb glare) ve kozmik ışınların sensörde oluşturduğu hatalı pikseller (hot pixels) bulunur.
   * *Çözüm:* Klasik tespit ağları (downsampling ile küçük nesneleri kaybeden yapılar) yerine, sığ özellik haritalarını koruyan (P2 detection head) ve temporal (zamansal) bilgiyi kullanan modeller geliştirilmelidir.

3. **Yalnızca Tespit Etmenin Yetmemesi (Tracking & Trajectory Estimation İhtiyacı):**
   * Tek bir karede bir ışık noktasını görmek enkaz olduğunu kanıtlamaz (sabit bir yıldız da olabilir).
   * Nesnenin arka plandaki yıldızlara göre bağıl hareketini hesaplamak için **Multi-Object Tracking (MOT)** gereklidir.
   * *Çözüm:* Tespit edilen enkazlar kareler boyunca takip edilerek bağıl hız ve çarpışma süresi (Time-To-Collision - TTC) tahmin edilmelidir.

---

### 4. SOMUT HEDEFLER VE BAŞARI KRİTERLERİ (METRICS & KPIS)

| Kategori | Metrik / Kriter | Hedef Değer | Açıklama |
| :--- | :--- | :--- | :--- |
| **Algılama Başarımı** | mAP@50 (Mean Average Precision) | **≥ %80** | Küçük enkaz sınıflarında yüksek doğruluk |
| **Takip Başarımı** | MOTA / HOTA Skoru | **≥ %70 - %75** | Kareler arası kimlik değişimi (ID switch) minimum olmalı |
| **Çıkarım Hızı (Inference)** | FPS (Kare/Saniye) | **≥ 15 - 30 FPS** | Uç cihaz üzerinde gerçek zamanlı akış performansı |
| **Gecikme (Latency)** | Uçtan Uca Gecikme | **< 60 ms** | Algılama + Takip + Risk analizi toplam süresi |
| **Bellek & Model Boyutu**| Model Boyutu (Disk/VRAM) | **< 15 MB / < 1.5 GB** | Gömülü bellek kısıtlarına tam uyum |
| **Güç Tüketimi** | Ortalama Güç Tüketimi | **< 10 Watt** | Jetson / Edge cihaz üzerinde 10W bütçe sınırı |

---

### 5. SİSTEM MİMARİSİ VE METODOLOJİ (TECHNICAL PIPELINE)

Proje 5 ana teknik katmandan oluşacaktır:

#### Katman 1: Ön İşleme ve Yıldız/Gürültü Ayrıştırma (Preprocessing Module)
* **Kozmik Işın / Hot Pixel Filtreleme:** Tekil piksellerde ani beliren yüksek yoğunluklu gürültülerin median filtreleme veya morfolojik operatörlerle temizlenmesi.
* **Yıldız Arka Plan Çıkarımı (Star Field Subtraction):** Yıldızların göksel küre koordinatlarındaki hareketi bilindiğinden veya homografi matrisi ile hizalandığından, ardışık kareler farkı (Frame Differencing) veya arka plan kestirimi ile sabit yıldızlar baskılanır; bağıl hareketi olan enkazlar ön plana çıkarılır.
* **Kontrast Artırma:** Uzayın karanlık ortamında zayıf ışık izlerini belirginleştirmek için CLAHE (Contrast Limited Adaptive Histogram Equalization).

#### Katman 2: Hafif Nesne Algılama (Lightweight Detection Engine)
* **Temel Model Seçimi:** YOLOv11-Nano (YOLO11n) veya YOLOv8-Nano / NanoDet / MobileNetV4.
* **Mimari İyileştirme (Küçük Nesne Başlığı - P2 Head):** Standart YOLO modelleri görüntüyü 8x, 16x ve 32x küçülterek büyük nesnelere odaklanır. Uzay enkazları birkaç piksel olduğu için ağa **160x160 (P2) seviyesinde yüksek çözünürlüklü sığ bir tespit başlığı** entegre edilir.
* **Hafifletilmiş Dikkat Mekanizması:** CBAM (Convolutional Block Attention Module) veya ECA (Efficient Channel Attention) eklenerek yıldız parıltıları arasından enkaz özniteliklerine odaklanma sağlanır.

#### Katman 3: Çoklu Nesne Takibi (Multi-Object Tracking & Association)
* **Algoritma:** ByteTrack veya BoT-SORT / SDT-Net (Space Debris Tracking Network yaklaşımı).
* **Kalman Filtresi (2D Yörünge Modeli):** Tespit edilen nesnelerin piksel koordinatlarındaki hız ve ivme vektörleri tahmin edilir. Geçici olarak parlaklığı azalan (flickering) enkazlar iz kaybetmeden takip edilmeye devam edilir.

#### Katman 4: Çarpışma Riski ve Bağıl Yörünge Kestirimi (Collision Risk Assessment)
* **Risk Göstergeleri:**
  * Bounding box merkezinin kameranın optik merkezine (Line-of-Sight) yaklaşma hızı.
  * Büyüme Oranı (Expansion Rate): Enkaz uydumuza yaklaşıyorsa boyutunda optik olarak genişleme oluşur (Time-to-Collision kestirimi).
  * Mesafe ve Teğet Geçme Uzaklığı (Miss Distance) simülasyonu.
* **Karar Mekanizması:** Çarpışma olasılığı belirli bir eşik değerini aştığında `CRITICAL_COLLISION_ALERT` telemetri paketi üretilir.

#### Katman 5: Model Sıkıştırma ve Uç Donanım Dağıtımı (Edge Optimization)
* **Budama (Pruning):** Gereksiz konvolüsyon filtrelerinin budanması (%20-%30 parametre tasarrufu).
* **Kuantizasyon (Quantization):** FP32 ağırlıkların FP16 ve INT8 hassasiyetine düşürülmesi (TensorRT Post-Training Quantization - PTQ).
* **Hedef Donanım Hızlandırma:** NVIDIA Jetson Orin Nano / Nano veya Raspberry Pi 5 + Hailo-8 AI modülü üzerinde donanıma özel TensorRT motoru (.engine) derlenmesi.

---

### 6. KULLANILACAK VERİ SETLERİ (DATASETS)

Uzay verileri gizlilik ve erişim zorluğu barındırdığı için projede hem gerçekçi akademik veri setleri hem de sentetik simülasyon verileri harmanlanacaktır:

1. **SDTD (Space Debris Tracking Dataset - Zhuang et al., 2025):**
   * *Açıklama:* Çalışma klasöründe bulunan en güncel literatür veri seti. 18.040 video sekansı, 62.562 kare ve 250.000 sentetik/gerçekçi uzay enkazı içerir.
   * *Avantajı:* Kutup istasyonu (Antarctic Station) gerçek arka planlarını, zorlu kutup ışıklarını (skylight) ve karmaşık uzay gürültülerini barındırır.
2. **SPEED+ (Next-Generation Spacecraft Pose Estimation Dataset - Stanford & ESA):**
   * *Açıklama:* Stanford Üniversitesi SLAB laboratuvarı ve Avrupa Uzay Ajansı (ESA) tarafından hazırlanan yüksek çözünürlüklü uzay aracı ve donanım enkazı veri seti.
   * *Avantajı:* Gerçekçi uzay ışıklandırması (Güneş parıltısı, derin uzay gölgeleri) testleri için idealdir.
3. **Sentetik Uzay Simülasyonu (Blender / Unreal Engine / Cosmos Simülasyonu):**
   * Blender veya Unreal Engine 5 üzerinde oluşturulacak senaryolarla CubeSat/mikro uydu kamerası perspektifinden enkaz yaklaşma videoları üretilebilir.
   * Farklı boyutlarda (1 cm - 50 cm), farklı bağıl hızlarda ve farklı aydınlatma açılarında sentetik veri çeşitlendirmesi (Data Augmentation) yapılacaktır.

---

### 7. KULLANILACAK ARAÇLAR, KÜTÜPHANELER VE DONANIMLAR (TECH STACK)

| Katman | Araç / Teknoloji | Kullanım Amacı |
| :--- | :--- | :--- |
| **Yapay Zekâ / Görüntü İşleme** | PyTorch, Ultralytics YOLOv11/v8, OpenCV, Albumentations | Model geliştirme, veri artırma, görüntü filtreleme |
| **Nesne Takibi (MOT)** | ByteTrack, FilterPy (Kalman Filter), SciPy | Çoklu hedef takibi ve yörünge ilişkilendirme |
| **Optimizasyon & Dağıtım** | NVIDIA TensorRT, ONNX, ONNX Runtime | Modeli INT8/FP16 formatına çevirme, hızlandırma |
| **Hedef Donanım (Edge AI)** | NVIDIA Jetson Orin Nano / Nano / Xavier NX (veya alternatif Raspberry Pi 5 + Coral/Hailo) | Uydu üzerindeki on-board bilgisayar simülasyonu |
| **Güç & Performans Profilleme** | `tegrastats`, NVIDIA Nsight Systems, `jtop` | Watt başına FPS ve bellek/ısınma tüketim ölçümü |
| **Uzay / Yörünge Simülasyonu** | Orekit, Poliastro veya Cesium Ion / STK simülasyonu | Uydunun ve enkazın yörünge hareketinin görselleştirilmesi |
| **Kullanıcı Arayüzü / Telemetri** | PyQt6 / Web Dashboard (Streamlit veya React) | Yer istasyonu takip ekranı ve tespit/takip gösterimi |

---

### 8. AKADEMİK VE ENDÜSTRİYEL YENİLİK DEĞERİ (NOVELTY)

Bu projeyi sıradan bir "YOLO çalıştırdım" projesinden ayırıp **birincilik / yüksek derece seviyesine** taşıyacak kritik inovasyon unsurları şunlardır:

1. **Sub-Pixel / Tiny Object İnovasyonu:** Klasik tespit modelleri uzaydaki 3-5 piksellik nesneleri kaçırır. Ağa eklenen P2 sığ katmanı ve uzay arka plan temizleme filtresi ile küçük nesne başarımı maksimize edilir.
2. **Uçta Gerçek Zamanlılık (Edge Verification):** Modelin sadece güçlü bir masaüstü ekran kartında değil, gerçek bir gömülü sistemde (Jetson) 10W altında 20+ FPS çalıştığının watt/FPS grafikleriyle ispatlanması.
3. **Bant Genişliği Tasarrufu İspatı (Bandwidth Reduction Factor):** Klasik sistemde 1 dakikalık HD video aktarımı ~150 MB veri tüketirken, bu sistemin yalnızca ~2 KB telemetri uyarısı göndererek **%99.99 veri iletim tasarrufu** sağladığının deneysel olarak ortaya konması.
4. **Fergani Takımyıldızı Vizyonuna Birebir Uyum:** Fergani'nin LEO takımyıldızlarında otonom operasyon kabiliyetine (Autonomous Space Situational Awareness - SSA) doğrudan yanıt veren bir konsept sunması.

---

### 9. 2 DÖNEMLİK PROJE YOL HARİTASI (TIMELINE & WORK PACKAGES)

#### Güz Dönemi (Dönem 1): Tasarım, Veri ve Temel Modeller (Prototip Oluşturma)
* **Hafta 1-3:** Literatür taraması (klasördeki `observation.pdf` ve `SDTD` makalesi temel alınarak), sistem gereksinimlerinin dökümante edilmesi.
* **Hafta 4-6:** SDTD ve SPEED+ veri setlerinin indirilmesi, ön işleme betiklerinin (star removal, CLAHE) yazılması, etiket formatlarının (YOLO formatı) standardizasyonu.
* **Hafta 7-10:** Temel modelin (Baseline YOLOv8n / YOLOv11n) eğitilmesi; mAP ve F1-skorlarının çıkarılması.
* **Hafta 11-14:** P2 sığ tespit başlığının eklenmesi ve küçük nesne başarımlarının kıyaslanması; 1. Dönem Bitirme Ara Raporu ve Sunumu.

#### Bahar Dönemi (Dönem 2): Takip, Optimizasyon, Gömülü Dağıtım ve Final
* **Hafta 1-4:** ByteTrack / Kalman Filter entegrasyonu; video sekansları üzerinde çoklu nesne takibi (MOT) ve yörünge izleme modülünün tamamlanması.
* **Hafta 5-7:** Çarpışma Riski Kestirimi (TTC ve yaklaşma hızı) algoritmasının yazılması; telemetri uyarı mekanizmasının kurulması.
* **Hafta 8-10:** Modelin TensorRT ile INT8/FP16 kuantizasyonu; Jetson Orin Nano / gömülü donanıma aktarımı; güç/gecikme (latency/watt) testleri.
* **Hafta 11-12:** Yer istasyonu simülasyon arayüzünün (GUI / Dashboard) geliştirilmesi; video üzerinde canlı demo hazırlığı.
* **Hafta 13-14:** Bitirme Tezi yazımı, akademik makale / bildiri veya TEKNOFEST Havacılık/Uzay kategorisi başvurusu; Final Savunması.

---

### 10. BİTİRME JÜRİSİNE SUNULACAK SOMUT ÇIKTILAR (FINAL DELIVERABLES)

1. **Eğitilmiş & Kuantize Edilmiş Model Dosyaları:** `space_debris_yolo11n_p2_int8.engine`
2. **Gömülü Cihaz Üzerinde Canlı Demo:** Jetson üzerinde gerçek zamanlı video akışında enkaz tespiti ve takip FPS sayacı.
3. **Yer Kontrol Arayüzü (Ground Control GUI):** Enkazın yörünge izini, kimlik numarasını (ID), anlık hızını ve "Çarpışma Tehlikesi: YÜKSEK / DÜŞÜK" durumunu gösteren kokpit arayüzü.
4. **Kapsamlı Karşılaştırma Raporu:**
   * Klasik Model vs. P2-Modifiyeli Model Karşılaştırması (mAP50, Precision, Recall).
   * FP32 vs. FP16 vs. INT8 Performans/Doğruluk Tablosu.
   * Güç Tüketimi (Watt) ve FPS Değerleri Grafiği.
5. **Akademik Bitirme Tezi Kitapçığı ve Proje Tanıtım Videosu.**
