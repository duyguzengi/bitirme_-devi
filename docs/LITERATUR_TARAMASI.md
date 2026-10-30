# 📚 KAPSAMLI LİTERATÜR TARAMASI VE KURAMSAL ÇERÇEVE
## Uydularda Uç Yapay Zekâ (Edge AI) ile Uzay Enkazı Algılama ve Takip Sistemleri

**Proje:** Kısıtlı Kaynaklı LEO Uyduları İçin Yerleşik Hafif Derin Öğrenme Tabanlı Enkaz Tespiti  
**Tarih:** 2026  
**Referans Dokümanlar:** 
1. *Advancing Earth Observation: A Survey on AI-Powered Image Processing in Satellites* (Aidan Duggan, Bruno Andrade, Haithem Afli, IEEE) [`observation.pdf`]
2. *High Performance Space Debris Tracking in Complex Skylight Backgrounds with a Large-Scale Dataset* (Guohang Zhuang, Weixi Song et al., Shanghai AI Lab & ZJU, 2025) [`SDTD Paper`]
3. *SPEED+: Next-Generation Spacecraft Pose Estimation Dataset* (Stanford Space Rendezvous Laboratory & ESA)

---

### 1. GİRİŞ VE PROBLEMİN ARKA PLANI

Alçak Dünya Yörüngesi (Low Earth Orbit - LEO, 200–2000 km), insanlığın en yoğun uydu konuşlandırdığı alandır. 2024 yılı sonu itibarıyla yörüngede 10.500'den fazla aktif uydu bulunmakta olup bu sayı NewSpace mega-takımyıldızları (Starlink, OneWeb, Fergani Takımyıldızı) ile her yıl binlerce yeni uydu ile artmaktadır.

Yörüngede hızı saatte ~28.000 km'ye (bağıl hızda 7–10 km/s) ulaşan 1 cm'lik bir uzay enkazının kinetik enerjisi, patlayan bir el bombasının enerjisine eşdeğerdir. Meydana gelebilecek tek bir çarpışma, binlerce yeni enkaz parçası üreterek **Kessler Sendromu**'nu tetikleyebilir ve belirli yörünge irtifalarını onlarca yıl kullanılamaz hale getirebilir.

Klasik uzay durumsal farkındalık (Space Situational Awareness - SSA) sistemleri, yer konuşlu radar ve optik teleskoplara dayanmaktadır. Ancak:
* Yer radarları 1–10 cm altındaki küçük enkazları yakalamakta zorlanır.
* Yer istasyonuna görüntü indirme (downlink) süreçleri, uydunun yer istasyonu görüş açısına (line-of-sight pass) girmesine bağımlıdır ve saatlerce gecikme (latency) yaratır.
* Dar bant genişliği (downlink bandwidth), ham optik görüntülerin sürekli yere aktarılmasını imkânsız kılar.

Bu sebeple çözüm: **Görüntüleri uyduda yerleşik (on-board) işleyerek yalnızca milisaniyeler içinde çarpışma uyarısı üreten Uç Yapay Zekâ (Edge AI) mimarisidir.**

---

### 2. BİRİNCİ TEMEL ÇALIŞMA ANALİZİ: `observation.pdf`
*(Duggan, Andrade, Afli - IEEE Survey on AI-Powered On-Board Image Processing)*

Bu kapsamlı derleme makalesi, uydularda derin öğrenme modellerinin çalıştırılmasındaki temel fiziksel ve donanımsal kısıtları ve bunlara yönelik çözüm stratejilerini ortaya koymaktadır:

#### 2.1. Uydulardaki Fiziksel ve Donanımsal Kısıtlar (SWaP-C)
1. **Güç Kısıtı (Power Constraints):**
   * Uydunun tek enerji kaynağı güneş panelleri ve bataryalardır.
   * Tipik bir 1U–3U CubeSat için üretilen toplam güç ortalama **5W – 15W** arasındadır. Bu gücün büyük kısmı aviyonik, telekomünikasyon (ADCS, UHF/X-band) ve ısıtma sistemlerine gider; uç yapay zekâ hızlandırıcısına ayrılabilen net güç bütçesi çoğunlukla **<5W – 10W** ile sınırlıdır.
2. **Termal Dağılım Kısıtı (Thermal Dissipation in Vacuum):**
   * Uzayda hava veya atmosferik konveksiyon (fan soğutması) bulunmaz.
   * Donanımın ürettiği ısı yalnızca iletim (conduction) yoluyla uydu gövdesine ve oradan ışıma (radiation) ile uzay boşluğuna yayılabilir. Yüksek TDP'ye sahip GPU'lar hızla aşırı ısınıp termal frenlemeye (thermal throttling) veya kalıcı hasara uğrar.
3. **Bellek ve Bant Genişliği Kısıtı (Memory & Bandwidth):**
   * Gömülü sistemlerdeki RAM kapasitesi (örn. 2 GB - 8 GB) ve bellek bant genişliği masaüstü sunuculardan çok daha düşüktür.
   * Model ağırlıklarının DRAM'den NPU/GPU çekirdeklerine aktarılması sırasındaki enerji tüketimi, aritmetik hesaplamadan katbekat fazladır.
4. **Uzay Radyasyonu (Radiation Effects):**
   * Kozmik ışınlar ve Van Allen kuşağındaki yüklü parçacıklar, silikon transistörlerde Tekil Olay Etkilerine (Single Event Effects - SEE, bit-flip, latch-up) ve Toplam İyonlaştırıcı Doza (Total Ionizing Dose - TID) yol açar.
   * Uç modellerin parametre hatalarına karşı gürbüz (fault-tolerant) olması gerekmektedir.

#### 2.2. Uydular İçin Model Hızlandırma ve Optimizasyon Stratejileri
Makalede önerilen ve projemize entegre edilecek yöntemler:
* **Budama (Pruning):** Konvolüsyonel katmanlardaki ağırlıkların %30–%50'sinin doğruluk kaybı olmaksızın sıfırlanabilmesi.
* **Kuantizasyon (Quantization):** FP32 (32-bit kayan nokta) ağırlıkların FP16 ve özellikle **INT8** (8-bit tamsayı) seviyesine düşürülmesi. Bu işlem bellek ayak izini 4 kat azaltırken çıkarım hızını (FPS) 2–3 kat artırır.
* **Hafif Mimariler (Lightweight Backbones):** MobileNet, ShuffleNet ve YOLO-Nano sınıfı mimarilerin kullanımı.
* **COTS (Commercial Off-The-Shelf) Donanımlar:** Uzay kalifikasyonlu pahalı ve yavaş çipler yerine, düşük maliyetli ve yüksek TOPS/Watt oranına sahip COTS hızlandırıcılar (NVIDIA Jetson Orin Nano / Nano, Google Coral Edge TPU, Hailo-8, Xilinx Kria K26).

---

### 3. İKİNCİ TEMEL ÇALIŞMA ANALİZİ: `SDTD Paper` (2025)
*(Zhuang, Song et al. - High Performance Space Debris Tracking in Complex Skylight Backgrounds with a Large-Scale Dataset)*

Bu makale, uzay enkazlarının optik doğasını, arka plan gürültülerini ve en güncel takip mimarisini sunmaktadır:

#### 3.1. Uzay Enkazlarının Optik Doğası (Line Source / Streak)
* Uzun pozlamalı (long-exposure) optik kameralarda veya yıldız izlerlerde (star tracker), sabit arka plan yıldızları dairesel nokta ışık kaynakları olarak kalırken, uydumuza göre bağıl hareket eden enkazlar birer **çizgisel ışık izi (line source / streak)** şeklinde görünür.
* Gerçek enkaz izleri, optik sistemin **Nokta Dağılım Fonksiyonuna (Point Spread Function - PSF)** uyar:
  $$I_{k}^{t}(x, y) = \frac{S}{2\pi \delta_{psf}^2} \exp\left(-\frac{(x - x_c)^2 + (y - y_c)^2}{2\delta_{psf}^2}\right)$$
  Burada $(x_c, y_c)$ enkazın anlık merkez koordinatları, $\delta_{psf}$ optik difüzyon standart sapmasıdır.

#### 3.2. Astronomik Ön İşleme: ZScale Yöntemi
Astronomik sensörler geniş dinamik aralığa (12-bit/16-bit) sahiptir. Makalede piksellerin [0, 255] standart aralığına gürbüz şekilde çekilmesi için ZScale kontrast gerdirme formülü kullanılır:
$$z_1 = z_{median} - k \cdot z_{std}, \quad z_2 = z_{median} + k \cdot z_{std} \quad (k=2.5)$$
$$I_{scaled}(x, y) = 255 \times \frac{I(x, y) - z_1}{z_2 - z_1}$$

#### 3.3. SDTD Veri Seti Anatomisi
* Zwicky Transient Facility (ZTF) teleskopundan alınan 16.040 yüksek kaliteli gerçek gökyüzü gözlemine dayalıdır.
* 18.040 video sekansı ve 62.562 kareden oluşur; toplam 250.000 enkaz nesnesi içerir.
* Arka planda kutup ışıkları (skylight), ay ışığı, atmosferik saçılmalar ve yıldız parıltıları gibi aşırı zorlu koşulları simüle eder.
* Gerçek kutup istasyonu (Antarctic Station) verileri üzerinde test edilerek **%73.2 MOTA** takip başarısı elde edilmiştir.

#### 3.4. SDT-Net Mimarisi ve Limitleri
* SDT-Net, enkaz çizgilerinin iki uç noktasını (endpoints) tespit eden ve CenterTrack tarzı kareler arası ilişkilendirme yapan bir ağdır.
* **Sınırlılığı:** SDT-Net ağır bir sunucu GPU mimarisine göre tasarlanmıştır; mikro uydu üzerindeki 5W–10W Jetson benzeri bir uç donanımda doğrudan gerçek zamanlı (real-time 20+ FPS) çalışması zordur.
* **Bizim İnovasyonumuz:** SDTD veri setinin ve enkaz izi mantığının, **hafifletilmiş P2-Head YOLO (YOLOv11n) ve ultra-hızlı ByteTrack/Kalman filtreleme** hattına dönüştürülerek gömülü hızlandırıcılar için optimize edilmesidir.

---

### 4. ÜÇÜNCÜ VERİ KAYNAĞI: `SPEED+` (Stanford & ESA)

* **Stanford SLAB & ESA ortaklığı:** Uzay buluşması ve enkaz yakalama (Rendezvous & Debris Capture) senaryoları için oluşturulmuştur.
* Donanım tabanlı test ortamında (Prisma Tango uydusu ölçekli modeli) gerçek Güneş lambası ve Dünya parıltısı ile oluşturulmuş yüksek kontrastlı uzay fotoğraflarını içerir.
* Sistemimizin yalnızca çizgisel izler (streak) değil, optik sensöre yakın geçen gövde/güneş paneli parçaları gibi **yakın mesafe 2D/3D enkaz geometrilerini** de başarıyla tespit edebildiğini doğrulamak için doğrulama seti olarak kullanılacaktır.

---

### 5. LİTERATÜRDEKİ EKSİKLİKLER (RESEARCH GAPS) VE PROJEMİZİN ÖZGÜN DEĞERİ

| Mevcut Literatürdeki Durum | Mevcut Eksiklik / Darboğaz | Projemizin Sunduğu Çözüm ve Yenilik |
| :--- | :--- | :--- |
| **Geleneksel Morfolojik / Şablon Eşleme** | Gürültülü ve kutup ışıklı ortamlarda aşırı yanlış alarm (false positive) üretir. | Derin öğrenme tabanlı öznitelik çıkarımı ile gürbüz tespit. |
| **Ağır Derin Öğrenme Modelleri (SDT-Net vb.)** | Sunucu GPU'larına ihtiyaç duyar; uydu SWaP-C kısıtlarını aşar. | Model budama, INT8 kuantizasyon ve hafif YOLO-Nano mimarisi. |
| **Downlink Bağımlı Çözümler** | Görüntüleri yere indirmek saatler sürer; çarpışma anında müdahale edilemez. | On-board gerçek zamanlı tespit (<60 ms gecikme) ve anlık manevra telemetrisi. |
| **Standart Nesne Tespiti (YOLO)** | Standart YOLOv8/v11 modelleri 3x3 piksellik mikro nesneleri downsampling'de kaybeder. | **P2 Sığ Tespit Başlığı (P2-Head)** entegrasyonu ile mikro enkaz hassasiyeti. |

---

### 6. SONUÇ VE ÇIKARIMLAR

İncelenen literatür, uydularda uzay enkazı tespiti için:
1. Görüntü ön işlemede kozmik ışın ve yıldız arka planının elenmesinin tespit başarımını doğrudan artırdığını,
2. Yörünge güvenliği için tespitin tek başına yetersiz olup **çoklu nesne takibi (MOT) ve bağıl yörünge kestiriminin** kaçınılmaz olduğunu,
3. Fergani Uzay gibi takımyıldızı vizyonuna sahip kurumlar için **Watt başına düşen FPS ve bant genişliği tasarrufunun** projenin en temel ticari ve operasyonel değeri olduğunu doğrulamaktadır.
