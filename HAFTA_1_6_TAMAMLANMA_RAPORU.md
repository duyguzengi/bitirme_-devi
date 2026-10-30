# 🛰️ BİTİRME PROJESİ İLERLEME RAPORU: 1. - 6. HAFTA TAMAMLANMA DÖKÜMANI
## LEO Uyduları için Uç Yapay Zekâ Tabanlı Uzay Enkazı Algılama ve Takip Sistemi

**Rapor Tarihi:** 2026  
**Durum:** 1. - 6. Hafta Görevleri Eksiksiz Tamamlandı (Testler Başarılı)  
**İlgili Modüller:** `docs/`, `src/preprocessing/`, `src/dataset/`, `src/utils/`, `tests/`  

---

### 1. YÖNETİCİ ÖZETİ

Bu rapor, Alçak Dünya Yörüngesi (LEO) uyduları için uç yapay zekâ (Edge AI) tabanlı uzay enkazı tespiti projesinin ilk 6 haftalık çalışma döneminin teknik çıktılarını sunmaktadır. Bu aşamada:
1. Klasörde bulunan iki temel akademik kaynak (`observation.pdf` ve 2025 tarihli `SDTD` makalesi) derinlemesine taranarak kapsamlı bir **Literatür Taraması Dokümanı** (`docs/LITERATUR_TARAMASI.md`) hazırlanmıştır.
2. Sistemin uçtan uca mimarisini ve kabul kriterlerini belirleyen **Sistem ve Yazılım Gereksinimleri Şartnamesi (SRS)** (`docs/SISTEM_GEREKSINIMLERI_SRS.md`) tamamlanmıştır.
3. Uzay görüntüleri için kritik olan **ön işleme boru hattı** (Kozmik ışın/hot pixel filtreleme, morfolojik ve zamansal star-removal, ZScale ve CLAHE kontrast artırma) sıfırdan Python ve OpenCV ile kodlanmıştır.
4. SDTD makalesindeki fiziksel hareket ve optik PSF difüzyon denklemlerini birebir uygulayan **Sentetik Uzay Enkazı Simülatörü** geliştirilmiştir.
5. Etiket formatları derin öğrenme standartlarına uygun olarak **YOLOv8/v11 (`<class> <x> <y> <w> <h>`)** ve **MOTChallenge (`<frame>,<id>,<x>,<y>,<w>,<h>`)** formatlarına dönüştürülmüştür.
6. Tüm boru hattı uçtan uca otomatik test edilmiş (`tests/test_pipeline.py`) ve yerel bir veri seti kıyaslama ortamı (`data/processed/yolo_debris_benchmark/`) kurulmuştur.

---

### 2. LİTERATÜR TARAMASI VE KURAMSAL BULGULAR (ÖZET)

* **Makale 1 (`observation.pdf` - Duggan et al., IEEE Survey):**
  * Uydularda on-board computing için en kritik kısıtın **SWaP-C (Size, Weight, Power and Cost)** ve özellikle **güç bütçesi (<10W)** ile **vakumda termal yayılım** olduğu saptanmıştır.
  * Uydularda çıkarım için INT8 kuantizasyon, filtre budama (pruning) ve hafif mimarilerin (YOLO-Nano, MobileNet) COTS donanımlarda (NVIDIA Jetson, Hailo-8, Google Coral) kullanılmasının en verimli yol olduğu teyit edilmiştir.
  * Uyduda yapay zekâ çalıştırmanın downlink veri trafiğini %99'un üzerinde azalttığı vurgulanmıştır.
* **Makale 2 (`SDTD Paper` - Zhuang, Song et al., Shanghai AI Lab 2025):**
  * Uzay enkazlarının uzun pozlamada dairesel değil, optik **çizgisel iz (streak / line source)** şeklinde göründüğü ve Point Spread Function (PSF) difüzyonuna uyduğu saptanmıştır.
  * Astronomik sensörlerin dinamik aralığını korumak için makalede tanımlanan **ZScale ($k=2.5$) formülü** kodlanarak projeye dahil edilmiştir.
  * SDTD veri setinin zorlu kutup istasyonu arka planlarında %73.2 MOTA elde ettiği görülmüş; ancak önerilen SDT-Net modelinin sunucu tipi GPU gerektirmesi nedeniyle uydumuzda çalıştırılmak üzere **P2-Head destekli hafif bir YOLOv11n** ile ikame edilmesine karar verilmiştir.
* **SPEED+ Veri Seti (Stanford SLAB & ESA):**
  * Sert uzay ışıklandırması (Güneş parıltısı ve derin uzay gölgeleri) altında nesne tespiti testleri için veri seti yöneticisine entegre edilmiştir.

---

### 3. SİSTEM GEREKSİNİMLERİ (SRS) ÖZETİ

* **Fonksiyonel:** Görüntü alma (FR-01), Hot pixel temizleme (FR-02), ZScale/CLAHE kontrast artırma (FR-03), Yıldız arka plan çıkarımı (FR-04), P2 mikro enkaz tespiti (FR-05), ByteTrack ile çoklu nesne takibi (FR-06), Bağıl hız ve TTC analizi (FR-07, FR-08), Otonom telemetri üretimi (FR-09), TensorRT INT8 uç optimizasyonu (FR-10).
* **Performans Hedefleri (NFRs):**
  * Çıkarım Hızı: $\ge$ 20 FPS (Jetson Orin Nano / Nano üzerinde)
  * Toplam Gecikme: $\le$ 50–60 ms
  * Güç Bütçesi: $\le$ 10 Watt
  * Model Boyutu: $\le$ 15 MB
  * Tespit Doğruluğu: mAP@50 $\ge$ %80
  * Bant Genişliği Tasarrufu: $\ge$ %99.9

---

### 4. GELİŞTİRİLEN YAZILIM MODÜLLERİ (KOD ENVANTERİ)

| Dosya Yolu | Modül Adı | İşlevi ve Teknik Detayı |
| :--- | :--- | :--- |
| `src/preprocessing/cosmic_ray_filter.py` | Kozmik Işın Filtresi | Radyasyon kaynaklı tekil sıcak pikselleri (hot pixels) MAD ve sigma-clipping ile debris izlerini bozmadan temizler. |
| `src/preprocessing/clahe_enhancer.py` | ZScale & CLAHE İyileştirici | SDTD denklem 1-3'e göre ZScale dinamik aralık dönüşümü ve yerel kontrast eşitlemesi (CLAHE) uygular. |
| `src/preprocessing/star_removal.py` | Yıldız Eleme & Arka Plan Çıkarıcı | Morfolojik Top-Hat filtreleme ve ardışık kareler arası zamansal fark (frame differencing) ile sabit yıldızları siler. |
| `src/dataset/synthetic_debris_generator.py` | Uzay Enkazı Simülatörü | SDTD denklem 4-6'daki fiziksel yörünge yayılımı ($v, \theta, dt$) ve PSF Gaussian difüzyonu ile gerçekçi uzay kareleri ve etiketleri üretir. |
| `src/dataset/label_converter.py` | Etiket Standardizasyonu | Çizgi uç noktaları (endpoints) ve koordinatları normalize YOLO (`<c> <x> <y> <w> <h>`) ve MOT Challenge formatına dönüştürür. |
| `src/dataset/dataset_downloader.py` | Veri Seti Yöneticisi | SPEED+ ve SDTD kaynaklarını yönetir; tek komutla yerel eğitim/doğrulama veri setini bootstrap eder. |
| `src/utils/visualizer.py` | Görsel Teşhis Aracı | Ön işleme aşamalarını 4'lü karşılaştırma ızgarasında (grid) birleştirir ve tespit kutularını çizer. |
| `tests/test_pipeline.py` | Otomatik Doğrulama Testi | Tüm modülleri uçtan uca test eden ve başarı durumunu onaylayan test otomasyonu. |

---

### 5. TEST VE DOĞRULAMA ÇIKTILARI

`python3 tests/test_pipeline.py` çalıştırılmış ve tüm aşamalar 0 hata ile doğrulanmıştır:
* **Görsel Çıktılar:**
  * `data/sample_demo/pipeline_stages_grid.jpg`: Ham Görüntü $\rightarrow$ Hot Pixel Filtresi $\rightarrow$ Yıldız Eleme $\rightarrow$ ZScale & CLAHE aşamalarını gösteren 4'lü karşılaştırma görseli oluşturuldu.
  * `data/sample_demo/annotated_frame_0.jpg`: Tespit edilen enkaz izi ve sınırlayıcı kutu görselleştirildi.
* **Veri Seti Hazırlığı:**
  * `data/processed/yolo_debris_benchmark/` altında eğitime hazır `data.yaml`, `images/train`, `images/val`, `labels/train`, `labels/val` klasör yapısı oluşturuldu.

---

### 6. GELECEK AŞAMA PLANI (7. - 10. HAFTALAR)

Ön işleme ve veri standardizasyonu tamamlandığına göre bir sonraki dönüm noktaları şunlardır:
1. **Temel Model Eğitimi (Baseline Training):** Standart `YOLOv8n` / `YOLOv11n` modelinin uzay enkazı verisi üzerinde eğitilmesi ve mAP50 referans değerinin ölçülmesi.
2. **P2 Head Mimari Modifikasyonu:** 3x3 piksellik mikro enkazların tespitini sağlamak için YOLO omurgasına 160x160 piksel ölçekli sığ P2 tespit katmanının eklenmesi.
3. **Karşılaştırmalı Analiz:** Standart YOLO vs. P2-Modifiyeli YOLO doğruluk ve hız farklarının raporlanması.
