# 🛰️ Alçak Dünya Yörüngesi (LEO) Uyduları İçin Uç Yapay Zekâ Tabanlı Uzay Enkazı Algılama, Takip ve Çarpışma Kaçınma Sistemi (Space-Edge-AI)

> **Bitirme Projesi (Graduation Capstone Project)**  
> **Uygulama Alanı:** Havacılık ve Uzay, Bilgisayarlı Görü (Computer Vision), Gömülü Sistemler (Edge AI), Yörünge Güvenliği (Space Situational Awareness - SSA)  
> **Hedef Vizyon:** NewSpace mikro ve küp uydu takımyıldızları (Fergani Uzay vizyonu) için yerleşik (on-board) gerçek zamanlı otonom yörünge güvenliği.

---

## 📌 Proje Genel Bakışı (Overview)

Alçak Dünya Yörüngesi'nde (LEO: 200–2000 km) insan yapımı kullanım dışı uyduların ve roket parçalarının hızla artması, zincirleme çarpışmalar riski olan **Kessler Sendromu**'nu kritik boyuta taşımıştır. Geleneksel sistemler uydu fotoğraflarını yer istasyonuna indirmeye (downlink) dayanır; ancak saatler süren gecikme ve dar bant genişliği nedeniyle bağıl hızı 7–10 km/s olan bir enkaz karşısında çaresiz kalır.

Bu projede geliştirilen **Space-Edge-AI** mimarisi:
1. **On-Board Edge AI:** Görüntüleri uyduda anlık işler.
2. **Kozmik Işın & Yıldız Ayrıştırma:** Sıcak pikselleri ve sabit arka plan yıldızlarını filtreler.
3. **P2-Head YOLOv11n (Özel Mimari):** 3x3 piksellik mikro enkaz izlerini yakalamak için 160x160 sığ tespit başlığı eklenmiştir.
4. **Uzay ByteTrack & 8D Kalman Filtresi:** Enkazın dönme hareketinden (tumbling) kaynaklanan parlaklık dalgalanmalarında iz kaybetmeden çoklu nesne takibi (MOT) yapar.
5. **Time-to-Collision (TTC) ve Otonom Kaçınma:** En Yakın Geçiş Zamanı ($t_{CPA}$) ve Teğet Mesafesi ($d_{miss}$) hesaplayarak uydunun itki sistemine anlık kaçınma manevra vektörü ($\mathbf{\Delta V}$) ve yer istasyonuna ~1 KB kompakt telemetri üretir.

---

## 🏆 Temel Başarımlar ve Benchmark Sonuçları

Aynı veri seti ve parametrelerle sıfırdan eğitilen **Baseline YOLOv11n** ile geliştirilen **P2-Head YOLOv11n** modelinin karşılaştırma tablosu:

| Değerlendirme Metriği | Baseline YOLOv11n (P3-P5) | P2-Head YOLOv11n (P2-P5) | Kazanç / Fark ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **mAP@50 (Doğrulama Başarımı)** | %69.39 (`0.6939`) | **%82.60 (`0.8260`)** | **+%19.04 (+0.1321)** 🚀 |
| **mAP@50-95 (Sıkı Kriter)** | %34.60 (`0.3460`) | **%50.30 (`0.5030`)** | **+%45.38 (+0.1570)** 🎯 |
| **Duyarlılık (Recall)** | %45.92 (`0.4592`) | **%73.30 (`0.7330`)** | **+%59.63 (+0.2738)** 🛰️ |
| **Kutu Konumlama Kaybı** | 1.7582 | **1.4267** | **-%18.85 (Daha Hassas)** |
| **Sınıflandırma Kaybı** | 2.7000 | **1.8461** | **-%31.63 (Daha Hızlı)** |
| **Model Ağırlığı (ONNX)** | 10.1 MB | **10.9 MB** | SWaP-C kısıtlarına tam uyum |
| **Apple M1 Çıkarım Hızı** | 13.3 FPS (75.0 ms) | **8.6 FPS (116.6 ms)** | Jetson INT8 ile 30+ FPS |

---

## 📐 Sistem ve Boru Hattı Mimarisi

```
[ Uydu CMOS Optik Sensörü / Star Tracker ]
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│             UYDU İÇİ ÖN İŞLEME KATMANI                 │
│  • Kozmik Işın / Sıcak Piksel Temizleme (MAD Clipping) │
│  • ZScale Kontrast Gerdirme (SDTD Eq. 1-3) & CLAHE     │
│  • Morfolojik Top-Hat & Zamansal Yıldız Eleme          │
└────────────────────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│      P2-HEAD YOLOv11n MİKRO ENKAZ TESPİT MOTORU        │
│  • 160x160 (Stride 4) Sığ Özellik Haritası             │
│  • 4 Ölçekli Tespit: P2 (Mikro), P3, P4, P5            │
│  • 2.66M Parametre, 10.9 MB ONNX                       │
└────────────────────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│      UZAY BYTETRACK & 8D KALMAN ÇOKLU NESNE TAKİBİ     │
│  • 8-Durumlu Kinematik Model: [x, y, a, h, vx, vy...]  │
│  • 2-Aşamalı İlişkilendirme (Dönen enkazları kurtarma) │
└────────────────────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│       ÇARPIŞMA RİSKİ VE TTC ANALİZ MOTORU              │
│  • En Yakın Geçiş Zamanı: t_CPA = - (r · v) / ||v||^2  │
│  • Teğet Uzaklığı: d_miss = ||p(t_CPA) - c_opt||       │
│  • Kaçınma İtki Yönü: u_avoid ⊥ v_rel                  │
└────────────────────────────────────────────────────────┘
                    │
                    ▼
[ Aviyonik İtki Vektörü Telemetrisi: ~1 KB ] ──► [ Yer İstasyonu & Aviyonik ]
```

---

## 📂 Proje Dizin Yapısı

```bash
├── data/
│   ├── processed/yolo_debris_benchmark/ # 640 train + 160 val karelik YOLO veri seti
│   └── sample_demo/                     # Ön işleme aşamaları ve demo görselleri
├── docs/
│   ├── LITERATUR_TARAMASI.md            # observation.pdf & SDTD makalesi analizi
│   ├── SISTEM_GEREKSINIMLERI_SRS.md     # Yazılım Gereksinim Şartnamesi (SRS)
│   ├── MODEL_EGITIMI_VE_P2_KARSILASTIRMA_RAPORU.md # Eğitim ve mAP sonuçları
│   └── COKLU_NESNE_TAKIBI_VE_CARPISMA_RISKI_RAPORU.md # MOT & TTC raporu
├── output/
│   ├── tracked_debris_collision_demo.mp4 # HUD göstergeli video çıktısı
│   ├── hud_critical_collision_alert.jpg  # Kritik alarm anı ekran görüntüsü
│   ├── hud_nominal_surveillance.jpg      # Güvenli uçuş ekran görüntüsü
│   └── flight_telemetry_log.json         # 25 karenin telemetri kayıtları
├── runs/space_debris/
│   ├── p2head/weights/                  # best.pt, best.onnx, best.torchscript
│   ├── baseline/weights/                # Baseline ağırlıkları
│   ├── comparison_chart.png             # Eğitim eğrileri ve kıyaslama grafiği
│   └── inference_benchmark.json         # Gecikme ve FPS raporu
├── src/
│   ├── preprocessing/                   # Kozmik ışın, CLAHE, yıldız eleme modülleri
│   ├── dataset/                         # Sentetik enkaz üreteci, YOLO etiket dönüştürücü
│   ├── models/                          # yolo11n_baseline.yaml, yolo11n_p2head.yaml
│   ├── tracking/                        # Kalman filtresi, ByteTrack, TTC analizörü, HUD
│   ├── train/                           # Eğitim, karşılaştırma ve görselleştirme betikleri
│   └── deploy/                          # ONNX/TorchScript exportu ve benchmark betiği
└── tests/
    ├── test_pipeline.py                 # Ön işleme & veri seti uçtan uca testi
    └── test_tracking_pipeline.py        # P2-Head + ByteTrack + TTC doğrulama testi
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Gereksinimlerin Yüklenmesi
```bash
git clone https://github.com/duyguzengi/bitirme_-devi.git
cd bitirme_-devi
pip install ultralytics torch torchvision opencv-python numpy scipy matplotlib lap
```

### 2. Modelleri Karşılaştırmalı Test Etme
```bash
python3 src/train/compare_models.py
```

### 3. Video Üzerinde Takip ve Çarpışma Analizi (TTC) Çalıştırma
```bash
python3 tests/test_tracking_pipeline.py
```
*Bu komut çalıştırıldığında `output/` dizininde HUD arayüzlü video (`.mp4`), kritik alarm görselleri ve telemetri logu (`flight_telemetry_log.json`) otomatik oluşturulur.*

### 4. Modeli ONNX Formatına Derleme ve Hız Testi
```bash
python3 src/deploy/export_models.py
```

---

## 📡 Üretilen Uçuş Telemetri Paketi Örneği (~500 Bayt)

```json
{
  "timestamp": "2026-10-29T20:42:08.649658Z",
  "satellite_id": "FERGANI-LEO-SAT01",
  "frame_index": 9,
  "overall_status": "CRITICAL_COLLISION_WARNING",
  "active_tracks_count": 2,
  "critical_threats_count": 1,
  "recommended_action": "EXECUTE_IMPULSIVE_BURN_DIRECTION_[-0.76,+0.65]",
  "threats": [
    {
      "debris_id": 3,
      "threat_level": "CRITICAL_COLLISION_ALERT",
      "time_to_collision_sec": 4.2,
      "miss_distance_pixels": 0.4,
      "relative_velocity_pixels_sec": [31.0, 36.5],
      "evasive_vector": [-0.762, 0.648]
    }
  ]
}
```

---

## 📄 Lisans
Bu proje akademik ve araştırma amaçlı geliştirilmiştir.
