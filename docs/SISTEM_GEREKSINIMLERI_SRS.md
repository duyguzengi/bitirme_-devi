# 📋 SİSTEM VE YAZILIM GEREKSİNİMLERİ ŞARTNAMESİ (SRS)
## LEO Uyduları için Uç Yapay Zekâ Tabanlı Uzay Enkazı Algılama ve Takip Sistemi

**Doküman Kodu:** SRS-SPACE-EDGE-01  
**Versiyon:** 1.0  
**Tarih:** 2026  
**Hedef Kitle:** Bitirme Projesi Jüri Heyeti, Fergani Uzay AR-GE Ekibi, Sistem Mühendisleri  

---

### 1. GİRİŞ VE SİSTEM GENEL BAKIŞI

Bu şartname belgesi, Alçak Dünya Yörüngesi'nde (LEO) görev yapan bir mikro/küp uydunun yerleşik optik sensörlerinden (star tracker veya görev kamerası) gelen görüntü akışını kullanarak, uydu üzerinde gerçek zamanlı olarak uzay enkazlarını tespit eden, takip eden ve çarpışma riski durumunda telemetri üreten sistemin gereksinimlerini tanımlar.

---

### 2. FONKSİYONEL GEREKSİNİMLER (FUNCTIONAL REQUIREMENTS - FR)

* **FR-01 (Sensör Görüntüsü Girişi):** Sistem, tek kanallı (grayscale) veya 3 kanallı (RGB) optik kamera akışından en az 1024x1024 piksel çözünürlükte ardışık kareleri alabilmelidir.
* **FR-02 (Kozmik Işın ve Sıcak Piksel Temizleme):** Sistem, uzay radyasyonunun sensörde oluşturduğu tekil hatalı pikselleri (hot pixels / cosmic ray spikes) median filtreleme ve sigma clipping yöntemi ile ham görüntüden temizlemelidir.
* **FR-03 (Astronomik Kontrast İyileştirme):** Sistem, zayıf ışık izlerini belirginleştirmek için ZScale lineer dinamik aralık dönüşümü ve CLAHE (Contrast Limited Adaptive Histogram Equalization) uygulamalıdır.
* **FR-04 (Yıldız Arka Planının Ayrıştırılması):** Sistem, gök küresindeki sabit arka plan yıldızlarını zamansal fark (frame differencing) ve morfolojik filtrelerle eleyerek yalnızca bağıl hareketi olan enkaz izlerini ön plana çıkarmalıdır.
* **FR-05 (Mikro Enkaz Algılama):** Sistem, görüntü alanında 2x2 piksel boyutundan başlayan küçük nesneleri ve çizgisel enkaz izlerini (streaks) modifiye edilmiş hafif bir derin öğrenme modeli (YOLOv11n + P2 Head) ile tespit etmelidir.
* **FR-06 (Çoklu Nesne Takibi - MOT):** Sistem, tespit edilen her bir enkaza benzersiz bir kimlik (ID) atamalı ve kareler boyunca ByteTrack / Kalman filtresi ile enkazın piksel koordinatlarındaki hareket yolunu (trajectory) sürekli izlemelidir.
* **FR-07 (Bağıl Hız ve Yörünge Kestirimi):** Sistem, ardışık kareler arasındaki piksel yer değiştirmesini ($v_x, v_y$) ve nesne genişleme oranını hesaplayarak uydumuza doğru yaklaşma vektörünü kestirmelidir.
* **FR-08 (Çarpışma Riski ve TTC Analizi):** Sistem, nesnenin optik merkeze yaklaşma hızına göre Çarpışma Zamanını (Time-to-Collision - TTC) ve En Yakın Geçiş Mesafesini (Miss Distance) hesaplamalıdır.
* **FR-09 (Otonom Telemetri Üretimi):** Kritik bir çarpışma riski tespit edildiğinde (TTC < Belirlenen Eşik Değeri), sistem görüntüyü indirmek yerine yalnızca ~1 KB boyutunda CCSDS / JSON formatlı `COLLISION_WARNING` telemetri paketi üretmelidir.
* **FR-10 (Uç Donanım Hızlandırma):** Sistem, eğitilen derin öğrenme ağırlıklarını TensorRT motoruna (.engine) derleyerek gömülü hızlandırıcıda GPU/NPU çekirdeklerini tam kapasite kullanmalıdır.

---

### 3. FONKSİYONEL OLMAYAN GEREKSİNİMLER (NON-FUNCTIONAL REQUIREMENTS - NFR)

| Kod | Gereksinim | Hedef Değer | Doğrulama Yöntemi |
| :--- | :--- | :--- | :--- |
| **NFR-01** | **Çıkarım Hızı (Inference Throughput)** | $\ge$ 20 FPS | Jetson Orin Nano / Nano üzerinde benchmark testi |
| **NFR-02** | **Uçtan Uca Gecikme (End-to-End Latency)** | $\le$ 50 - 60 ms | Ön işleme + Tespit + Takip toplam süresinin kronometresi |
| **NFR-03** | **Güç Tüketimi (Power Consumption)** | $\le$ 10 Watt | `tegrastats` ve `jtop` ile aktif çıkarım sırasındaki güç ölçümü |
| **NFR-04** | **Model Disk Ayak İzi (Model Size)** | $\le$ 15 MB | Kuantize edilmiş `.engine` dosya boyutu kontrolü |
| **NFR-05** | **Bellek (VRAM / RAM) Kullanımı** | $\le$ 1.5 GB | Gömülü Linux bellek yöneticisi izleme |
| **NFR-06** | **Tespit Doğruluğu (mAP@50)** | $\ge$ %80 | SDTD ve SPEED+ test setleri üzerinde hesaplama |
| **NFR-07** | **Çoklu Takip Başarımı (MOTA / HOTA)** | $\ge$ %70 | Standart MOTChallenge metrik kütüphanesi ile ölçüm |
| **NFR-08** | **Bant Genişliği Tasarruf Oranı** | $\ge$ %99.9 | Video dosya boyutu vs. Üretilen Telemetri paketi boyutu |

---

### 4. DONANIM VE ÇEVRESEL ARAYÜZ GEREKSİNİMLERİ (HARDWARE REQUIREMENTS)

#### 4.1. Uç İşlemci (Target Edge AI Hardware)
* **Birincil Hedef Platform:** NVIDIA Jetson Orin Nano Developer Kit (8GB RAM, 40 TOPS AI performansı, 7W–15W güç modu) veya NVIDIA Jetson Nano / Xavier NX.
* **Alternatif Düşük Güçlü Platform:** Raspberry Pi 5 + Raspberry Pi AI Kit (Hailo-8L, 13 TOPS, ~2W–5W ek güç).
* **Masaüstü Geliştirme Platformu:** CUDA destekli NVIDIA GPU (RTX serisi) veya Apple Silicon (MPS / CPU doğrulama ortamı).

#### 4.2. Sensör ve Kamera Arayüzü
* **Optik Sensör Tipi:** CMOS Star Tracker / Monochrome Görev Kamerası.
* **Giriş Arayüzü:** MIPI-CSI2 veya USB 3.0 UVC kamera protokolü.
* **Kare Hızı:** 10 – 30 FPS.

#### 4.3. İletişim ve Veriyolu Arayüzleri
* **Aviyonik Veriyolu:** Uydu Yerleşik Bilgisayarı (OBC) ile iletişim için UART (115200 baud) veya CAN-Bus / Ethernet arayüzü.
* **Telemetri Çıktı Formatı:**
```json
{
  "timestamp_utc": "2026-10-29T15:50:00Z",
  "satellite_id": "FERGANI-SAT-01",
  "alert_level": "CRITICAL_COLLISION_WARNING",
  "debris_id": 42,
  "relative_velocity_pixels_sec": [-12.4, 45.8],
  "time_to_collision_sec": 4.82,
  "bounding_box_norm": [0.452, 0.518, 0.035, 0.028],
  "recommended_maneuver": "THRUST_VECTOR_Z_POS"
}
```

---

### 5. YAZILIM ORTAMI VE KÜTÜPHANE GEREKSİNİMLERİ

* **İşletim Sistemi:** Ubuntu 22.04 LTS (Jetson Linux BSP / JetPack 5.x / 6.x)
* **Programlama Dili:** Python 3.10+ (ve kritik modüller için C++ / CUDA opsiyonu)
* **Kritik Kütüphaneler:**
  * `torch` ($\ge$ 2.0) & `torchvision`
  * `ultralytics` ($\ge$ 8.3)
  * `opencv-python` ($\ge$ 4.8)
  * `numpy` ($\ge$ 1.23) & `scipy` ($\ge$ 1.10)
  * `tensorrt` ($\ge$ 8.6)
  * `filterpy` ($\ge$ 1.4 - Kalman Filtresi için)

---

### 6. DOĞRULAMA VE KABUL KRİTERLERİ (VERIFICATION MATRIX)

| Gereksinim No | Doğrulama Yöntemi | Başarı / Kabul Kriteri |
| :--- | :--- | :--- |
| **FR-02 & FR-04** | Görsel ve SNR İncelemesi | Arka plan gürültüsü ve sabit yıldızlar %85+ oranında elenmeli |
| **FR-05** | Test Seti Değerlendirmesi | SDTD test setinde en az 0.80 mAP@50 başarısı |
| **FR-06** | MOT Değerlendirmesi | En az 50 ardışık kare boyunca kimlik kaybı (ID switch) < 3 olmalı |
| **NFR-01** | FPS Profillemesi | Jetson Orin Nano üzerinde 20+ FPS sürekli çıktı |
| **NFR-03** | Güç Profillemesi | Sistem tam yükte iken çekilen güç 10W'ı aşmamalı |
| **NFR-08** | Telemetri Testi | 1 dakikalık video karşılığında üretilen veri < 50 KB olmalı |
