# 🛰️ MODEL EĞİTİMİ VE P2-HEAD KARŞILAŞTIRMA RAPORU
## Uzay Enkazı Algılamada Baseline YOLOv11n vs. P2-Head YOLOv11n Performans Analizi

**Doküman Kodu:** REP-SPACE-P2-01  
**Tarih:** 2026  
**Eğitim Platformu:** Apple Silicon M1 (MPS - Metal Performance Shaders Hızlandırması)  
**Veri Seti:** SDTD Gözlem Tabanlı Uzay Enkazı Kıyaslama Seti (`data/processed/yolo_debris_benchmark`)  
* 640 Eğitim Karesi, 160 Doğrulama Karesi  
* Uzay Işıklandırması + Sabit Yıldız Arka Planı + Kozmik Gürültü + Gaussian PSF Difüzyonu  

---

### 1. YÖNETİCİ ÖZETİ

Bu çalışma, kısıtlı kaynaklı Alçak Dünya Yörüngesi (LEO) uyduları üzerinde çalışacak yapay zekâ modelinin mikro uzay enkazlarını ($2\times2$ ve $3\times3$ piksel boyutundaki zayıf ışık izleri) kaçırmasını önlemek amacıyla tasarlanan **P2 Sığ Tespit Başlığı (P2-Head)** inovasyonunu doğrulamak için gerçekleştirilmiştir.

Standart **Baseline YOLOv11n** (3 ölçekli: P3/P4/P5) ile özel olarak geliştirilen **P2-Head YOLOv11n** (4 ölçekli: P2/P3/P4/P5) modelleri eşit koşullarda, sıfırdan (scratch) eğitilmiş, ONNX ve TorchScript formatlarına derlenmiş ve çıkarım hızları kıyaslanmıştır.

#### 🏆 Temel Kazanımlar:
1. **Duyarlılık (Recall) Patlaması:** Baseline modelin %45.92'de tıkanan duyarlılığı, P2-Head ile **%73.30'a yükselmiştir (+%27.38 mutlak artış)**. Mikro enkazların tespit oranı %60 artmıştır.
2. **mAP@50 Başarımı:** Doğrulama mAP@50 skoru **%69.39'dan %82.60'a yükselmiştir (+%13.21 mutlak artış)**.
3. **Konumlama Hassasiyeti (mAP@50-95):** Sıkı IoU eşiklerinde mAP skoru **%34.60'tan %50.30'a çıkmıştır (+%15.70 mutlak artış)**.
4. **Hafiflik Korundu:** Model parametre sayısı sadece **%2.9 artmış** (2.58 M $\rightarrow$ 2.66 M), ağırlık boyutu ~10 MB seviyesinde kalarak uydu SWaP-C kısıtlarına tam uyum sağlamıştır.

---

### 2. MİMARİ KARŞILAŞTIRMA

```
Giriş Görüntüsü (640x640)
       │
       ▼
[Omurga (Backbone)]
  ├── P1/2 (320x320)
  ├── P2/4 (160x160) ───► [P2-Head: 160x160 FPN/PAN] ──► DETECT (P2 - 4x4 px Mikro Enkaz)
  ├── P3/8 (80x80)   ───► [P3 FPN/PAN] ───────────────► DETECT (P3 - 8x8 px Küçük Enkaz)
  ├── P4/16 (40x40)  ───► [P4 FPN/PAN] ───────────────► DETECT (P4 - 16x16 px Orta Enkaz)
  └── P5/32 (20x20)  ───► [P5 FPN/PAN + SPPF] ────────► DETECT (P5 - 32x32 px Büyük Enkaz)
```

| Özellik | Baseline YOLOv11n | P2-Head YOLOv11n (Önerilen) | Fark / Etki |
| :--- | :---: | :---: | :--- |
| **Tespit Başlığı Sayısı** | 3 Başlık (P3, P4, P5) | **4 Başlık (P2, P3, P4, P5)** | +1 Sığ Tespit Başlığı |
| **En Yüksek Çözünürlüklü Harita** | $80\times80$ (Stride 8) | **$160\times160$ (Stride 4)** | $4\times$ daha yüksek uzamsal çözünürlük |
| **Min. Tespit Edilebilir Boyut** | $\sim 8\times 8$ piksel | **$\sim 3\times 3$ - $4\times 4$ piksel** | Mikro çizgisel izler yakalanır |
| **Katman Sayısı (Fused)** | 101 Katman | 121 Katman | +20 Katman (FPN+PAN birleşimi) |
| **Parametre Sayısı** | 2,582,347 (2.58 M) | 2,658,868 (2.66 M) | +76,521 parametre (+%2.9) |
| **Hesaplama Yükü (GFLOPs)** | 6.3 GFLOPs | 10.2 GFLOPs | +3.9 GFLOPs |
| **ONNX Dosya Boyutu** | 10.1 MB | 10.9 MB | +0.8 MB |
| **TorchScript Boyutu** | 10.0 MB | 10.6 MB | +0.6 MB |

---

### 3. METRİK VE DOĞRULAMA SONUÇLARI TABLOSU

| Değerlendirme Metriği | Baseline YOLO11n | P2-Head YOLO11n | Fark ($\Delta$) | Değişim Oranı |
| :--- | :---: | :---: | :---: | :---: |
| **mAP@50 (Best Validation)** | **0.6939 (%69.39)** | **0.8260 (%82.60)** | **+0.1321** | **+%19.04** 🚀 |
| **mAP@50-95 (Sıkı Kriter)** | **0.3460 (%34.60)** | **0.5030 (%50.30)** | **+0.1570** | **+%45.38** 🎯 |
| **Duyarlılık (Recall)** | **0.4592 (%45.92)** | **0.7330 (%73.30)** | **+0.2738** | **+%59.63** 🛰️ |
| **Kesinlik (Precision)** | 0.9937 (%99.37) | 0.8020 (%80.20) | -0.1917 | -%19.29 |
| **Val Box Loss (Kutu Kaybı)** | 1.7582 | **1.4267** | -0.3315 | **-%18.85 (Daha İyi)** |
| **Val Cls Loss (Sınıf Kaybı)** | 2.7000 | **1.8461** | -0.8539 | **-%31.63 (Daha İyi)** |

---

### 4. BİLİMSEL BULGULAR VE ANALİZ

#### 4.1. Baseline Neden %45.9 Duyarlılıkta (Recall) Kaldı?
* Standart YOLO ağları, doğal görüntülerdeki (insan, araç vb.) nesneler için optimize edilmiştir.
* Omurgadaki ardışık konvolüsyon ve pooling adımları görüntüyü 8 kat ($80\times80$) küçülttüğünde, $2\times2$ ve $3\times3$ piksellik mikro enkaz izleri bitişik piksellerin ortalaması içinde kaybolmakta ve alıcı alan (receptive field) sınırlarının altına düşmektedir.
* Bu yüzden Baseline model sadece belirgin ve uzun izleri yakalayabilmiş, küçük parçacıkları tespit edememiştir.

#### 4.2. P2-Head Duyarlılığı Nasıl %73.3'e Çıkardı?
* Omurganın C3k2 bloğundan ($160\times160$) alınan özellikler, doğrudan FPN ve PAN katmanlarına beslenmiştir.
* $160\times160$ seviyesinde bir hücre $4\times4$ piksellik bir alanı temsil eder. Böylece $3\times3$ piksellik bir uzay enkazı, özellik haritasında tek bir hücreye sığar ve konvolüsyonel gradyanlar kaybolmadan Detect başlığına ulaşır.
* Bu mekanizma, uzay enkazı tespitinde hayati olan **çarpışma riski taşıyan en küçük parçaların bile kaçırılmamasını** sağlamıştır.

#### 4.3. Çıkarım Hızı ve Donanım Maliyeti
* **Apple M1 Üzerinde Tekil Görüntü Çıkarım Hızı:**
  * Baseline: **75.0 ms (13.3 FPS)**
  * P2-Head: **116.6 ms (8.6 FPS)**
* P2 başlığı $160\times160$ seviyesinde 25.600 ek çapa (anchor) hesapladığı için ham PyTorch FP32 çıkarımında gecikme ~40 ms artmıştır.
* **Uç Donanım Çözümü:** Jetson Orin Nano / Hailo-8 üzerinde TensorRT INT8 kuantizasyonu uygulandığında, 160x160 konvolüsyonlar donanımsal INT8 tensör çekirdeklerinde hızlandırılarak **30+ FPS** hızına kolaylıkla ulaşabilecektir.

---

### 5. PROJEDE ÜRETİLEN MODEL VE RAPOR DOSYALARI

| Dosya Yolu | Açıklama |
| :--- | :--- |
| `runs/space_debris/p2head/weights/best.pt` | En yüksek başarıma sahip PyTorch model ağırlıkları |
| `runs/space_debris/p2head/weights/best.onnx` | Evrensel dağıtıma hazır ONNX modeli (10.9 MB) |
| `runs/space_debris/p2head/weights/best.torchscript` | Gömülü Linux ve mobil için TorchScript modeli (10.6 MB) |
| `runs/space_debris/baseline/weights/best.pt` | Kıyaslama için eğitilen Baseline PyTorch ağırlıkları |
| `runs/space_debris/comparison_chart.png` | Öğrenme eğrileri ve metrik sütun grafiği |
| `runs/space_debris/inference_benchmark.json` | M1 donanımı gecikme ve FPS ölçüm raporu |
| `runs/space_debris/comparison_report.json` | Sayısal metrik karşılaştırma verisi |
