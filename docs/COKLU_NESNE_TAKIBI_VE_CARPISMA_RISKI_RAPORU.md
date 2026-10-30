# 🛰️ ÇOKLU NESNE TAKİBİ (MOT) VE ÇARPIŞMA RİSKİ ANALİZİ (TTC) RAPORU
## P2-Head YOLO11n + ByteTrack + Kalman Filtresi Tabanlı Yörünge Güvenliği ve Erken Uyarı Katmanı

**Doküman Kodu:** REP-SPACE-MOT-TTC-01  
**Tarih:** 2026  
**Hedef Sistem:** Fergani LEO Takımyıldızı ve Küp/Mikro Uydu Aviyonik Platformları  
**Geliştirilen Modüller:**
* `src/tracking/kalman_filter.py` (8 Boyutlu Kinematik Durum Kestirimi)
* `src/tracking/byte_tracker.py` (Uzay Işık Eğrileri için 2 Kademeli ByteTrack)
* `src/tracking/collision_risk.py` (CPA, Teğet Mesafesi, TTC ve Kaçınma Manevrası)
* `src/tracking/video_pipeline.py` (Kokpit HUD Görselleştirici ve Telemetri Üreteci)
* `tests/test_tracking_pipeline.py` (Uçtan Uca Doğrulama Testi)

---

### 1. GİRİŞ VE SİSTEM MİMARİSİ

Bir LEO uydusunun yörüngede hayatta kalabilmesi için yalnızca nesneleri tekil karelerde tespit etmesi (object detection) yetersizdir. Yörüngedeki bağıl hızların $7 - 10\text{ km/s}$ mertebesinde olması nedeniyle:
1. Sabit göksel yıldızlar ile bağıl hareketi olan enkazların ayırt edilmesi,
2. Enkazın dönme hareketinden (tumbling / light curve dalgalanması) kaynaklanan parlaklık düşüşlerinde izin kaybedilmemesi,
3. Enkazın uydunun güvenlik balonuna (Safety Bubble) girip girmeyeceğinin ve Çarpışma Zamanının (Time-to-Collision - TTC) hesaplanması,
4. Kritik durumda uydunun itki sistemine anlık **Kaçınma Manevra Vektörü ($\mathbf{\Delta V}$)** telemetrisinin iletilmesi zorunludur.

```
[ P2-Head YOLO11n Tespitleri ] (Kutu, Skor, Sınıf)
               │
               ▼
┌────────────────────────────────────────────────────────┐
│             UZAY BYTETRACK TAKİP KATMANI               │
│                                                        │
│  1. Kalman Filtresi Durum Tahmini (Konum + Hız)        │
│  2. Kademe 1: Yüksek Güvenilirlikli Eşleştirme (IoU)  │
│  3. Kademe 2: Düşük Güvenilirlikli Eşleştirme          │
│     (Tumbling nedeniyle sönen enkazların kurtarılması) │
│  4. Kimlik (ID) Atama ve Yörünge İzi Kaydı             │
└────────────────────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────┐
│        ÇARPIŞMA RİSKİ VE TTC ANALİZ MOTORU             │
│                                                        │
│  • En Yakın Geçiş Zamanı: t_CPA = - (r · v) / ||v||^2  │
│  • Teğet Geçme Uzaklığı: d_miss = ||p(t_CPA) - c_opt|| │
│  • Tehdit Seviyesi Sınıflandırması (CRITICAL / WARNING)│
│  • İtki Kaçınma Vektörü Hesabı (u_avoid ⊥ v_rel)       │
└────────────────────────────────────────────────────────┘
               │
               ▼
[ Kokpit HUD Video Çıktısı ] & [ 1 KB Otonom Telemetri Paketi ]
```

---

### 2. MATEMATİKSEL VE KİNEMATİK FORMÜLASYON

#### 2.1. Kalman Filtresi Durum Vektörü
Enkazın görüntü düzlemindeki hareketi Sabit Hızlı (Constant Velocity - CV) 8 boyutlu durum vektörü ile modellenir:
$$\mathbf{x} = \begin{bmatrix} x_c & y_c & a & h & \dot{x}_c & \dot{y}_c & \dot{a} & \dot{h} \end{bmatrix}^T$$
* $(x_c, y_c)$: Sınırlayıcı kutu merkezi
* $a = w/h$: En-boy oranı (aspect ratio)
* $h$: Kutu yüksekliği
* $(\dot{x}_c, \dot{y}_c)$: Piksel/saniye cinsinden bağıl hız vektörü

Durum geçiş matrisi $\mathbf{F}$ ve ölçüm matrisi $\mathbf{H}$ ile zaman adımı $\Delta t = 1/\text{FPS}$ üzerinden kestirim ve düzeltme (predict/update) adımları icra edilir.

#### 2.2. En Yakın Geçiş Noktası (Closest Point of Approach - CPA)
Uydunun optik merkezi $\mathbf{c} = (W/2, H/2)$ referans alındığında, enkazın optik merkeze olan bağıl konum vektörü $\mathbf{r} = \mathbf{p} - \mathbf{c}$ ve bağıl hız vektörü $\mathbf{v}$ olmak üzere:
$$t_{\text{CPA}} = -\frac{\mathbf{r} \cdot \mathbf{v}}{\|\mathbf{v}\|^2}$$

* **$t_{\text{CPA}} \le 0$ Durumu:** Enkaz uydudan uzaklaşmaktadır (`DEPARTING_CLEAR`). Çarpışma riski yoktur.
* **$t_{\text{CPA}} > 0$ Durumu:** Enkaz yaklaşmaktadır. En yakın geçiş mesafesi (Miss Distance):
$$d_{\text{miss}} = \|\mathbf{p}(t_{\text{CPA}}) - \mathbf{c}\| = \sqrt{\|\mathbf{r}\|^2 - \frac{(\mathbf{r} \cdot \mathbf{v})^2}{\|\mathbf{v}\|^2}}$$

#### 2.3. Tehdit Seviyeleri ve Karar Mekanizması
* **KRİTİK ÇARPIŞMA ALARMI (`CRITICAL_COLLISION_ALERT`):**
  $$t_{\text{CPA}} \le 4.5\text{ s} \quad \text{ve} \quad d_{\text{miss}} \le R_{\text{safety}} \quad (R_{\text{safety}} = 75\text{ px})$$
* **ÇARPIŞMA UYARISI (`COLLISION_WARNING`):**
  $$t_{\text{CPA}} \le 10.0\text{ s} \quad \text{ve} \quad d_{\text{miss}} \le R_{\text{caution}} \quad (R_{\text{caution}} = 150\text{ px})$$
* **GÜVENLİ GEÇİŞ (`APPROACHING_SAFE`):** $d_{\text{miss}} > R_{\text{caution}}$
* **UZAKLAŞAN (`DEPARTING_CLEAR`):** $t_{\text{CPA}} \le 0$

#### 2.4. Otonom Kaçınma Manevra Vektörü ($\mathbf{\Delta V}$)
Kritik bir çarpışma rotasında ($d_{\text{miss}} < R_{\text{safety}}$), uydunun itki motorunun ateşlemesi gereken en az yakıtla en yüksek kaçınma mesafesini sağlayan yön, bağıl hız vektörüne dik olan birim vektördür:
$$\mathbf{u}_{\text{avoid}} = \frac{[-v_y, v_x]^T}{\|\mathbf{v}\|}$$

---

### 3. DOĞRULAMA SENARYOSU VE TEST SONUÇLARI

[`tests/test_tracking_pipeline.py`](file:///Users/pc/Desktop/bitirme%20o%CC%88devi/tests/test_tracking_pipeline.py) betiği ile kontrollü bir uzay karşılaşma senaryosu koşturulmuştur:
* **Enkaz #1 (Tehdit Nesnesi):** $(140, 110)$ konumundan optik merkeze $(320, 320)$ doğru doğrudan çarpmak üzere ilerleyen $v=9\text{ px/frame}$ hızında enkaz.
* **Enkaz #2 (Güvenli Geçiş):** Sağ kenardan aşağıya doğru teğet geçen $v=11\text{ px/frame}$ hızında zararsız enkaz.

#### Elde Edilen Doğrulama Çıktıları:
* **İşlenen Toplam Kare Sayısı:** 25 Kare
* **Kritik Alarm Üretilen Kare Sayısı:** 6 Kare
* **Tespit Edilen Çarpışma Parametreleri (Örnek Telemetri):**
  * **Takip Kimliği (ID):** 3
  * **Tehdit Derecesi:** `CRITICAL_COLLISION_ALERT`
  * **Hesaplanan Time-to-Collision (TTC):** **4.2 saniye**
  * **Hesaplanan Teğet Geçme Uzaklığı:** **0.4 piksel** (Doğrudan çarpışma rotası!)
  * **Bağıl Hız Vektörü:** $[+31.0, +36.5]\text{ px/s}$
  * **Önerilen Otonom İtki Yönü:** $\mathbf{u}_{\text{avoid}} = [-0.762, +0.648]$
* **Güvenli Enkaz Davranışı:** Enkaz #2 başarıyla takip edilmiş, teğet mesafesi $190+\text{ px}$ olduğu için alarm tetiklenmeden `APPROACHING_SAFE` olarak izlenmiştir.

---

### 4. ÜRETİLEN TELEMETRİ PAKETİ STANDARDI (ÖRNEK JSON)

Sistem tarafından uydunun aviyonik veriyoluna (UART/CAN-bus) ve yer istasyonuna gönderilen kompakt telemetri paketi:

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

### 5. OLUŞTURULAN ÇIKTILAR VE DOSYA LİSTESİ

1. **Takip ve Çarpışma Göstergeli Video:** [`output/tracked_debris_collision_demo.mp4`](file:///Users/pc/Desktop/bitirme%20o%CC%88devi/output/tracked_debris_collision_demo.mp4)
   * Kırmızı güvenlik balonu ($R=75\text{ px}$), sarı uyarı koridoru ($R=150\text{ px}$), optik nişangah.
   * Enkaz kimlikleri, anlık hız ve TTC etiketleri, geçmiş yörünge kuyruğu ve hız okları.
   * Tehdit durumuna göre dinamik değişen üst HUD alarm bandı ve sol alt aviyonik telemetri kutusu.
2. **Kritik Alarm Önizleme Görseli:** [`output/hud_critical_collision_alert.jpg`](file:///Users/pc/Desktop/bitirme%20o%CC%88devi/output/hud_critical_collision_alert.jpg)
3. **Nominal Gözlem Önizleme Görseli:** [`output/hud_nominal_surveillance.jpg`](file:///Users/pc/Desktop/bitirme%20o%CC%88devi/output/hud_nominal_surveillance.jpg)
4. **Uçuş Telemetri Günlüğü:** [`output/flight_telemetry_log.json`](file:///Users/pc/Desktop/bitirme%20o%CC%88devi/output/flight_telemetry_log.json)
