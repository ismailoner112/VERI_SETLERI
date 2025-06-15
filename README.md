# Mobil Güvenlik ve Tehdit Analizi Projesi

Bu proje, mobil cihazlar ve IoT cihazları için güvenlik analizi ve tehdit tespiti yapan bir makine öğrenmesi sistemidir.

## Proje Yapısı

- `api.py`: Ana API endpoint'leri
- `frontend/`: Web arayüzü dosyaları
- `unified_models/`: Birleştirilmiş model dosyaları
- `IoT_Intrusion/`: IoT saldırı veri seti
- `Mobile_Security_Dataset/`: Mobil güvenlik veri seti
- `IOT_Malware_dataset/`: IoT zararlı yazılım veri seti
- `Android_Malware/`: Android zararlı yazılım veri seti

## Kurulum

1. Python 3.8 veya üstü sürümü yükleyin
2. Sanal ortam oluşturun:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. Gerekli paketleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

## Kullanım

1. API'yi başlatın:
   ```bash
   python api.py
   ```
2. Web arayüzüne erişin:
   ```
   http://localhost:5000
   ```

## Özellikler

- Mobil cihaz güvenlik analizi
- IoT cihaz tehdit tespiti
- Android zararlı yazılım tespiti
- Gerçek zamanlı tehdit analizi
- Web tabanlı kullanıcı arayüzü

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. 