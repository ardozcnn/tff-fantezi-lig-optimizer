# TFF Fantezi Lig Takım Öneri Aracı

TFF fiyatları ve dış lig istatistiklerini birleştirerek **100M / 15'li** kadro önerir. Diziliş (4-4-2, 4-5-1, 3-4-3 …), ilk 11, yedekler ve kaptan otomatik seçilir.

## Hızlı başlangıç

1. Python 3.10+ yüklü olsun.
2. TFF hesabını `data/tff_login.txt` dosyasına yaz (örnek: `data/tff_login.example.txt`).
3. `calistir.bat` çalıştır.

```bat
calistir.bat
```

Çıktı: diziliş, ilk 11, yedekler, kaptan, maliyet.

## Giriş dosyası

`data/tff_login.txt`:

```
email=senin@mail.com
password=sifren
```

Alternatif: ortam değişkenleri `TFF_EMAIL` ve `TFF_PASSWORD`.

## Komut satırı

```bat
python -m src.main              rem sade kadro
python -m src.main --verbose    rem ayrıntılı log
python -m src.main --no-fetch-prices   rem kayıtlı prices.csv
python -m src.main --refresh-cache     rem Sofascore önbelleğini temizle
python -m src.main --export-stats out.csv   rem tam analiz tablosu
```

## Analiz (arka planda)

- TFF resmi puan kuralları: dakika (60+ = 2p), gol/asist, CS, kart, bonus, penaltı
- Süper Lig form (son 6 maç) + sezon bazı (xG/xA, şut, kilit pas)
- Yeni transferler: son 1–2 lig sezonu + hazırlık maçı dakikası (varsa)
- Sakat/cezalı oyuncular TFF `availabilityStatus` ile kırpılır
- Optimizer en iyi dizilişi seçer; yedekler otomatik değişim için ağırlıklı

## Kadro kuralları

| Kural | Değer |
|--------|--------|
| Bütçe | 100M TL |
| Kadro | 2 GK, 5 DF, 5 MF, 3 FW |
| Aynı takım | en fazla 3 |

## Güvenlik

`data/tff_login.txt`, `data/tff_cookies.txt`, `data/prices.csv` ve `data/cache/` **commit edilmemeli** (`.gitignore` içinde).

## Uyarı

Bu araç resmî TFF uygulaması değildir; öneri amaçlıdır.
