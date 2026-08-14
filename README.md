# TFF Fantezi Lig Takım Öneri Aracı

Süper Lig formu, TFF fiyatları ve dış lig istatistiklerini birleştirerek **100M / 15'li** kadro önerir. Diziliş (4-4-2, 4-5-1, 3-4-3 …), ilk 11, yedekler ve kaptan otomatik seçilir.

## Hızlı başlangıç

1. Python 3.10+ yüklü olsun.
2. TFF hesabını `data/tff_login.txt` dosyasına yaz (örnek: `data/tff_login.example.txt`).
3. `calistir.bat` çalıştır.

```bat
calistir.bat
```

Çıktı: diziliş, ilk 11, yedekler, kaptan, maliyet.

## Giriş dosyası

`data/tff_login.txt` (git'e eklenmez):

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
- Sofascore sezon/form verisi + FotMob ikinci kaynak doğrulaması
- FotMob ilk 11, xG/xA, şut ve güncel hazırlık/resmî kulüp maçları
- Mevcut sezon az maçlıysa veri atılmaz; önceki sezonla örnek büyüklüğüne göre karıştırılır
- Haftanın rakibi, iç/dış saha ve rakibin hücum/savunma gücü
- Yeni transferler: son 1–2 lig sezonu + hazırlık maçı dakikası (varsa)
- Dış lig oranları sabit tahminle değil, geçmişte o ligden ilk kez Süper Lig'e
  gelen oyuncuların iki taraftaki dakika başı performansıyla kalibre edilir
- Sezon başladıktan sonra resmî TFF puanı/BPS alanlarıyla düşük ağırlıklı kalibrasyon
- Sakat/cezalı oyuncular TFF `availabilityStatus` ile kırpılır
- Optimizer en iyi dizilişi seçer; yedekler otomatik değişim için ağırlıklı

## Lig dönüşüm kalibrasyonu

`data/league_translation.json`, geçmiş dış lig → Süper Lig geçişlerinden üretilen
çalışma zamanı modelidir. Yalnızca kaynak ve hedef sezonda en az 450 dakika oynayan,
Süper Lig'e ilk kez gelen oyuncular kullanılır. Mevki ve metrik ayrı modellenir;
az örnekli ligler mevki geneline küçültülür. Küçültme miktarı ve model etkisi son
üç sezon üzerinde ileri-zaman doğrulamasıyla seçilir. Turnuva ID'leri canlı
Sofascore `tier` metadatasıyla doğrulanır; eksik eski xG/xA alanları sıfır sayılmaz.
Lig örneği yoksa aynı seviye liglerin (üst uçuş / ikinci kademe) modelleri
kullanılır. Greenwood gibi yüksek tempolu yıldızlar ortalama transfer eğimine
çekilmez; kaynak G/A oranına doğru karıştırılır. 10 dakikalık form cameo'su
sezon üretimini ezmez; geçen sezonun tam TFF puan özeti yeni sezonu çift saymaz.

Veriyi yenilemek için:

```bat
python calibrate_leagues.py
```

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
