# TFF Fantezi Lig Takım Öneri Aracı

Bu proje, TFF Fantezi Lig için haftalık kadro önerisi hazırlar. Oyuncu fiyatlarını, güncel formu, sezon istatistiklerini ve fikstür zorluğunu birlikte değerlendirir; 100 milyon TL bütçeye uyan 15 kişilik kadroyu seçer. Diziliş, ilk 11, yedekler ve kaptan ayrıca hesaplanır.

## Kurulum

Python 3.10 veya daha yeni bir sürüm gereklidir. TFF hesap bilgileri, örneği verilen `data/tff_login.txt` dosyasına yazılmalıdır:

```text
email=senin@mail.com
password=sifren
```

Bu dosya GitHub'a gönderilmez. Hesap bilgileri istenirse `TFF_EMAIL` ve `TFF_PASSWORD` ortam değişkenleriyle de verilebilir.

Windows'ta projeyi çalıştırmanın en kısa yolu:

```bat
calistir.bat
```

## Kullanım

```bat
python -m src.main
python -m src.main --verbose
python -m src.main --no-fetch-prices
python -m src.main --refresh-cache
python -m src.main --export-stats out.csv
python -m src.main --report-png data/weekly_report.png
```

İlk komut yalnızca kadro sonucunu gösterir. Diğer seçeneklerle ayrıntılı kayıt alınabilir, kayıtlı fiyatlar kullanılabilir, Sofascore önbelleği yenilenebilir veya analiz sonucu CSV ve PNG olarak dışarı aktarılabilir.

## Hesaplama yöntemi

Oyuncular TFF'nin resmî puan kurallarına göre değerlendirilir. Dakika, gol, asist, temiz sayfa, yenilen gol, kart, penaltı ve bonus puanları tahmine dâhildir.

Sezon ve son maç verileri Sofascore'dan alınır; FotMob verileri ilk 11 durumu, xG, xA, şut ve güncel kulüp maçları için ikinci kaynak olarak kullanılır. Haftanın rakibi, iç veya dış saha durumu ve rakibin hücum-savunma gücü de tahmini etkiler.

Sezonun ilk haftalarında tek bir maçın sonucu gereğinden fazla belirleyici olmaz. Gol ve clean sheet gibi nadir sonuçlar temkinli tutulur; kaleci kurtarışları gibi sayım verileri ise daha hızlı ağırlık kazanır. Yeni sezon verisi, oynanan maç sayısına göre önceki sezonla dengelenir.

Yeni transferlerde son lig sezonları ve mevcutsa hazırlık maçlarındaki dakikalar dikkate alınır. Dış ligden gelen oyuncular için sabit bir katsayı kullanılmaz. Geçmişte Süper Lig'e gelen oyuncuların transfer öncesi ve sonrası dakika başına üretimleriyle lig ve mevki bazında dönüşüm hesaplanır. Oyuncu Süper Lig'de ilk maçını oynadıysa dış lig tahmini bu açılışla karıştırılır; tamamen silinmez.

Sakat, cezalı veya kadro dışı oyuncular optimizasyona alınmaz. Şüpheli durumdaki oyuncular ise oynama riskine göre daha düşük puanlanır. Her oyuncu için oynama olasılığı ve oynarsa beklenen puan ayrı tutulur.

## Kadro seçimi

Optimizasyon şu kurallara uyar:

- 100 milyon TL bütçe
- 2 kaleci, 5 savunmacı, 5 orta saha ve 3 forvet
- Bir kulüpten en fazla 3 oyuncu
- Geçerli ilk 11 dizilişlerinden biri

PuLP ile kurulan tamsayı programlama modeli yasal kadroyu seçer. İlk 11 ve yedek değeri, TFF'nin otomatik değişiklik kuralına göre hesaplanır: oynamayan bir ilk-11 oyuncusunun yerine, diziliş en az 1 kaleci / 3 defans / 1 forvet kalacak şekilde yedekten sırayla oyuncu girer. Kaleci yalnız kaleciyle değişir. Sonuçta en yüksek beklenen puana sahip yasal kadro, kaptan ve uygun diziliş seçilir.

## Haftalık rapor ve menajer kartları

`data/weekly_report.png` dosyasında ilk 11, yedekler, rakipler, kaptan ve haftanın menajer kartı kararı yer alır. Kartın beklenen getirisi yeterli değilse sistem kartı kullanmak yerine saklamayı önerir.

## Lig dönüşüm verisini yenileme

```bat
python calibrate_leagues.py
```

Bu işlem `data/league_translation.json` dosyasını geçmiş transfer verileriyle yeniden oluşturur.

## Not

Bu proje TFF'nin resmî bir ürünü değildir. Üretilen kadrolar istatistiksel tahmindir ve karar desteği amacıyla sunulur.
