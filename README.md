# TFF Fantezi Lig Takım Öneri Aracı

TFF Fantezi Lig’de haftalık kadro kurarken fiyat, form, sezon istatistikleri ve fikstür zorluğunu bir arada değerlendiren bir komut satırı aracıdır. 100 milyon TL bütçeye uyan 15 kişilik kadroyu seçer; diziliş, ilk 11, yedek sırası ve kaptanı da aynı analizden çıkarır.

Bu proje TFF’nin resmî bir ürünü değildir. Üretilen kadrolar istatistiksel tahmindir; karar desteği amacıyla sunulur.

## Kurulum

Python 3.10 veya daha yeni bir sürüm yeterlidir.

TFF hesabınızdan canlı fiyat çekmek için giriş bilgileri gerekir. Örnek dosyayı kopyalayıp kendi bilgilerinizi yazın:

```bat
copy data\tff_login.example.txt data\tff_login.txt
```

`data/tff_login.txt` dosyası GitHub’a gönderilmez. İsterseniz aynı bilgileri `TFF_EMAIL` ve `TFF_PASSWORD` ortam değişkenleriyle de verebilirsiniz.

Bağımlılıkları kurduktan sonra Windows’ta en pratik yol:

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

İlk komut yalnızca kadro sonucunu gösterir. `--verbose` ayrıntılı kayıt alır; `--no-fetch-prices` kayıtlı fiyat dosyasını kullanır; `--refresh-cache` Sofascore önbelleğini yeniler. CSV ve PNG dışa aktarımı isteğe bağlıdır.

Fiyatları çevrimdışı denemek için `data/prices.example.csv` dosyasını `data/prices.csv` olarak kopyalayıp `--no-fetch-prices` bayrağını kullanabilirsiniz. İstatistik adımı yine Sofascore’dan veri ister.

Sofascore ara sıra 403/429 ile geçici engel koyar. Program önce farklı tarayıcı kimlikleriyle ve kısa aralarla tekrar dener; olmazsa bir önceki başarılı çekimin önbelleğini kullanır. `--refresh-cache` bu yedeği siler, engel anında çalıştırmayın. Fikstür Sofascore’dan gelmezse FotMob takvimi devreye girer. 10–20 dakika sonra yeniden denemek genelde yeterlidir.

## Analiz neye dayanıyor?

Her hafta birkaç kaynak birlikte okunur:

**Bu sezon** — Son haftaların formu (L6) ile sezon toplam istatistikleri Sofascore üzerinden gelir. FotMob, ilk 11 durumu, xG, xA, şut ve güncel maç bilgisi için ikinci kaynak olarak kullanılır.

**Geçen sezon** — Oyuncunun Süper Lig geçmişi erken haftalarda daha ağırlıklıdır; sezon ilerledikçe bu pay kendiliğinden azalır. Tek maçlık örneklerin tüm tahmini sürüklemesine izin verilmez.

**Resmî TFF puanları** — TFF’deki dakika, maç sayısı ve maç başı puan, özellikle sezonun ilk haftalarında modeli kalibre eder.

**Fikstür** — Haftanın rakibi, iç veya dış saha ile rakip takımın hücum-savunma gücü beklenen puana yansır. Güç, geçmiş maç skorlarından Poisson modeliyle tahmin edilir; oyuncunun gol/asist payı bu maç beklentisine bağlanır. Kadro seçiminde bu hafta ağır basar, sonraki iki rakip daha düşük ağırlıkla eklenir. Kaptan ve otomatik yedek hâlâ yalnızca bu haftaya bakar.

**Yedek sırası** — TFF’nin otomatik değişik kuralına göre hesaplanır: aynı mevkideki yedek önce gelir, ardından oynama olasılığı × beklenen puan en yüksek olan tercih edilir.

**Menajer kartı** — Sezon boyunca toplam 10 hak vardır. Kalan hafta ve kalan hak oranına bakılarak erken haftalarda kart kullanımı daha temkinli önerilir; beklenen getiri yeterli değilse saklamak tercih edilir.

Oyuncular TFF’nin resmî puan kurallarına göre puanlanır: dakika, gol, asist, temiz sayfa, yenilen gol, kart, penaltı ve bonus kalemleri tahmine dâhildir. Kaleci kurtarışları gibi sayım verileri erken ağırlık kazanır; gol ve temiz sayfa gibi nadir sonuçlar daha temkinli tutulur.

Yeni transferlerde son lig sezonları ve varsa hazırlık maçlarındaki dakikalar dikkate alınır. Dış ligden gelen oyuncular için sabit bir katsayı yoktur; geçmişte Süper Lig’e gelen oyuncuların transfer öncesi ve sonrası üretimlerinden lig ve mevki bazında dönüşüm hesaplanır. Oyuncu ligde ilk maçını oynadıysa dış lig tahmini tamamen silinmez, açılış verisiyle harmanlanır.

Sakat, cezalı veya kadro dışı oyuncular optimizasyona alınmaz. Şüpheli durumdaki oyuncular oynama riskine göre daha düşük puanlanır.

## Kadro seçimi

Optimizasyon şu kurallara uyar: 100 milyon TL bütçe; 2 kaleci, 5 savunmacı, 5 orta saha, 3 forvet; kulüp başına en fazla 3 oyuncu; geçerli bir ilk 11 dizilişi.

PuLP ile kurulan tamsayı programlama modeli yasal kadroyu seçer. İlk 11 ve yedek değeri, TFF otomatik değişik kuralına göre hesaplanır: oynamayan bir ilk-11 oyuncusunun yerine, diziliş en az 1 kaleci, 3 defans ve 1 forvet kalacak şekilde yedekten sırayla oyuncu girer. Kaleci yalnız kaleciyle değişir. Sonuçta en yüksek beklenen puana sahip yasal kadro, kaptan ve uygun diziliş birlikte seçilir.

## Haftalık rapor

`data/weekly_report.png` dosyasında ilk 11, yedekler (otomatik giriş sırasıyla), rakipler, kaptan ve haftanın menajer kartı önerisi yer alır.

## Lig dönüşüm verisini yenileme

Geçmiş transfer verilerinden lig dönüşüm katsayılarını yeniden üretmek için:

```bat
python calibrate_leagues.py
```

Çıktı `data/league_translation.json` dosyasına yazılır. Normal kullanımda bu dosya repoda hazır gelir; yalnızca modeli güncellemek istediğinizde çalıştırmanız gerekir.

## Testler

```bat
python -m unittest discover -s tests -v
```

Testler ağ bağlantısı gerektirmez.
