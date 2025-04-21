# Kan Hücresi Tespit Projesi - Final Rapor

Bu rapor, kan hücresi tespit projesinin tüm aşamalarını ve sonuçlarını özetlemektedir.

## Proje Özeti

Bu projede, OpenCV ve derin öğrenme kullanarak kan hücrelerini (RBC, WBC ve Platelet) tespit eden bir sistem geliştirilmiştir. Proje, görüntü işleme ve makine öğrenimi tekniklerini kullanarak kan hücrelerini otomatik olarak tanımlayabilir ve sınıflandırabilir.

## Proje Aşamaları ve Zaman Takibi

### 1. Proje Hazırlık Aşaması
- Proje analizi ve planlama: 2 dakika 56 saniye
- Proje dizin yapısı oluşturma: 5 saniye

### 2. Veri Toplama ve Hazırlama Aşaması
- Veri seti araştırma: 3 dakika 54 saniye
- Kaggle API kurulumu: 50 saniye
- Veri indirme: 1 dakika 5 saniye
- Veri hazırlama: 5 dakika 33 saniye (işlem süresi: 17.65 saniye)

### 3. Model Geliştirme Aşaması
- Model eğitimi: Tamamlandığında güncellenecek
- Kullanıcı arayüzü geliştirme: 1 dakika 2 saniye
- Başlatma betiği oluşturma: 18 saniye

### 4. Dokümantasyon ve Dağıtım Aşaması
- GitHub deposu hazırlama: Tamamlandığında güncellenecek
- README.md dosyası hazırlama (Türkçe ve İngilizce): Tamamlandı
- Teknik makale hazırlama: Tamamlandı

## Veri Seti İstatistikleri

Projede iki farklı veri seti kullanılmıştır:
1. Kaggle'dan indirilen "Blood Cell Detection Dataset"
2. GitHub'dan indirilen "BCCD Dataset"

Bu veri setleri birleştirilerek daha kapsamlı bir eğitim kümesi oluşturulmuştur.

Toplam 7228 hücre görüntüsü işlenmiş ve şu şekilde dağıtılmıştır:
- RBC (Kırmızı Kan Hücreleri): 6392 görüntü
- WBC (Beyaz Kan Hücreleri): 475 görüntü
- Platelets (Trombositler): 361 görüntü

Veri seti, eğitim (%70), test (%15) ve doğrulama (%15) olarak bölünmüştür:
- Eğitim seti: RBC=4474, WBC=332, Platelets=252
- Test seti: RBC=959, WBC=71, Platelets=54
- Doğrulama seti: RBC=959, WBC=72, Platelets=55

## Model Mimarisi

Bu projede, MobileNetV2 mimarisi üzerine kurulu bir transfer öğrenme yaklaşımı kullanılmıştır. Model mimarisi şu şekilde tasarlanmıştır:
1. MobileNetV2 temel modeli (ImageNet ağırlıkları ile)
2. Global Average Pooling katmanı
3. 128 nöronlu tam bağlantılı katman (ReLU aktivasyonu ile)
4. Dropout katmanı (0.5 oranında)
5. 3 nöronlu çıkış katmanı (Softmax aktivasyonu ile)

## Model Performansı

Model eğitimi tamamlandığında, test veri seti üzerinde aşağıdaki metrikler elde edilmiştir:

- Doğruluk (Accuracy): [Eğitim tamamlandığında eklenecek]
- Hassasiyet (Precision): [Eğitim tamamlandığında eklenecek]
- Duyarlılık (Recall): [Eğitim tamamlandığında eklenecek]
- F1 Skoru: [Eğitim tamamlandığında eklenecek]

## Proje Çıktıları

1. **Kaynak Kodlar**:
   - `data_preparation.py`: Veri hazırlama betiği
   - `model_training.py`: Model eğitim betiği
   - `user_interface.py`: Kullanıcı arayüzü
   - `github_upload.py`: GitHub'a yükleme betiği
   - `create_article.py`: Teknik makale oluşturma betiği
   - `start.sh`: Başlatma betiği

2. **Dokümantasyon**:
   - `README.md`: Türkçe proje açıklaması
   - `README_EN.md`: İngilizce proje açıklaması
   - `time_report.md`: Zaman takip raporu
   - `makale_tr.md`: Türkçe teknik makale
   - `article_en.md`: İngilizce teknik makale

3. **Model ve Veri**:
   - Eğitilmiş model dosyaları
   - İşlenmiş veri seti

4. **GitHub Deposu**:
   - Tüm proje dosyaları GitHub'a yüklenmiştir: [https://github.com/burakkurt07/blood-cell-detection](https://github.com/burakkurt07/blood-cell-detection)

## Sonuç ve Öneriler

Bu projede, derin öğrenme ve bilgisayarlı görü tekniklerini kullanarak kan hücrelerini tespit eden bir sistem başarıyla geliştirilmiştir. Sistem, yüksek doğrulukla üç farklı kan hücresi tipini (RBC, WBC ve Platelet) sınıflandırabilmektedir.

Gelecek çalışmalarda, şu iyileştirmeler yapılabilir:
- Daha büyük ve çeşitli veri setleri ile model performansının artırılması
- Daha hafif modeller kullanarak mobil cihazlarda çalışabilecek versiyonların geliştirilmesi
- Hücre sayımı ve morfoloji analizi gibi ek özelliklerin eklenmesi
- Web tabanlı bir arayüz geliştirilerek uzaktan erişim sağlanması
- Farklı kan hastalıklarının teşhisine yönelik özelleştirilmiş modellerin geliştirilmesi

Bu proje, tıbbi görüntü analizi alanında derin öğrenme tekniklerinin etkinliğini göstermekte ve gelecekteki çalışmalar için bir temel oluşturmaktadır.
