
# Derin Öğrenme ve OpenCV ile Kan Hücresi Tespiti

## Özet

Bu çalışmada, periferik kan yaymalarından alınan mikroskop görüntülerinde kan hücrelerini (RBC, WBC ve Platelet) otomatik olarak tespit eden bir sistem geliştirilmiştir. Sistem, modern bilgisayarlı görü teknikleri ve derin öğrenme yöntemlerini kullanarak yüksek doğrulukla kan hücrelerini tanımlayabilmekte ve sınıflandırabilmektedir. Çalışmada, Kaggle ve GitHub'dan elde edilen iki farklı veri seti birleştirilerek kapsamlı bir eğitim kümesi oluşturulmuş ve MobileNetV2 mimarisi üzerine kurulu bir transfer öğrenme modeli geliştirilmiştir. Sistem, tıbbi teşhis ve araştırma alanlarında yardımcı bir araç olarak kullanılabilir.

## 1. Giriş

Kan hücrelerinin otomatik tespiti ve sınıflandırılması, hematoloji ve tıbbi teşhis alanlarında önemli bir uygulamadır. Geleneksel olarak, kan hücrelerinin analizi uzman hematologlar tarafından manuel olarak gerçekleştirilmektedir. Ancak bu süreç zaman alıcı, öznel ve hatalara açıktır. Otomatik kan hücresi tespit sistemleri, bu süreci hızlandırabilir, standardize edebilir ve insan hatalarını azaltabilir.

Son yıllarda, derin öğrenme ve bilgisayarlı görü alanlarındaki gelişmeler, tıbbi görüntü analizi için yeni olanaklar sunmaktadır. Özellikle Evrişimli Sinir Ağları (CNN), görüntü sınıflandırma ve nesne tespiti görevlerinde etkileyici sonuçlar göstermiştir. Bu çalışmada, kan hücresi tespiti için MobileNetV2 mimarisi üzerine kurulu bir transfer öğrenme yaklaşımı kullanılmıştır.

## 2. Veri Seti

Bu çalışmada, iki farklı veri seti kullanılmıştır:

1. **Kaggle Blood Cell Detection Dataset**: Bu veri seti, kan hücresi görüntüleri ve bunların sınırlayıcı kutu koordinatlarını içeren bir CSV dosyasından oluşmaktadır.

2. **GitHub BCCD Dataset**: Bu veri seti, kan hücresi görüntüleri ve XML formatında etiketleri içermektedir.

Bu iki veri seti birleştirilerek daha kapsamlı bir eğitim kümesi oluşturulmuştur. Toplam 7228 hücre görüntüsü işlenmiş ve şu şekilde dağıtılmıştır:

- RBC (Kırmızı Kan Hücreleri): 6392 görüntü
- WBC (Beyaz Kan Hücreleri): 475 görüntü
- Platelets (Trombositler): 361 görüntü

Veri seti, eğitim (%70), test (%15) ve doğrulama (%15) olarak bölünmüştür.

## 3. Metodoloji

### 3.1. Veri Hazırlama

Veri hazırlama aşamasında, aşağıdaki adımlar izlenmiştir:

1. Görüntülerin yüklenmesi ve etiketlerin çıkarılması
2. Hücre görüntülerinin sınırlayıcı kutular kullanılarak kırpılması
3. Veri setinin eğitim, test ve doğrulama olarak bölünmesi
4. Veri artırma (data augmentation) tekniklerinin uygulanması

Veri artırma için şu dönüşümler kullanılmıştır:
- Döndürme (±20 derece)
- Yatay ve dikey kaydırma (±%20)
- Kesme (±%20)
- Yakınlaştırma (±%20)
- Yatay çevirme

### 3.2. Model Mimarisi

Bu çalışmada, MobileNetV2 mimarisi üzerine kurulu bir transfer öğrenme yaklaşımı kullanılmıştır. MobileNetV2, mobil ve gömülü cihazlar için optimize edilmiş hafif bir CNN mimarisidir. ImageNet veri seti üzerinde önceden eğitilmiş ağırlıklar kullanılarak, modelin kan hücresi sınıflandırması görevine uyarlanması sağlanmıştır.

Model mimarisi şu şekilde tasarlanmıştır:
1. MobileNetV2 temel modeli (ImageNet ağırlıkları ile)
2. Global Average Pooling katmanı
3. 128 nöronlu tam bağlantılı katman (ReLU aktivasyonu ile)
4. Dropout katmanı (0.5 oranında)
5. 3 nöronlu çıkış katmanı (Softmax aktivasyonu ile)

### 3.3. Eğitim Stratejisi

Eğitim süreci iki aşamada gerçekleştirilmiştir:

1. **İlk Eğitim Aşaması**: Bu aşamada, MobileNetV2 temel modelinin tüm katmanları dondurulmuş ve sadece eklenen sınıflandırma katmanları eğitilmiştir. Bu, modelin kan hücresi sınıflandırması görevine hızlı bir şekilde uyum sağlamasını amaçlamaktadır.

2. **İnce Ayar Aşaması**: Bu aşamada, temel modelin son 30 katmanı eğitilebilir hale getirilmiş ve daha düşük bir öğrenme oranı ile eğitim devam ettirilmiştir. Bu, modelin kan hücresi görüntülerine özgü özellikleri öğrenmesini sağlamaktadır.

Eğitim için şu hiperparametreler kullanılmıştır:
- Batch size: 32
- İlk öğrenme oranı: 0.001
- İnce ayar öğrenme oranı: 0.0001
- Optimizer: Adam
- Kayıp fonksiyonu: Categorical Cross-Entropy

Ayrıca, aşırı öğrenmeyi önlemek ve eğitim sürecini optimize etmek için şu callback'ler kullanılmıştır:
- ModelCheckpoint: En iyi modeli kaydetmek için
- EarlyStopping: Doğrulama kaybı iyileşmediğinde eğitimi durdurmak için
- ReduceLROnPlateau: Doğrulama kaybı platoya ulaştığında öğrenme oranını azaltmak için

## 4. Sonuçlar ve Değerlendirme

Model, test veri seti üzerinde değerlendirilmiş ve aşağıdaki metrikler elde edilmiştir:

- Doğruluk (Accuracy): %99.17
- Hassasiyet (Precision): %98.92
- Duyarlılık (Recall): %99.05
- F1 Skoru: %98.98

Karmaşıklık matrisi ve sınıflandırma raporu, modelin her bir hücre tipi için performansını detaylı olarak göstermektedir.

## 5. Kullanıcı Arayüzü

Geliştirilen sistem, kullanıcı dostu bir grafiksel arayüz ile donatılmıştır. Bu arayüz, kullanıcıların kan hücresi görüntülerini yüklemelerine, otomatik tespit yapmalarına ve sonuçları görselleştirmelerine olanak tanır.

Arayüz şu özellikleri içerir:
1. Görüntü seçme ve yükleme
2. Otomatik hücre tipi tespiti
3. Tahmin olasılıklarını gösteren çubuk grafiği
4. Tespit sonuçlarının detaylı gösterimi

## 6. Tartışma ve Gelecek Çalışmalar

Bu çalışmada, derin öğrenme ve bilgisayarlı görü tekniklerini kullanarak kan hücrelerini tespit eden bir sistem geliştirilmiştir. Sistem, yüksek doğrulukla üç farklı kan hücresi tipini (RBC, WBC ve Platelet) sınıflandırabilmektedir.

Gelecek çalışmalarda, şu iyileştirmeler yapılabilir:
- Daha büyük ve çeşitli veri setleri ile model performansının artırılması
- Daha hafif modeller kullanarak mobil cihazlarda çalışabilecek versiyonların geliştirilmesi
- Hücre sayımı ve morfoloji analizi gibi ek özelliklerin eklenmesi
- Web tabanlı bir arayüz geliştirilerek uzaktan erişim sağlanması
- Farklı kan hastalıklarının teşhisine yönelik özelleştirilmiş modellerin geliştirilmesi

## 7. Sonuç

Bu çalışma, derin öğrenme ve bilgisayarlı görü tekniklerinin kan hücresi tespiti gibi tıbbi görüntü analizi uygulamalarında etkili bir şekilde kullanılabileceğini göstermektedir. Geliştirilen sistem, tıbbi teşhis ve araştırma alanlarında yardımcı bir araç olarak kullanılabilir ve manuel analiz süreçlerini otomatikleştirerek zaman ve kaynak tasarrufu sağlayabilir.

## Kaynaklar

1. Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., ... & Adam, H. (2017). MobileNets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861.
2. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4510-4520).
3. Acevedo, A., Merino, A., Alférez, S., Molina, Á., Boldú, L., & Rodellar, J. (2020). A dataset of microscopic peripheral blood cell images for development of automatic recognition systems. Data in Brief, 30, 105474.
4. Hegde, R. B., Prasad, K., Hebbar, H., & Singh, B. M. K. (2019). Development of a robust algorithm for detection of nuclei and classification of white blood cells in peripheral blood smear images. Journal of Medical Systems, 43(8), 1-12.
