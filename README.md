# Kan Hücresi Tespit Projesi

Bu proje, OpenCV ve derin öğrenme kullanarak kan hücrelerini (RBC, WBC ve Platelet) tespit eden bir sistem geliştirmeyi amaçlamaktadır. Proje, görüntü işleme ve makine öğrenimi tekniklerini kullanarak kan hücrelerini otomatik olarak tanımlayabilir ve sınıflandırabilir.

## Proje Hakkında

Kan hücresi tespiti, tıbbi teşhis ve araştırma alanlarında önemli bir uygulamadır. Bu proje, periferik kan yaymalarından alınan mikroskop görüntülerinde kan hücrelerini tespit etmek için modern bilgisayarlı görü ve derin öğrenme tekniklerini kullanmaktadır.

Projede iki farklı veri seti kullanılmıştır:
1. Kaggle'dan indirilen "Blood Cell Detection Dataset"
2. GitHub'dan indirilen "BCCD Dataset"

Bu veri setleri birleştirilerek daha kapsamlı bir eğitim kümesi oluşturulmuştur.

## Özellikler

- Kan hücresi görüntülerini işleme ve hazırlama
- MobileNetV2 tabanlı transfer öğrenme modeli
- Üç farklı kan hücresi tipini (RBC, WBC, Platelet) sınıflandırma
- Kullanıcı dostu grafiksel arayüz
- Detaylı zaman takibi ve performans değerlendirmesi

## Proje Yapısı

```
blood_cell_recognition/
├── data/                      # Veri setleri ve işlenmiş veriler
│   ├── kaggle/                # Kaggle'dan indirilen veri seti
│   ├── github/                # GitHub'dan indirilen veri seti
│   └── processed/             # İşlenmiş veri seti (eğitim, test, doğrulama)
├── models/                    # Eğitilmiş modeller ve değerlendirme sonuçları
├── src/                       # Kaynak kodlar
│   ├── data_preparation.py    # Veri hazırlama betiği
│   ├── model_training.py      # Model eğitim betiği
│   ├── user_interface.py      # Kullanıcı arayüzü
│   ├── github_upload.py       # GitHub'a yükleme betiği
│   ├── requirements.txt       # Gerekli paketler
│   └── start.sh               # Başlatma betiği
├── time_report.md             # Zaman takip raporu
└── README.md                  # Bu dosya
```

## Kurulum

Projeyi çalıştırmak için aşağıdaki adımları izleyin:

1. Depoyu klonlayın:
```bash
git clone https://github.com/burakkurt07/blood-cell-detection.git
cd blood-cell-detection
```

2. Gerekli paketleri yükleyin:
```bash
pip install -r src/requirements.txt
```

3. Başlatma betiğini çalıştırın:
```bash
bash src/start.sh
```

## Veri Hazırlama

Veri hazırlama aşamasında, Kaggle ve GitHub'dan indirilen veri setleri birleştirilmiş ve işlenmiştir. Bu işlem şunları içerir:

1. Görüntülerin yüklenmesi ve etiketlerin çıkarılması
2. Hücre görüntülerinin kırpılması
3. Veri setinin eğitim (%70), test (%15) ve doğrulama (%15) olarak bölünmesi
4. Veri seti dağılımının görselleştirilmesi

Veri seti dağılımı:
- RBC (Kırmızı Kan Hücreleri): 6392 görüntü
- WBC (Beyaz Kan Hücreleri): 475 görüntü
- Platelets (Trombositler): 361 görüntü

## Model Eğitimi

Model eğitimi, transfer öğrenme yaklaşımı kullanılarak gerçekleştirilmiştir. MobileNetV2 mimarisi temel alınarak, kan hücresi sınıflandırması için özelleştirilmiştir.

Eğitim süreci:
1. MobileNetV2 temel modelinin yüklenmesi (ImageNet ağırlıkları ile)
2. Sınıflandırma katmanlarının eklenmesi
3. İlk eğitim aşaması (temel model dondurulmuş)
4. İnce ayar aşaması (temel modelin son katmanları eğitilebilir)
5. Model değerlendirmesi ve optimizasyonu

## Kullanıcı Arayüzü

Kullanıcı arayüzü, eğitilmiş modeli kullanarak kan hücresi tespiti yapmanızı sağlar. Arayüz şu özellikleri içerir:

1. Görüntü seçme ve yükleme
2. Otomatik hücre tipi tespiti
3. Tahmin olasılıklarını gösteren çubuk grafiği
4. Tespit sonuçlarının detaylı gösterimi

## Performans Değerlendirmesi

Model, test veri seti üzerinde değerlendirilmiş ve aşağıdaki metrikler elde edilmiştir:

- Doğruluk (Accuracy): %99.17
- Hassasiyet (Precision): %98.92
- Duyarlılık (Recall): %99.05
- F1 Skoru: %98.98

## Zaman Takibi

Projenin her aşamasında harcanan süreler detaylı olarak takip edilmiştir. Zaman takip raporu `time_report.md` dosyasında bulunabilir.

## Gelecek Çalışmalar

- Daha büyük ve çeşitli veri setleri ile model performansının iyileştirilmesi
- Daha hafif modeller kullanarak mobil cihazlarda çalışabilecek versiyonların geliştirilmesi
- Hücre sayımı ve morfoloji analizi gibi ek özelliklerin eklenmesi
- Web tabanlı bir arayüz geliştirilerek uzaktan erişim sağlanması

## Katkıda Bulunanlar

- Burak Kurt - Proje Geliştiricisi

## Lisans

Bu proje Apache 2.0 lisansı altında lisanslanmıştır. Detaylar için lütfen `LICENSE` dosyasına bakın.

## İletişim

Sorularınız veya geri bildirimleriniz için GitHub üzerinden iletişime geçebilirsiniz.

------------

# Blood Cell Detection Project

This project aims to develop a system that detects blood cells (RBC, WBC, and Platelets) using OpenCV and deep learning. The project can automatically identify and classify blood cells using image processing and machine learning techniques.

## About the Project

Blood cell detection is an important application in medical diagnosis and research. This project uses modern computer vision and deep learning techniques to detect blood cells in microscope images taken from peripheral blood smears.

Two different datasets were used in the project:
1. "Blood Cell Detection Dataset" downloaded from Kaggle
2. "BCCD Dataset" downloaded from GitHub

These datasets were combined to create a more comprehensive training set.

## Features

- Processing and preparing blood cell images
- MobileNetV2-based transfer learning model
- Classification of three different blood cell types (RBC, WBC, Platelet)
- User-friendly graphical interface
- Detailed time tracking and performance evaluation

## Project Structure

```
blood_cell_recognition/
├── data/                      # Datasets and processed data
│   ├── kaggle/                # Dataset downloaded from Kaggle
│   ├── github/                # Dataset downloaded from GitHub
│   └── processed/             # Processed dataset (training, test, validation)
├── models/                    # Trained models and evaluation results
├── src/                       # Source code
│   ├── data_preparation.py    # Data preparation script
│   ├── model_training.py      # Model training script
│   ├── user_interface.py      # User interface
│   ├── github_upload.py       # GitHub upload script
│   ├── requirements.txt       # Required packages
│   └── start.sh               # Startup script
├── time_report.md             # Time tracking report
└── README.md                  # This file
```

## Installation

Follow these steps to run the project:

1. Clone the repository:
```bash
git clone https://github.com/burakkurt07/blood-cell-detection.git
cd blood-cell-detection
```

2. Install the required packages:
```bash
pip install -r src/requirements.txt
```

3. Run the startup script:
```bash
bash src/start.sh
```

## Data Preparation

In the data preparation phase, datasets downloaded from Kaggle and GitHub were combined and processed. This process includes:

1. Loading images and extracting labels
2. Cropping cell images
3. Splitting the dataset into training (70%), test (15%), and validation (15%)
4. Visualizing the dataset distribution

Dataset distribution:
- RBC (Red Blood Cells): 6392 images
- WBC (White Blood Cells): 475 images
- Platelets: 361 images

## Model Training

Model training was performed using a transfer learning approach. The MobileNetV2 architecture was used as a base and customized for blood cell classification.

Training process:
1. Loading the MobileNetV2 base model (with ImageNet weights)
2. Adding classification layers
3. First training phase (base model frozen)
4. Fine-tuning phase (last layers of the base model trainable)
5. Model evaluation and optimization

## User Interface

The user interface allows you to detect blood cells using the trained model. The interface includes the following features:

1. Image selection and loading
2. Automatic cell type detection
3. Bar chart showing prediction probabilities
4. Detailed display of detection results

## Performance Evaluation

The model was evaluated on the test dataset and the following metrics were obtained:

- Accuracy: 99.17%
- Precision: 98.92%
- Recall: 99.05%
- F1 Score: 98.98%

## Time Tracking

The time spent on each phase of the project has been tracked in detail. The time tracking report can be found in the `time_report.md` file.

## Future Work

- Improving model performance with larger and more diverse datasets
- Developing versions that can run on mobile devices using lighter models
- Adding additional features such as cell counting and morphology analysis
- Developing a web-based interface to provide remote access

## Contributors

- Burak Kurt - Project Developer

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

You can contact us through GitHub for questions or feedback.

---

*This project demonstrates the effectiveness of using modern computer vision and deep learning techniques for blood cell detection. It can be used as an auxiliary tool in medical diagnosis and research.*


