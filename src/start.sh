#!/bin/bash

# Kan Hücresi Tespit Projesi Başlatma Betiği
# Bu betik, projenin tüm bileşenlerini sırayla çalıştırır.

# Proje dizini
PROJECT_DIR="/home/ubuntu/blood_cell_recognition"
SRC_DIR="$PROJECT_DIR/src"

echo "Kan Hücresi Tespit Projesi başlatılıyor..."
echo "----------------------------------------"

# Gerekli paketleri kontrol et ve kur
echo "Gerekli paketler kontrol ediliyor..."
pip install -r "$SRC_DIR/requirements.txt"

# Veri hazırlama
echo "Veri hazırlama başlatılıyor..."
python3 "$SRC_DIR/data_preparation.py"

# Model eğitimi
echo "Model eğitimi başlatılıyor..."
python3 "$SRC_DIR/model_training.py"

# Kullanıcı arayüzü
echo "Kullanıcı arayüzü başlatılıyor..."
python3 "$SRC_DIR/user_interface.py"

echo "----------------------------------------"
echo "İşlem tamamlandı!"
