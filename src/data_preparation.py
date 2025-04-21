#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kan Hücresi Tespit Projesi - Veri Hazırlama Modülü
Bu modül, Kaggle ve GitHub'dan indirilen veri setlerini birleştirip hazırlar.
"""

import os
import shutil
import pandas as pd
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# Zaman takibi için başlangıç zamanını kaydet
start_time = time.time()

# Proje dizinleri
PROJECT_DIR = '/home/ubuntu/blood_cell_recognition'
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
KAGGLE_DATA_DIR = os.path.join(DATA_DIR, 'kaggle')
GITHUB_DATA_DIR = os.path.join(DATA_DIR, 'github/BCCD_Dataset')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, 'train')
TEST_DIR = os.path.join(PROCESSED_DATA_DIR, 'test')
VALIDATION_DIR = os.path.join(PROCESSED_DATA_DIR, 'validation')

# İşlenmiş veri dizinlerini oluştur
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)

# Kan hücresi sınıfları için dizinler oluştur
for dir_path in [TRAIN_DIR, TEST_DIR, VALIDATION_DIR]:
    os.makedirs(os.path.join(dir_path, 'RBC'), exist_ok=True)
    os.makedirs(os.path.join(dir_path, 'WBC'), exist_ok=True)
    os.makedirs(os.path.join(dir_path, 'Platelets'), exist_ok=True)

def process_kaggle_dataset():
    """Kaggle veri setini işler ve etiketleri çıkarır."""
    print("Kaggle veri seti işleniyor...")
    
    # Kaggle veri setindeki annotations.csv dosyasını oku
    annotations_path = os.path.join(KAGGLE_DATA_DIR, 'annotations.csv')
    if not os.path.exists(annotations_path):
        print(f"Hata: {annotations_path} bulunamadı!")
        return []
    
    annotations_df = pd.read_csv(annotations_path)
    
    # Görüntü dosyalarını ve etiketleri işle
    images_dir = os.path.join(KAGGLE_DATA_DIR, 'images')
    if not os.path.exists(images_dir):
        print(f"Hata: {images_dir} bulunamadı!")
        return []
    
    processed_data = []
    
    for _, row in tqdm(annotations_df.iterrows(), total=len(annotations_df), desc="Kaggle görüntüleri işleniyor"):
        image_id = row['image']
        image_path = os.path.join(images_dir, f"{image_id}")
        
        if not os.path.exists(image_path):
            # Dosya adı formatını kontrol et
            if not os.path.exists(os.path.join(images_dir, image_id)):
                print(f"Uyarı: {image_path} bulunamadı, atlanıyor.")
                continue
            else:
                image_path = os.path.join(images_dir, image_id)
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Uyarı: {image_path} okunamadı, atlanıyor.")
                continue
                
            cell_type = row['label'].upper()  # RBC, WBC veya Platelets olarak standartlaştır
            x_min = int(float(row['xmin']))
            y_min = int(float(row['ymin']))
            x_max = int(float(row['xmax']))
            y_max = int(float(row['ymax']))
            
            # Görüntüyü kırp
            cell_image = image[y_min:y_max, x_min:x_max]
            
            # Veri yapısına ekle
            processed_data.append({
                'image': cell_image,
                'cell_type': cell_type,
                'source': 'kaggle',
                'original_path': image_path
            })
            
        except Exception as e:
            print(f"Hata: {image_path} işlenirken bir sorun oluştu: {e}")
    
    print(f"Kaggle veri setinden {len(processed_data)} hücre görüntüsü işlendi.")
    return processed_data

def process_github_dataset():
    """GitHub'dan indirilen BCCD veri setini işler."""
    print("GitHub BCCD veri seti işleniyor...")
    
    bccd_images_dir = os.path.join(GITHUB_DATA_DIR, 'BCCD', 'JPEGImages')
    bccd_annotations_dir = os.path.join(GITHUB_DATA_DIR, 'BCCD', 'Annotations')
    
    if not os.path.exists(bccd_images_dir) or not os.path.exists(bccd_annotations_dir):
        print(f"Hata: BCCD veri seti dizinleri bulunamadı!")
        return []
    
    processed_data = []
    
    # XML dosyalarını işle
    for xml_file in tqdm(os.listdir(bccd_annotations_dir), desc="GitHub BCCD görüntüleri işleniyor"):
        if not xml_file.endswith('.xml'):
            continue
            
        xml_path = os.path.join(bccd_annotations_dir, xml_file)
        image_file = xml_file.replace('.xml', '.jpg')
        image_path = os.path.join(bccd_images_dir, image_file)
        
        if not os.path.exists(image_path):
            print(f"Uyarı: {image_path} bulunamadı, atlanıyor.")
            continue
            
        try:
            # XML dosyasını parse et
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Görüntüyü oku
            image = cv2.imread(image_path)
            if image is None:
                print(f"Uyarı: {image_path} okunamadı, atlanıyor.")
                continue
                
            # Her bir nesneyi işle
            for obj in root.findall('object'):
                cell_type = obj.find('name').text
                
                # Sınıf adını standartlaştır
                if cell_type.lower() == 'rbc':
                    cell_type = 'RBC'
                elif cell_type.lower() in ['wbc', 'white blood cell']:
                    cell_type = 'WBC'
                elif cell_type.lower() in ['platelets', 'platelet']:
                    cell_type = 'Platelets'
                else:
                    print(f"Uyarı: Bilinmeyen hücre tipi {cell_type}, atlanıyor.")
                    continue
                
                # Sınırlayıcı kutuyu al
                bbox = obj.find('bndbox')
                x_min = int(float(bbox.find('xmin').text))
                y_min = int(float(bbox.find('ymin').text))
                x_max = int(float(bbox.find('xmax').text))
                y_max = int(float(bbox.find('ymax').text))
                
                # Görüntüyü kırp
                cell_image = image[y_min:y_max, x_min:x_max]
                
                # Veri yapısına ekle
                processed_data.append({
                    'image': cell_image,
                    'cell_type': cell_type,
                    'source': 'github',
                    'original_path': image_path
                })
                
        except Exception as e:
            print(f"Hata: {xml_path} işlenirken bir sorun oluştu: {e}")
    
    print(f"GitHub veri setinden {len(processed_data)} hücre görüntüsü işlendi.")
    return processed_data

def split_and_save_data(processed_data):
    """İşlenmiş veriyi eğitim, test ve doğrulama setlerine böler ve kaydeder."""
    print("Veri seti bölünüyor ve kaydediliyor...")
    
    # Veriyi karıştır
    np.random.shuffle(processed_data)
    
    # Hücre tipine göre veriyi grupla
    rbc_data = [item for item in processed_data if item['cell_type'] == 'RBC']
    wbc_data = [item for item in processed_data if item['cell_type'] == 'WBC']
    platelets_data = [item for item in processed_data if item['cell_type'] == 'Platelets']
    
    print(f"Toplam RBC: {len(rbc_data)}, WBC: {len(wbc_data)}, Platelets: {len(platelets_data)}")
    
    # Her sınıf için veriyi böl (70% eğitim, 15% test, 15% doğrulama)
    def split_data(data):
        n = len(data)
        train_idx = int(0.7 * n)
        test_idx = int(0.85 * n)
        
        return data[:train_idx], data[train_idx:test_idx], data[test_idx:]
    
    rbc_train, rbc_test, rbc_val = split_data(rbc_data)
    wbc_train, wbc_test, wbc_val = split_data(wbc_data)
    platelets_train, platelets_test, platelets_val = split_data(platelets_data)
    
    # Görüntüleri kaydet
    def save_images(data_list, target_dir, cell_type):
        for i, item in enumerate(data_list):
            # Boş görüntüleri kontrol et
            if item['image'] is None or item['image'].size == 0:
                print(f"Uyarı: {i}. {cell_type} görüntüsü boş, atlanıyor.")
                continue
                
            # Görüntü boyutunu kontrol et
            if item['image'].shape[0] <= 0 or item['image'].shape[1] <= 0:
                print(f"Uyarı: {i}. {cell_type} görüntüsü geçersiz boyutta, atlanıyor.")
                continue
                
            filename = f"{cell_type}_{i:04d}.png"
            save_path = os.path.join(target_dir, cell_type, filename)
            try:
                cv2.imwrite(save_path, item['image'])
            except Exception as e:
                print(f"Hata: {i}. {cell_type} görüntüsü kaydedilemedi: {e}")
    
    # RBC görüntülerini kaydet
    save_images(rbc_train, TRAIN_DIR, 'RBC')
    save_images(rbc_test, TEST_DIR, 'RBC')
    save_images(rbc_val, VALIDATION_DIR, 'RBC')
    
    # WBC görüntülerini kaydet
    save_images(wbc_train, TRAIN_DIR, 'WBC')
    save_images(wbc_test, TEST_DIR, 'WBC')
    save_images(wbc_val, VALIDATION_DIR, 'WBC')
    
    # Platelets görüntülerini kaydet
    save_images(platelets_train, TRAIN_DIR, 'Platelets')
    save_images(platelets_test, TEST_DIR, 'Platelets')
    save_images(platelets_val, VALIDATION_DIR, 'Platelets')
    
    # Özet bilgileri yazdır
    print(f"Eğitim seti: RBC={len(rbc_train)}, WBC={len(wbc_train)}, Platelets={len(platelets_train)}")
    print(f"Test seti: RBC={len(rbc_test)}, WBC={len(wbc_test)}, Platelets={len(platelets_test)}")
    print(f"Doğrulama seti: RBC={len(rbc_val)}, WBC={len(wbc_val)}, Platelets={len(platelets_val)}")
    
    # Veri seti dağılımını görselleştir
    class_names = ['RBC', 'WBC', 'Platelets']
    train_counts = [len(rbc_train), len(wbc_train), len(platelets_train)]
    test_counts = [len(rbc_test), len(wbc_test), len(platelets_test)]
    val_counts = [len(rbc_val), len(wbc_val), len(platelets_val)]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(class_names))
    width = 0.25
    
    plt.bar(x - width, train_counts, width, label='Eğitim')
    plt.bar(x, test_counts, width, label='Test')
    plt.bar(x + width, val_counts, width, label='Doğrulama')
    
    plt.xlabel('Hücre Tipi')
    plt.ylabel('Görüntü Sayısı')
    plt.title('Veri Seti Dağılımı')
    plt.xticks(x, class_names)
    plt.legend()
    
    # Grafiği kaydet
    plt.savefig(os.path.join(PROCESSED_DATA_DIR, 'data_distribution.png'))
    
    # Veri seti bilgilerini bir CSV dosyasına kaydet
    dataset_info = {
        'Class': class_names * 3,
        'Split': ['Train'] * 3 + ['Test'] * 3 + ['Validation'] * 3,
        'Count': train_counts + test_counts + val_counts
    }
    
    dataset_df = pd.DataFrame(dataset_info)
    dataset_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'dataset_info.csv'), index=False)
    
    return {
        'train': {
            'RBC': len(rbc_train),
            'WBC': len(wbc_train),
            'Platelets': len(platelets_train),
            'total': len(rbc_train) + len(wbc_train) + len(platelets_train)
        },
        'test': {
            'RBC': len(rbc_test),
            'WBC': len(wbc_test),
            'Platelets': len(platelets_test),
            'total': len(rbc_test) + len(wbc_test) + len(platelets_test)
        },
        'validation': {
            'RBC': len(rbc_val),
            'WBC': len(wbc_val),
            'Platelets': len(platelets_val),
            'total': len(rbc_val) + len(wbc_val) + len(platelets_val)
        }
    }

def main():
    """Ana işlev: Veri setlerini işler ve hazırlar."""
    print("Kan hücresi tespit projesi için veri hazırlama başlatılıyor...")
    
    # Kaggle veri setini işle
    kaggle_data = process_kaggle_dataset()
    
    # GitHub veri setini işle
    github_data = process_github_dataset()
    
    # Tüm veriyi birleştir
    all_data = kaggle_data + github_data
    print(f"Toplam {len(all_data)} hücre görüntüsü işlendi.")
    
    if len(all_data) == 0:
        print("Hata: İşlenecek veri bulunamadı!")
        return
    
    # Veriyi böl ve kaydet
    dataset_stats = split_and_save_data(all_data)
    
    # İşlem süresini hesapla
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Sonuçları yazdır
    print(f"\nVeri hazırlama tamamlandı!")
    print(f"Toplam işlem süresi: {processing_time:.2f} saniye")
    print(f"Toplam görüntü sayısı: {sum(dataset_stats[split]['total'] for split in dataset_stats)}")
    
    # Zaman bilgisini dosyaya kaydet
    with open(os.path.join(PROJECT_DIR, 'time_tracking.md'), 'a') as f:
        f.write(f"- Veri hazırlama: Bitiş - {time.strftime('%d Nisan %Y %H:%M:%S')}, Süre - {processing_time:.2f} saniye\n")

if __name__ == "__main__":
    main()
