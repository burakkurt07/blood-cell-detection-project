#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kan Hücresi Tespit Projesi - Model Eğitim Modülü
Bu modül, hazırlanan veri seti üzerinde kan hücresi tespit modelini eğitir.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from tqdm import tqdm

# Zaman takibi için başlangıç zamanını kaydet
start_time = time.time()

# Proje dizinleri
PROJECT_DIR = '/home/ubuntu/blood_cell_recognition'
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, 'train')
TEST_DIR = os.path.join(PROCESSED_DATA_DIR, 'test')
VALIDATION_DIR = os.path.join(PROCESSED_DATA_DIR, 'validation')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')

# Model dizinini oluştur
os.makedirs(MODELS_DIR, exist_ok=True)

# Görüntü boyutu ve batch size
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

# Sınıf sayısı
NUM_CLASSES = 3
CLASS_NAMES = ['Platelets', 'RBC', 'WBC']

def create_data_generators():
    """Eğitim, doğrulama ve test veri üreteçlerini oluşturur."""
    print("Veri üreteçleri oluşturuluyor...")
    
    # Veri artırma (data augmentation) için ImageDataGenerator
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Doğrulama ve test için sadece ölçeklendirme yapılır
    valid_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Eğitim veri üreteci
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    # Doğrulama veri üreteci
    validation_generator = valid_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Test veri üreteci
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator

def build_model():
    """MobileNetV2 tabanlı transfer öğrenme modeli oluşturur."""
    print("Model oluşturuluyor...")
    
    # MobileNetV2 temel modelini yükle (ImageNet ağırlıkları ile)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Temel modelin katmanlarını dondur
    for layer in base_model.layers:
        layer.trainable = False
    
    # Sınıflandırma katmanlarını ekle
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Modeli oluştur
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Modeli derle
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def train_model(model, train_generator, validation_generator):
    """Modeli eğitir ve eğitim geçmişini döndürür."""
    print("Model eğitimi başlatılıyor...")
    
    # Eğitim sırasında en iyi modeli kaydetmek için callback
    checkpoint = ModelCheckpoint(
        os.path.join(MODELS_DIR, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Erken durdurma callback'i
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Öğrenme oranını azaltma callback'i
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
    
    # Modeli eğit
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    return history

def fine_tune_model(model, base_model, train_generator, validation_generator):
    """Modelin son katmanlarını ince ayarlar."""
    print("Model ince ayarı başlatılıyor...")
    
    # Temel modelin son 30 katmanını eğitilebilir yap
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    
    # Modeli daha düşük öğrenme oranı ile yeniden derle
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # İnce ayar için checkpoint
    fine_tune_checkpoint = ModelCheckpoint(
        os.path.join(MODELS_DIR, 'fine_tuned_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # İnce ayar için erken durdurma
    fine_tune_early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    )
    
    # İnce ayar için öğrenme oranını azaltma
    fine_tune_reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.000001,
        verbose=1
    )
    
    # İnce ayar eğitimi
    fine_tune_history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=[fine_tune_checkpoint, fine_tune_early_stopping, fine_tune_reduce_lr]
    )
    
    return fine_tune_history

def evaluate_model(model, test_generator):
    """Modeli test veri seti üzerinde değerlendirir."""
    print("Model değerlendiriliyor...")
    
    # Test veri seti üzerinde değerlendirme
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test doğruluğu: {test_accuracy:.4f}")
    print(f"Test kaybı: {test_loss:.4f}")
    
    # Tahminleri al
    test_generator.reset()
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Gerçek sınıfları al
    y_true = test_generator.classes
    
    # Sınıf etiketlerini al
    class_labels = list(test_generator.class_indices.keys())
    
    # Sınıflandırma raporu
    report = classification_report(y_true, y_pred_classes, target_names=class_labels)
    print("Sınıflandırma Raporu:")
    print(report)
    
    # Karmaşıklık matrisi
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Sonuçları görselleştir
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Karmaşıklık Matrisi')
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)
    
    # Matris değerlerini ekle
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    
    # Grafiği kaydet
    plt.savefig(os.path.join(MODELS_DIR, 'confusion_matrix.png'))
    
    # Değerlendirme sonuçlarını dosyaya kaydet
    with open(os.path.join(MODELS_DIR, 'evaluation_results.txt'), 'w') as f:
        f.write(f"Test Doğruluğu: {test_accuracy:.4f}\n")
        f.write(f"Test Kaybı: {test_loss:.4f}\n\n")
        f.write("Sınıflandırma Raporu:\n")
        f.write(report)
    
    return test_accuracy, report, cm

def plot_training_history(history, fine_tune_history=None):
    """Eğitim geçmişini görselleştirir."""
    print("Eğitim geçmişi görselleştiriliyor...")
    
    # Eğitim ve doğrulama doğruluğu
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    
    if fine_tune_history:
        # İlk eğitim sonrası epoch sayısı
        initial_epochs = len(history.history['accuracy'])
        
        # x ekseni değerlerini oluştur
        epochs_range = range(1, initial_epochs + 1)
        fine_tune_epochs_range = range(initial_epochs + 1, initial_epochs + len(fine_tune_history.history['accuracy']) + 1)
        
        plt.plot(epochs_range, history.history['accuracy'], label='Eğitim Doğruluğu (İlk Aşama)')
        plt.plot(epochs_range, history.history['val_accuracy'], label='Doğrulama Doğruluğu (İlk Aşama)')
        plt.plot(fine_tune_epochs_range, fine_tune_history.history['accuracy'], label='Eğitim Doğruluğu (İnce Ayar)')
        plt.plot(fine_tune_epochs_range, fine_tune_history.history['val_accuracy'], label='Doğrulama Doğruluğu (İnce Ayar)')
    else:
        plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
        plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
    
    plt.title('Model Doğruluğu')
    plt.ylabel('Doğruluk')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    
    # Eğitim ve doğrulama kaybı
    plt.subplot(1, 2, 2)
    
    if fine_tune_history:
        plt.plot(epochs_range, history.history['loss'], label='Eğitim Kaybı (İlk Aşama)')
        plt.plot(epochs_range, history.history['val_loss'], label='Doğrulama Kaybı (İlk Aşama)')
        plt.plot(fine_tune_epochs_range, fine_tune_history.history['loss'], label='Eğitim Kaybı (İnce Ayar)')
        plt.plot(fine_tune_epochs_range, fine_tune_history.history['val_loss'], label='Doğrulama Kaybı (İnce Ayar)')
    else:
        plt.plot(history.history['loss'], label='Eğitim Kaybı')
        plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    
    plt.title('Model Kaybı')
    plt.ylabel('Kayıp')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'training_history.png'))

def optimize_model(model):
    """Modeli optimize eder ve kaydeder."""
    print("Model optimize ediliyor...")
    
    # Modeli TensorFlow Lite formatına dönüştür
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # TFLite modelini kaydet
    with open(os.path.join(MODELS_DIR, 'model.tflite'), 'wb') as f:
        f.write(tflite_model)
    
    # Optimize edilmiş modeli kaydet
    model.save(os.path.join(MODELS_DIR, 'optimized_mobilenet.h5'))
    
    print("Model optimizasyonu tamamlandı ve kaydedildi.")

def main():
    """Ana işlev: Modeli eğitir, değerlendirir ve optimize eder."""
    print("Kan hücresi tespit modeli eğitimi başlatılıyor...")
    
    # Veri üreteçlerini oluştur
    train_generator, validation_generator, test_generator = create_data_generators()
    
    # Modeli oluştur
    model, base_model = build_model()
    
    # Modeli eğit
    history = train_model(model, train_generator, validation_generator)
    
    # Modeli ince ayarla
    fine_tune_history = fine_tune_model(model, base_model, train_generator, validation_generator)
    
    # Modeli değerlendir
    test_accuracy, report, cm = evaluate_model(model, test_generator)
    
    # Eğitim geçmişini görselleştir
    plot_training_history(history, fine_tune_history)
    
    # Modeli optimize et
    optimize_model(model)
    
    # İşlem süresini hesapla
    end_time = time.time()
    training_time = end_time - start_time
    
    # Sonuçları yazdır
    print(f"\nModel eğitimi tamamlandı!")
    print(f"Toplam eğitim süresi: {training_time:.2f} saniye")
    print(f"Test doğruluğu: {test_accuracy:.4f}")
    
    # Zaman bilgisini dosyaya kaydet
    with open(os.path.join(PROJECT_DIR, 'time_tracking.md'), 'a') as f:
        f.write(f"- Model eğitimi: Başlangıç - {time.strftime('%d Nisan %Y %H:%M:%S')}, Bitiş - {time.strftime('%d Nisan %Y %H:%M:%S', time.localtime(end_time))}, Süre - {training_time:.2f} saniye\n")

if __name__ == "__main__":
    main()
