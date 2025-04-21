#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kan Hücresi Tespit Projesi - Kullanıcı Arayüzü
Bu modül, eğitilmiş modeli kullanarak kan hücresi tespiti yapan bir arayüz sunar.
"""

import os
import sys
import time
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Proje dizinleri
PROJECT_DIR = '/home/ubuntu/blood_cell_recognition'
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'optimized_mobilenet.h5')

# Sınıf isimleri
CLASS_NAMES = ['Platelets', 'RBC', 'WBC']

# Görüntü boyutu
IMG_SIZE = 224

class BloodCellDetectionApp:
    def __init__(self, root):
        """Uygulamayı başlatır ve arayüzü oluşturur."""
        self.root = root
        self.root.title("Kan Hücresi Tespit Uygulaması")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
        
        # Model yükleme
        try:
            self.model = load_model(MODEL_PATH)
            print("Model başarıyla yüklendi.")
        except Exception as e:
            print(f"Model yüklenirken hata oluştu: {e}")
            messagebox.showerror("Hata", f"Model yüklenirken hata oluştu: {e}")
            self.model = None
        
        # Ana çerçeve
        self.main_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Başlık
        self.title_label = tk.Label(
            self.main_frame, 
            text="Kan Hücresi Tespit Uygulaması", 
            font=("Arial", 18, "bold"),
            bg="#f0f0f0"
        )
        self.title_label.pack(pady=10)
        
        # Açıklama
        self.description_label = tk.Label(
            self.main_frame,
            text="Bu uygulama, kan hücresi görüntülerini analiz ederek RBC, WBC ve Platelet hücrelerini tespit eder.",
            font=("Arial", 12),
            wraplength=800,
            bg="#f0f0f0"
        )
        self.description_label.pack(pady=10)
        
        # Görüntü ve sonuç çerçevesi
        self.content_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Görüntü çerçevesi
        self.image_frame = tk.LabelFrame(
            self.content_frame,
            text="Görüntü",
            font=("Arial", 12),
            bg="#f0f0f0"
        )
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        # Görüntü etiketi
        self.image_label = tk.Label(self.image_frame, bg="white")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Sonuç çerçevesi
        self.result_frame = tk.LabelFrame(
            self.content_frame,
            text="Tespit Sonuçları",
            font=("Arial", 12),
            bg="#f0f0f0"
        )
        self.result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        # Grafik için figür
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.result_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Sonuç etiketi
        self.result_label = tk.Label(
            self.result_frame,
            text="Henüz bir görüntü analiz edilmedi.",
            font=("Arial", 12),
            bg="#f0f0f0",
            wraplength=400
        )
        self.result_label.pack(pady=10)
        
        # Buton çerçevesi
        self.button_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.button_frame.pack(fill=tk.X, pady=10)
        
        # Görüntü seç butonu
        self.select_button = tk.Button(
            self.button_frame,
            text="Görüntü Seç",
            font=("Arial", 12),
            command=self.select_image,
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=10
        )
        self.select_button.pack(side=tk.LEFT, padx=10)
        
        # Analiz et butonu
        self.analyze_button = tk.Button(
            self.button_frame,
            text="Analiz Et",
            font=("Arial", 12),
            command=self.analyze_image,
            bg="#2196F3",
            fg="white",
            padx=20,
            pady=10,
            state=tk.DISABLED
        )
        self.analyze_button.pack(side=tk.LEFT, padx=10)
        
        # Temizle butonu
        self.clear_button = tk.Button(
            self.button_frame,
            text="Temizle",
            font=("Arial", 12),
            command=self.clear_results,
            bg="#f44336",
            fg="white",
            padx=20,
            pady=10
        )
        self.clear_button.pack(side=tk.LEFT, padx=10)
        
        # Çıkış butonu
        self.exit_button = tk.Button(
            self.button_frame,
            text="Çıkış",
            font=("Arial", 12),
            command=self.root.quit,
            bg="#607D8B",
            fg="white",
            padx=20,
            pady=10
        )
        self.exit_button.pack(side=tk.RIGHT, padx=10)
        
        # Durum çubuğu
        self.status_bar = tk.Label(
            self.root,
            text="Hazır",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Görüntü değişkenleri
        self.image_path = None
        self.original_image = None
        self.processed_image = None
        
        # Başlangıç grafiği
        self.update_plot([0, 0, 0])
    
    def select_image(self):
        """Kullanıcının bir görüntü seçmesini sağlar."""
        self.image_path = filedialog.askopenfilename(
            title="Görüntü Seç",
            filetypes=[
                ("Görüntü Dosyaları", "*.jpg *.jpeg *.png *.bmp"),
                ("Tüm Dosyalar", "*.*")
            ]
        )
        
        if self.image_path:
            try:
                # Görüntüyü yükle
                self.original_image = cv2.imread(self.image_path)
                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                
                # Görüntüyü göster
                self.display_image(self.original_image)
                
                # Analiz butonunu etkinleştir
                self.analyze_button.config(state=tk.NORMAL)
                
                # Durum çubuğunu güncelle
                self.status_bar.config(text=f"Görüntü yüklendi: {os.path.basename(self.image_path)}")
            
            except Exception as e:
                messagebox.showerror("Hata", f"Görüntü yüklenirken hata oluştu: {e}")
                self.status_bar.config(text="Hata: Görüntü yüklenemedi")
    
    def display_image(self, img):
        """Seçilen görüntüyü arayüzde gösterir."""
        # Görüntüyü yeniden boyutlandır
        h, w = img.shape[:2]
        max_size = 400
        
        if h > max_size or w > max_size:
            if h > w:
                new_h = max_size
                new_w = int(w * (max_size / h))
            else:
                new_w = max_size
                new_h = int(h * (max_size / w))
            
            img = cv2.resize(img, (new_w, new_h))
        
        # Görüntüyü PIL formatına dönüştür
        pil_img = Image.fromarray(img)
        
        # Tkinter için PhotoImage oluştur
        tk_img = ImageTk.PhotoImage(pil_img)
        
        # Görüntüyü etikete yerleştir
        self.image_label.config(image=tk_img)
        self.image_label.image = tk_img  # Referansı koru
    
    def preprocess_image(self, img):
        """Görüntüyü model için hazırlar."""
        # Görüntüyü yeniden boyutlandır
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Görüntüyü normalize et
        img = img.astype(np.float32) / 255.0
        
        # Batch boyutunu ekle
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def analyze_image(self):
        """Seçilen görüntüyü analiz eder ve sonuçları gösterir."""
        if self.original_image is None or self.model is None:
            messagebox.showerror("Hata", "Lütfen önce bir görüntü seçin veya model yüklendiğinden emin olun.")
            return
        
        try:
            # Durum çubuğunu güncelle
            self.status_bar.config(text="Görüntü analiz ediliyor...")
            self.root.update()
            
            # Görüntüyü ön işle
            processed_img = self.preprocess_image(self.original_image)
            
            # Tahmin yap
            start_time = time.time()
            predictions = self.model.predict(processed_img)
            end_time = time.time()
            
            # Tahmin sonuçlarını al
            pred_probabilities = predictions[0]
            pred_class_idx = np.argmax(pred_probabilities)
            pred_class = CLASS_NAMES[pred_class_idx]
            confidence = pred_probabilities[pred_class_idx] * 100
            
            # Sonuç metnini oluştur
            result_text = f"Tespit Edilen Hücre: {pred_class}\n"
            result_text += f"Güven Oranı: {confidence:.2f}%\n"
            result_text += f"İşlem Süresi: {(end_time - start_time):.4f} saniye\n\n"
            
            # Tüm sınıflar için olasılıkları ekle
            for i, class_name in enumerate(CLASS_NAMES):
                result_text += f"{class_name}: {pred_probabilities[i] * 100:.2f}%\n"
            
            # Sonuç etiketini güncelle
            self.result_label.config(text=result_text)
            
            # Grafiği güncelle
            self.update_plot(pred_probabilities * 100)
            
            # Durum çubuğunu güncelle
            self.status_bar.config(text=f"Analiz tamamlandı: {pred_class} tespit edildi ({confidence:.2f}% güven)")
        
        except Exception as e:
            messagebox.showerror("Hata", f"Görüntü analiz edilirken hata oluştu: {e}")
            self.status_bar.config(text="Hata: Görüntü analiz edilemedi")
    
    def update_plot(self, probabilities):
        """Tahmin olasılıklarını gösteren çubuk grafiği günceller."""
        # Grafiği temizle
        self.ax.clear()
        
        # Çubuk grafiği oluştur
        bars = self.ax.bar(CLASS_NAMES, probabilities, color=['purple', 'red', 'blue'])
        
        # Grafiği özelleştir
        self.ax.set_ylim([0, 100])
        self.ax.set_ylabel('Olasılık (%)')
        self.ax.set_title('Hücre Tipi Tahmin Olasılıkları')
        
        # Çubukların üzerine değerleri ekle
        for bar in bars:
            height = bar.get_height()
            self.ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 1,
                f'{height:.1f}%',
                ha='center',
                va='bottom',
                rotation=0
            )
        
        # Grafiği güncelle
        self.canvas.draw()
    
    def clear_results(self):
        """Sonuçları ve görüntüyü temizler."""
        # Görüntüyü temizle
        self.image_label.config(image='')
        self.original_image = None
        self.image_path = None
        
        # Sonuç etiketini sıfırla
        self.result_label.config(text="Henüz bir görüntü analiz edilmedi.")
        
        # Grafiği sıfırla
        self.update_plot([0, 0, 0])
        
        # Analiz butonunu devre dışı bırak
        self.analyze_button.config(state=tk.DISABLED)
        
        # Durum çubuğunu güncelle
        self.status_bar.config(text="Hazır")

def main():
    """Ana işlev: Uygulamayı başlatır."""
    # Zaman takibi için başlangıç zamanını kaydet
    start_time = time.time()
    
    # Tkinter uygulamasını başlat
    root = tk.Tk()
    app = BloodCellDetectionApp(root)
    
    # İşlem süresini hesapla
    end_time = time.time()
    ui_init_time = end_time - start_time
    
    # Zaman bilgisini dosyaya kaydet
    with open(os.path.join(PROJECT_DIR, 'time_tracking.md'), 'a') as f:
        f.write(f"- Kullanıcı arayüzü başlatma: Başlangıç - {time.strftime('%d Nisan %Y %H:%M:%S')}, Bitiş - {time.strftime('%d Nisan %Y %H:%M:%S', time.localtime(end_time))}, Süre - {ui_init_time:.2f} saniye\n")
    
    # Uygulamayı çalıştır
    root.mainloop()

if __name__ == "__main__":
    main()
