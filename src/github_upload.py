#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kan Hücresi Tespit Projesi - GitHub Yükleme Betiği
Bu betik, projenin tüm dosyalarını GitHub'a yükler.
"""

import os
import sys
import time
import subprocess
import argparse

# Zaman takibi için başlangıç zamanını kaydet
start_time = time.time()

# Proje dizinleri
PROJECT_DIR = '/home/ubuntu/blood_cell_recognition'
import os
# Token'ı çevre değişkeni olarak kullan
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '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')
GITHUB_USERNAME = 'burakkurt07'
REPO_NAME = 'blood-cell-detection'

def run_command(command):
    """Komutu çalıştırır ve çıktıyı döndürür."""
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Hata: {e}")
        print(f"Çıktı: {e.stdout}")
        print(f"Hata çıktısı: {e.stderr}")
        return None

def create_github_repo():
    """GitHub'da yeni bir depo oluşturur."""
    print("GitHub deposu oluşturuluyor...")
    
    # GitHub API kullanarak depo oluştur
    create_repo_cmd = f"""
    curl -X POST -H "Authorization: token {GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github.v3+json" \
    https://api.github.com/user/repos \
    -d '{{"name":"{REPO_NAME}","description":"Kan Hücresi Tespit Projesi - OpenCV ve Derin Öğrenme kullanarak kan hücrelerinin tespiti","private":false}}'
    """
    
    result = run_command(create_repo_cmd)
    if result:
        print(f"GitHub deposu başarıyla oluşturuldu: {GITHUB_USERNAME}/{REPO_NAME}")
        return True
    else:
        print("GitHub deposu oluşturulurken bir hata oluştu.")
        return False

def setup_git_repo():
    """Yerel Git deposunu ayarlar ve GitHub'a bağlar."""
    print("Git deposu ayarlanıyor...")
    
    # Git deposunu başlat
    os.chdir(PROJECT_DIR)
    
    # .gitignore dosyası oluştur
    with open(os.path.join(PROJECT_DIR, '.gitignore'), 'w') as f:
        f.write("__pycache__/\n*.py[cod]\n*$py.class\n*.so\n.Python\nenv/\nbuild/\ndevelop-eggs/\ndist/\ndownloads/\neggs/\n.eggs/\nlib/\nlib64/\nparts/\nsdist/\nvar/\n*.egg-info/\n.installed.cfg\n*.egg\n.env\n.venv\nvenv/\nENV/\n.DS_Store\n")
    
    # Git deposunu başlat
    run_command("git init")
    
    # Kullanıcı bilgilerini ayarla
    run_command(f'git config user.name "{GITHUB_USERNAME}"')
    run_command('git config user.email "github@example.com"')
    
    # Uzak depoyu ekle
    run_command(f'git remote add origin https://{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git')
    
    return True

def add_and_commit_files():
    """Tüm dosyaları ekler ve commit yapar."""
    print("Dosyalar ekleniyor ve commit yapılıyor...")
    
    # Tüm dosyaları ekle
    run_command("git add .")
    
    # Commit yap
    run_command('git commit -m "İlk commit: Kan Hücresi Tespit Projesi"')
    
    return True

def push_to_github():
    """Değişiklikleri GitHub'a gönderir."""
    print("Değişiklikler GitHub'a gönderiliyor...")
    
    # GitHub'a gönder
    result = run_command("git push -u origin master")
    
    if result is not None:
        print("Dosyalar başarıyla GitHub'a yüklendi.")
        return True
    else:
        print("Dosyalar GitHub'a yüklenirken bir hata oluştu.")
        return False

def main():
    """Ana işlev: GitHub'a yükleme işlemini gerçekleştirir."""
    print("Kan Hücresi Tespit Projesi GitHub'a yükleniyor...")
    
    # GitHub deposu oluştur
    if not create_github_repo():
        print("GitHub deposu oluşturulamadı, mevcut depo kullanılacak.")
    
    # Git deposunu ayarla
    if not setup_git_repo():
        print("Git deposu ayarlanamadı.")
        return False
    
    # Dosyaları ekle ve commit yap
    if not add_and_commit_files():
        print("Dosyalar eklenemedi veya commit yapılamadı.")
        return False
    
    # GitHub'a gönder
    if not push_to_github():
        print("Dosyalar GitHub'a gönderilemedi.")
        return False
    
    # İşlem süresini hesapla
    end_time = time.time()
    github_time = end_time - start_time
    
    # Zaman bilgisini dosyaya kaydet
    with open(os.path.join(PROJECT_DIR, 'time_tracking.md'), 'a') as f:
        f.write(f"- GitHub'a yükleme: Başlangıç - {time.strftime('%d Nisan %Y %H:%M:%S')}, Bitiş - {time.strftime('%d Nisan %Y %H:%M:%S', time.localtime(end_time))}, Süre - {github_time:.2f} saniye\n")
    
    print(f"\nGitHub'a yükleme tamamlandı!")
    print(f"Depo URL: https://github.com/{GITHUB_USERNAME}/{REPO_NAME}")
    print(f"Toplam işlem süresi: {github_time:.2f} saniye")
    
    return True

if __name__ == "__main__":
    main()
