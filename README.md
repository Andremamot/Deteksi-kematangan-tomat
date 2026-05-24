<div align="center">
  <h1>🍅 Deteksi Kematangan Tomat</h1>
  <p><i>Aplikasi Computer Vision Berbasis GUI dengan Python & OpenCV</i></p>
  
  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
  [![OpenCV](https://img.shields.io/badge/OpenCV-4.13+-green.svg?logo=opencv&logoColor=white)](https://opencv.org/)
  [![Tkinter](https://img.shields.io/badge/GUI-Tkinter-lightgrey.svg)](#)
</div>

---

Aplikasi ini menggunakan metode pengolahan citra digital untuk mendeteksi tingkat kematangan buah tomat secara otomatis. Algoritma membagi proses ke dalam tiga tahapan utama: **Preprocessing**, **Segmentasi**, dan **Klasifikasi**.

## ✨ Fitur Utama

- 🔍 **Deteksi Citra Tunggal**: Pilih satu gambar tomat dan amati proses dari citra asli, hasil segmentasi (penghapusan latar belakang), hingga prediksi akhir secara langsung berdampingan.
- 📁 **Evaluasi Dataset Batch**: Uji performa algoritma sekaligus pada puluhan hingga ratusan gambar. Program akan menganalisis keseluruhan *folder* dan memberikan ringkasan statistik deteksi.

## 📂 Struktur Folder & Penempatan Dataset

Agar aplikasi berjalan optimal, perhatikan letak folder dataset Anda. Buat sebuah folder bernama `dataset_tomat` di direktori utama proyek, dan masukkan seluruh gambar *sample* Anda ke dalamnya.

```text
Deteksi-kematangan-tomat/
├── app.py                     # File utama untuk menjalankan GUI
├── core/                      # Algoritma pemrosesan utama
│   ├── evaluator.py
│   └── image_processing.py
├── dataset_tomat/             # 📁 LETAKKAN SEMUA GAMBAR DATASET DI SINI (*.bmp, *.jpg)
├── environment.yml            # Konfigurasi environment Conda
├── requirements.txt           # Konfigurasi requirements pip
└── README.md                  # Dokumentasi proyek
```

---

## 🛠️ Langkah 1: Unduh Proyek & Dataset

Karena dataset gambar ukurannya besar, gambar tidak ikut diunggah ke GitHub. Ikuti langkah ini dari awal:

1. **Clone Repositori**:
   Buka terminal/CMD Anda, lalu jalankan:
   ```bash
   git clone https://github.com/Andremamot/Deteksi-kematangan-tomat.git
   cd Deteksi-kematangan-tomat
   ```
2. **Siapkan Dataset**:
   - Buat folder bernama `dataset_tomat` di dalam direktori proyek ini.
   - Pergi ke bagian **Releases** di sebelah kanan halaman GitHub ini (atau klik [di sini](https://github.com/Andremamot/Deteksi-kematangan-tomat/releases/download/v1.0/dataset_tomat.zip)).
   - Unduh file `dataset_tomat.zip` dari sana.
   - Ekstrak isi file zip tersebut (semua gambar `.bmp` / `.jpg`) dan masukkan ke dalam folder `dataset_tomat` yang baru saja Anda buat.

---

## ⚙️ Langkah 2: Persiapan Environment Python

Pastikan Anda memiliki [Python](https://www.python.org/downloads/) terinstal. Sangat disarankan menggunakan *virtual environment*.

### Opsi A: Menggunakan Conda (Sangat Disarankan)
Metode terbaik untuk menjaga *environment* tetap rapi dan terisolasi.
```bash
# 1. Buat environment baru dari file konfigurasi
conda env create -f environment.yml

# 2. Aktifkan environment
conda activate tomat_env
```

### Opsi B: Menggunakan Pip
Alternatif jika Anda menggunakan *virtual environment* Python bawaan (`venv`).
```bash
# 1. Buat virtual environment (Lewati jika sudah ada)
python -m venv .venv

# 2. Aktifkan virtual environment
# -> Linux/Mac (Bash/Zsh):
source .venv/bin/activate
# -> Linux/Mac (Fish Shell):
source .venv/bin/activate.fish
# -> Windows:
.venv\Scripts\activate

# 3. Instal semua dependencies
pip install -r requirements.txt
```

> **Catatan Khusus Pengguna Linux**: Jika aplikasi mengalami *error* `ImportError: libtk8.6.so` saat dijalankan, pastikan paket `tk` sudah terinstal di sistem operasi Anda (contoh di Arch Linux: `sudo pacman -S tk`).

---

## 🚀 Cara Menjalankan Aplikasi

Setelah environment aktif, jalankan perintah ini di terminal:
```bash
python app.py
```
*Jendela GUI aplikasi akan segera terbuka.*

### 📖 Panduan Penggunaan Antarmuka
1. **Deteksi Satuan**:
   - Klik tombol **Pilih Gambar** di panel kiri.
   - Arahkan ke file gambar di dalam folder `dataset_tomat`.
   - Lihat hasil segmentasi dan prediksi kematangan (Mentah/Setengah Matang/Matang) yang otomatis muncul.
   
2. **Uji Masal (Dataset)**:
   - Klik tombol **Uji Dataset**.
   - Pilih folder `dataset_tomat` yang berisi kumpulan gambar Anda.
   - Tunggu beberapa saat, lalu sistem akan memunculkan *pop-up* berisi laporan jumlah tomat di tiap kategori.

---
<p align="center"><i>Dibuat untuk analisis tugas pengolahan citra digital.</i></p>
