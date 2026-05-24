# Deteksi Kematangan Tomat 🍅

Aplikasi berbasis GUI (Graphical User Interface) yang dibuat menggunakan Python dan OpenCV untuk mendeteksi tingkat kematangan tomat berdasarkan pengolahan citra digital. Aplikasi ini membagi tahapan menjadi Preprocessing, Segmentasi, dan Klasifikasi.

## Fitur Utama
1. **Deteksi Citra Tunggal**: Memilih satu file gambar tomat untuk melihat gambar asli dan hasil segmentasi berdampingan beserta prediksi label kematangan.
2. **Evaluasi Dataset Batch**: Memproses sekumpulan besar gambar dalam satu folder untuk melihat distribusi hasil deteksi dan akurasi (jika gambar berlabel).

## Persyaratan (Prerequisites)
Pastikan Anda sudah menginstal [Miniconda/Anaconda](https://docs.conda.io/en/latest/miniconda.html) atau Python standar di sistem Anda.

---

## 🛠️ Cara Instalasi & Persiapan

Anda bisa memilih salah satu dari dua metode di bawah ini. Metode pertama (Conda) sangat direkomendasikan.

### Opsi 1: Menggunakan Conda (Direkomendasikan)
Gunakan opsi ini agar lingkungan Python tertata rapi dan tidak mengganggu proyek Anda yang lain.

1. Buka terminal atau Anaconda Prompt.
2. Navigasikan ke dalam folder proyek ini:
   ```bash
   cd /path/ke/folder/Deteksi-kematangan-tomat
   ```
3. Buat environment baru dengan conda (menggunakan file `environment.yml`):
   ```bash
   conda env create -f environment.yml
   ```
   *(Alternatif jika Anda ingin menginstal manual dengan conda)*:
   ```bash
   conda create -n tomat_env python=3.10 opencv numpy pillow tk -y
   ```
4. Aktifkan environment yang baru saja dibuat:
   ```bash
   conda activate tomat_env
   ```

### Opsi 2: Menggunakan pip (Untuk pengguna Non-Conda)
Opsi ini cocok jika teman Anda hanya menggunakan Python standar tanpa Conda.

1. Buka terminal/Command Prompt.
2. Navigasikan ke dalam folder proyek ini:
   ```bash
   cd /path/ke/folder/Deteksi-kematangan-tomat
   ```
3. *(Opsional namun sangat disarankan)* Buat virtual environment pip:
   ```bash
   python -m venv venv
   # Aktivasi (Windows): venv\Scripts\activate
   # Aktivasi (Linux/Mac - Bash/Zsh): source venv/bin/activate
   # Aktivasi (Linux/Mac - Fish): source venv/bin/activate.fish
   ```
4. Instal paket menggunakan `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Cara Menjalankan Aplikasi

Setelah Anda menyelesaikan tahap Instalasi dan mengaktifkan environment (conda atau pip), jalankan perintah berikut di dalam terminal:

```bash
python app.py
```

Jendela antarmuka GUI akan segera terbuka.

## 📖 Panduan Penggunaan

### 1. Mendeteksi Gambar Tunggal
- Klik tombol **"Pilih Gambar"** di sisi kiri jendela.
- Cari dan pilih satu file gambar tomat (ekstensi `.bmp`, `.jpg`, `.png`).
- Sistem akan langsung memproses gambar tersebut.
- Hasil yang ditampilkan meliputi:
  - **Citra Asli** di sebelah kiri.
  - **Hasil Segmentasi** (hanya area tomat tanpa *background*) di sebelah kanan.
  - **Hasil Klasifikasi** (Matang / Setengah Matang / Mentah) di panel kiri bawah.

### 2. Menguji Kumpulan Dataset (Batch Testing)
- Klik tombol **"Uji Dataset"** di sisi kiri jendela.
- Pilih folder `dataset_tomat` (atau folder lain yang berisi koleksi gambar tomat).
- Klik "Pilih" atau "Open".
- Aplikasi akan menjalankan algoritma pada seluruh gambar yang ada di folder tersebut di latar belakang.
- Sebuah *Pop-up Window* (Pesan Info) akan muncul berisi total gambar yang diproses, distribusi kematangan (jumlah Matang, Setengah Matang, Mentah), dan estimasi akurasi apabila file gambar memiliki label tertentu pada namanya.

---
*Dibuat untuk kebutuhan tugas/analisis pengolahan citra digital.*
