# Rencana Implementasi: Deteksi Kematangan Tomat

Aplikasi ini akan mendeteksi tingkat kematangan tomat dari sebuah citra menggunakan metode pengolahan citra digital dengan Python dan OpenCV.

## Struktur Proyek

- `requirements.txt`: Daftar pustaka yang dibutuhkan (`opencv-python`, `numpy`, `Pillow`, `customtkinter`).
- `core/image_processing.py`: Modul pemrosesan citra (Preprocessing, Segmentasi, Klasifikasi).
- `core/evaluator.py`: Modul untuk memproses 100+ gambar dataset dan menyimpan/menampilkan hasil prediksi.
- `app.py`: File utama antarmuka pengguna (GUI).

## Tahapan Pemrosesan (3 Tahapan Berbeda)
1. **Preprocessing**: *Resize* gambar, *Noise Reduction* (Gaussian Blur), dan konversi BGR ke HSV.
2. **Segmentasi**: *Color Thresholding* pada ruang warna HSV untuk memisahkan objek tomat dari latar (masking) + operasi morfologi.
3. **Klasifikasi**: Menghitung porsi warna (Merah, Kuning/Oranye, Hijau) pada area tersegmentasi untuk menentukan kematangan: **Mentah**, **Setengah Matang**, atau **Matang**.

## Evaluasi
Membaca gambar dari folder `dataset_tomat` (1.bmp hingga selesai) dan mengekstrak prediksi akurasi atau mendemonstrasikan proses otomatisasi pada 100 citra.
