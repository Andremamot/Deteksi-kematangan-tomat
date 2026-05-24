import cv2
import numpy as np

def process_image(image_path):
    """
    Melakukan proses deteksi kematangan tomat.
    Mengembalikan (original_img, segmented_img, mask, label_kematangan)
    """
    # 1. Preprocessing
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None, "Error: Gambar tidak valid"
    
    # Resize agar proses lebih cepat dan konsisten (misal: 400x400)
    img = cv2.resize(img, (400, 400))
    
    # Gaussian Blur untuk mengurangi noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Konversi ruang warna dari BGR ke HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # 2. Segmentasi (Memisahkan tomat dari background)
    # Mask untuk warna merah (Matang)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2
    
    # Mask untuk warna hijau (Mentah)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Mask untuk warna kuning/oranye (Setengah Matang)
    lower_yellow = np.array([11, 50, 50])
    upper_yellow = np.array([34, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Gabungkan mask untuk mendapatkan keseluruhan area tomat
    mask_tomato = mask_red + mask_green + mask_yellow
    
    # Operasi Morfologi untuk menghilangkan noise kecil dan mengisi lubang
    kernel = np.ones((5,5), np.uint8)
    mask_tomato = cv2.morphologyEx(mask_tomato, cv2.MORPH_CLOSE, kernel)
    mask_tomato = cv2.morphologyEx(mask_tomato, cv2.MORPH_OPEN, kernel)
    
    # Mengaplikasikan mask ke gambar asli (segmentasi)
    segmented = cv2.bitwise_and(img, img, mask=mask_tomato)
    
    # 3. Klasifikasi
    # Menghitung jumlah piksel untuk masing-masing warna pada area tomat
    red_pixels = cv2.countNonZero(mask_red)
    green_pixels = cv2.countNonZero(mask_green)
    yellow_pixels = cv2.countNonZero(mask_yellow)
    
    total_pixels = red_pixels + green_pixels + yellow_pixels
    
    if total_pixels == 0:
        return img, segmented, mask_tomato, "Bukan Tomat / Gagal Deteksi"
        
    p_red = red_pixels / total_pixels
    p_green = green_pixels / total_pixels
    p_yellow = yellow_pixels / total_pixels
    
    # Logika klasifikasi sederhana berdasarkan persentase warna dominan
    if p_red > 0.5:
        label = "Matang"
    elif p_green > 0.5:
        label = "Mentah"
    elif p_yellow > 0.5:
        label = "Setengah Matang"
    else:
        # Jika tidak ada yang mayoritas mutlak, cari yang paling banyak
        max_p = max(p_red, p_green, p_yellow)
        if max_p == p_red:
            label = "Matang"
        elif max_p == p_green:
            label = "Mentah"
        else:
            label = "Setengah Matang"
            
    return img, segmented, mask_tomato, label
