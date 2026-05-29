import cv2
import numpy as np


def process_image(image_path):
    """
    Melakukan proses deteksi kematangan tomat (versi ringkas untuk evaluator).
    Mengembalikan (original_img, segmented_img, mask, label_kematangan)
    """
    result = process_image_full(image_path)
    if result["original"] is None:
        return None, None, None, result["label"]
    return result["original"], result["segmented"], result["mask_final"], result["label"]


def process_image_full(image_path):
    """
    Melakukan proses deteksi kematangan tomat secara lengkap.
    Mengembalikan dict berisi semua tahapan pipeline:
      - original      : citra asli (BGR)
      - blurred       : hasil Gaussian Blur (BGR)
      - hsv_display   : citra HSV yang dikonversi balik ke BGR untuk ditampilkan
      - mask_red      : binary mask warna merah (grayscale)
      - mask_green    : binary mask warna hijau (grayscale)
      - mask_yellow   : binary mask warna kuning (grayscale)
      - mask_final    : mask setelah operasi morfologi (grayscale)
      - segmented     : hasil segmentasi (BGR)
      - label         : label kematangan ('Matang', 'Mentah', 'Setengah Matang', ...)
      - p_red         : persentase piksel merah (0.0–1.0)
      - p_green       : persentase piksel hijau (0.0–1.0)
      - p_yellow      : persentase piksel kuning (0.0–1.0)
    """
    EMPTY = {
        "original": None, "blurred": None, "hsv_display": None,
        "mask_red": None, "mask_green": None, "mask_yellow": None,
        "mask_final": None, "segmented": None,
        "label": "Error: Gambar tidak valid",
        "p_red": 0.0, "p_green": 0.0, "p_yellow": 0.0,
    }

    # ── 1. Preprocessing ──────────────────────────────────────────────────────
    img = cv2.imread(image_path)
    if img is None:
        return EMPTY

    # Resize ke ukuran seragam
    img = cv2.resize(img, (400, 400))

    # Gaussian Blur untuk meredam noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Konversi BGR → HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # HSV → BGR untuk keperluan tampilan (agar warna tampak jelas)
    hsv_display = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # ── 2. Segmentasi ─────────────────────────────────────────────────────────
    # Mask merah (Matang): dua rentang karena merah berada di kedua ujung Hue
    mask_r1 = cv2.inRange(hsv, np.array([0,  50, 50]), np.array([10,  255, 255]))
    mask_r2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    mask_red = cv2.add(mask_r1, mask_r2)

    # Mask hijau (Mentah)
    mask_green = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))

    # Mask kuning/oranye (Setengah Matang)
    mask_yellow = cv2.inRange(hsv, np.array([11, 50, 50]), np.array([34, 255, 255]))

    # Gabungkan semua mask
    mask_all = cv2.add(cv2.add(mask_red, mask_green), mask_yellow)

    # Operasi Morfologi: Closing lalu Opening untuk membersihkan mask
    kernel = np.ones((5, 5), np.uint8)
    mask_closed = cv2.morphologyEx(mask_all, cv2.MORPH_CLOSE, kernel)
    mask_final  = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    # Aplikasikan mask ke citra asli → hasil segmentasi
    segmented = cv2.bitwise_and(img, img, mask=mask_final)

    # ── 3. Klasifikasi ────────────────────────────────────────────────────────
    red_px    = cv2.countNonZero(mask_red)
    green_px  = cv2.countNonZero(mask_green)
    yellow_px = cv2.countNonZero(mask_yellow)
    total     = red_px + green_px + yellow_px

    if total == 0:
        label  = "Bukan Tomat / Gagal Deteksi"
        p_red = p_green = p_yellow = 0.0
    else:
        p_red    = red_px    / total
        p_green  = green_px  / total
        p_yellow = yellow_px / total

        if p_red > 0.5:
            label = "Matang"
        elif p_green > 0.5:
            label = "Mentah"
        elif p_yellow > 0.5:
            label = "Setengah Matang"
        else:
            max_p = max(p_red, p_green, p_yellow)
            if max_p == p_red:
                label = "Matang"
            elif max_p == p_green:
                label = "Mentah"
            else:
                label = "Setengah Matang"

    return {
        "original":    img,
        "blurred":     blurred,
        "hsv_display": hsv_display,
        "mask_red":    mask_red,
        "mask_green":  mask_green,
        "mask_yellow": mask_yellow,
        "mask_final":  mask_final,
        "segmented":   segmented,
        "label":       label,
        "p_red":       p_red,
        "p_green":     p_green,
        "p_yellow":    p_yellow,
    }
