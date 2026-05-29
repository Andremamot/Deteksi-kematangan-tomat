import cv2
import numpy as np


def process_image(image_path):
    """
    Versi ringkas untuk evaluator.py — mengembalikan (orig, segmented, mask, label).
    """
    result = process_image_full(image_path)
    if result["original"] is None:
        return None, None, None, result["label"]
    return result["original"], result["segmented"], result["mask_final"], result["label"]


def process_image_full(image_path):
    """
    Pipeline lengkap deteksi kematangan tomat.
    Mengembalikan dict berisi semua tahapan dan metadata klasifikasi.

    Kunci:
      original, blurred, hsv_display,
      mask_red, mask_green, mask_yellow,
      mask_final, segmented,
      label, p_red, p_green, p_yellow
    """
    EMPTY = {
        "original": None, "blurred": None, "hsv_display": None,
        "mask_red": None, "mask_green": None, "mask_yellow": None,
        "mask_final": None, "segmented": None,
        "label": "Error: Gambar tidak valid",
        "p_red": 0.0, "p_green": 0.0, "p_yellow": 0.0,
    }

    # ── 1. PREPROCESSING ─────────────────────────────────────────────────────
    img = cv2.imread(image_path)
    if img is None:
        return EMPTY

    img = cv2.resize(img, (400, 400))

    # Gaussian Blur untuk meredam noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # BGR → HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # HSV → BGR untuk keperluan tampilan GUI
    hsv_display = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # ── 2. SEGMENTASI ────────────────────────────────────────────────────────
    # PERBAIKAN: Threshold Saturasi (S) dinaikkan ke ≥ 100
    # untuk mengecualikan background yang memiliki S rendah (S ≈ 43).
    # Tomat selalu memiliki S ≥ 150, sehingga threshold S=100 aman.

    # Mask Merah — Matang (dua rentang karena merah berada di kedua ujung Hue)
    mask_r1 = cv2.inRange(hsv, np.array([0,   100, 50]),  np.array([12,  255, 255]))
    mask_r2 = cv2.inRange(hsv, np.array([158, 100, 50]),  np.array([180, 255, 255]))
    mask_red = cv2.add(mask_r1, mask_r2)

    # Mask Hijau — Mentah (S ≥ 100 mengeliminasi background cyan/teal S≈43)
    mask_green = cv2.inRange(hsv, np.array([38, 100, 50]), np.array([82, 255, 255]))

    # Mask Kuning/Oranye — Setengah Matang
    mask_yellow = cv2.inRange(hsv, np.array([13, 100, 80]), np.array([37, 255, 255]))

    # Gabungkan semua mask warna tomat
    mask_all = cv2.add(cv2.add(mask_red, mask_green), mask_yellow)

    # Operasi Morfologi: Closing lalu Opening untuk membersihkan mask
    kernel = np.ones((7, 7), np.uint8)
    mask_closed = cv2.morphologyEx(mask_all, cv2.MORPH_CLOSE, kernel)
    mask_final  = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    # ── 3. ISOLASI KONTUR TERBESAR (ROI Tomat) ───────────────────────────────
    # Hanya hitung proporsi warna di dalam area objek terbesar (tomat)
    # agar noise dan sisa background tidak mempengaruhi klasifikasi.
    contours, _ = cv2.findContours(
        mask_final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > 800:          # minimal 800 piksel agar tidak salah objek
            roi_mask = np.zeros_like(mask_final)
            cv2.drawContours(roi_mask, [largest], -1, 255, thickness=cv2.FILLED)

            # Hitung piksel masing-masing warna di dalam ROI saja
            red_px    = cv2.countNonZero(cv2.bitwise_and(mask_red,    mask_red,    mask=roi_mask))
            green_px  = cv2.countNonZero(cv2.bitwise_and(mask_green,  mask_green,  mask=roi_mask))
            yellow_px = cv2.countNonZero(cv2.bitwise_and(mask_yellow, mask_yellow, mask=roi_mask))
        else:
            red_px = green_px = yellow_px = 0
    else:
        red_px = green_px = yellow_px = 0

    # Fallback: jika tidak ada kontur valid, hitung dari seluruh mask
    if red_px + green_px + yellow_px == 0:
        red_px    = cv2.countNonZero(mask_red)
        green_px  = cv2.countNonZero(mask_green)
        yellow_px = cv2.countNonZero(mask_yellow)

    # Segmentasi visual (apply mask ke citra asli)
    segmented = cv2.bitwise_and(img, img, mask=mask_final)

    # ── 4. KLASIFIKASI ────────────────────────────────────────────────────────
    total = red_px + green_px + yellow_px

    if total == 0:
        return {**EMPTY,
                "original": img, "blurred": blurred, "hsv_display": hsv_display,
                "mask_red": mask_red, "mask_green": mask_green,
                "mask_yellow": mask_yellow, "mask_final": mask_final,
                "segmented": segmented,
                "label": "Bukan Tomat / Gagal Deteksi"}

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
