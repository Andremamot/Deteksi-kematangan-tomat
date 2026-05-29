"""
Script untuk menghasilkan gambar tahapan pipeline dan hasil evaluasi akurasi.
Menyimpan semua output ke folder report_images/.
"""
import cv2
import numpy as np
import os
import glob
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from core.image_processing import process_image

OUTPUT_DIR = "report_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 1. PILIH 3 SAMPEL CITRA (Matang, Mentah, Campur)
# ──────────────────────────────────────────────
SAMPLES = {
    "Matang":         "dataset_tomat/merah.bmp",
    "Mentah":         "dataset_tomat/hijau.bmp",
    "Setengah_Matang": "dataset_tomat/campur.bmp",
}

def generate_pipeline_figure(img_path, label_kelas):
    """Hasilkan satu figure berisi semua tahapan pipeline untuk satu citra."""
    img = cv2.imread(img_path)
    img = cv2.resize(img, (400, 400))

    # --- Tahap 1: Preprocessing ---
    blurred      = cv2.GaussianBlur(img, (5, 5), 0)
    hsv          = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # --- Tahap 2: Segmentasi ---
    # Mask merah
    mask_r1 = cv2.inRange(hsv, np.array([0,  50, 50]), np.array([10,  255, 255]))
    mask_r2 = cv2.inRange(hsv, np.array([170,50, 50]), np.array([180, 255, 255]))
    mask_red   = mask_r1 + mask_r2
    mask_green  = cv2.inRange(hsv, np.array([35, 40, 40]),  np.array([85,  255, 255]))
    mask_yellow = cv2.inRange(hsv, np.array([11, 50, 50]),  np.array([34,  255, 255]))
    mask_all    = mask_red + mask_green + mask_yellow

    # Morfologi
    kernel = np.ones((5,5), np.uint8)
    mask_closed = cv2.morphologyEx(mask_all, cv2.MORPH_CLOSE, kernel)
    mask_final  = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    # Segmentasi final
    segmented = cv2.bitwise_and(img, img, mask=mask_final)

    # --- Hitung proporsi warna ---
    red_px    = cv2.countNonZero(mask_red)
    green_px  = cv2.countNonZero(mask_green)
    yellow_px = cv2.countNonZero(mask_yellow)
    total     = red_px + green_px + yellow_px or 1
    p_r = red_px / total * 100
    p_g = green_px / total * 100
    p_y = yellow_px / total * 100

    # ── Convert ke RGB untuk matplotlib ──
    def bgr2rgb(im): return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    hsv_display = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(f"Pipeline Deteksi Kematangan Tomat  —  Kelas: {label_kelas.replace('_',' ')}",
                 fontsize=16, fontweight='bold', y=1.01)

    panels = [
        (bgr2rgb(img),         "1. Citra Asli\n(Input)"),
        (bgr2rgb(blurred),     "2. Gaussian Blur\n(Noise Reduction)"),
        (bgr2rgb(hsv_display), "3. Konversi HSV\n(Color Space)"),
        (mask_red,             f"4. Mask Merah\n({p_r:.1f}% piksel)"),
        (mask_green,           f"5. Mask Hijau\n({p_g:.1f}% piksel)"),
        (mask_yellow,          f"6. Mask Kuning\n({p_y:.1f}% piksel)"),
        (mask_final,           "7. Morfologi\n(Closing + Opening)"),
        (bgr2rgb(segmented),   f"8. Hasil Segmentasi\n→ {label_kelas.replace('_',' ')}"),
    ]

    for ax, (image, title) in zip(axes.flat, panels):
        if len(image.shape) == 2:  # grayscale / mask
            ax.imshow(image, cmap='gray')
        else:
            ax.imshow(image)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"pipeline_{label_kelas.lower()}.png")
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Tersimpan: {save_path}")
    return save_path


# ──────────────────────────────────────────────
# 2. GENERATE PIPELINE UNTUK TIAP SAMPEL
# ──────────────────────────────────────────────
print("\n[1/2] Menghasilkan gambar pipeline per kelas...")
for kelas, path in SAMPLES.items():
    if os.path.exists(path):
        generate_pipeline_figure(path, kelas)
    else:
        print(f"  ✗ File tidak ditemukan: {path}")


# ──────────────────────────────────────────────
# 3. EVALUASI AKURASI PADA DATASET
# ──────────────────────────────────────────────
print("\n[2/2] Menjalankan evaluasi pada dataset_tomat/...")

dataset_path = "dataset_tomat"
image_files = (glob.glob(os.path.join(dataset_path, "*.bmp")) +
               glob.glob(os.path.join(dataset_path, "*.jpg")) +
               glob.glob(os.path.join(dataset_path, "*.png")))

results       = {"Matang": 0, "Setengah Matang": 0, "Mentah": 0, "Lainnya": 0}
correct       = 0
total_labeled = 0
total_proc    = 0

detail_rows = []

for img_path in sorted(image_files):
    basename = os.path.basename(img_path).lower()
    _, _, _, label = process_image(img_path)

    if label in results:
        results[label] += 1
    else:
        results["Lainnya"] += 1
    total_proc += 1

    gt = None
    if 'merah' in basename:
        gt = "Matang"
    elif 'hijau' in basename:
        gt = "Mentah"
    elif 'campur' in basename:
        gt = "Setengah Matang"

    if gt:
        total_labeled += 1
        ok = (label == gt)
        if ok:
            correct += 1
        detail_rows.append((os.path.basename(img_path), gt, label, "✓" if ok else "✗"))

accuracy = (correct / total_labeled * 100) if total_labeled > 0 else None

print(f"  Total citra diproses : {total_proc}")
print(f"  Matang               : {results['Matang']}")
print(f"  Setengah Matang      : {results['Setengah Matang']}")
print(f"  Mentah               : {results['Mentah']}")
print(f"  Lainnya/Gagal        : {results['Lainnya']}")
if accuracy is not None:
    print(f"  Akurasi (berlabel)   : {correct}/{total_labeled} = {accuracy:.2f}%")


# ──────────────────────────────────────────────
# 4. FIGURE EVALUASI / AKURASI
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Hasil Evaluasi Sistem Deteksi Kematangan Tomat", fontsize=16, fontweight='bold')

# ── (a) Distribusi hasil klasifikasi (pie chart) ──
ax1 = axes[0]
labels_pie  = [k for k, v in results.items() if v > 0]
sizes_pie   = [v for v in results.values() if v > 0]
colors_pie  = ['#e74c3c', '#f39c12', '#2ecc71', '#95a5a6'][:len(labels_pie)]
wedges, texts, autotexts = ax1.pie(
    sizes_pie, labels=labels_pie, colors=colors_pie,
    autopct='%1.1f%%', startangle=140,
    textprops={'fontsize': 11}
)
for at in autotexts:
    at.set_fontweight('bold')
ax1.set_title(f"Distribusi Klasifikasi\n(Total: {total_proc} citra)", fontsize=13, fontweight='bold')

# ── (b) Bar chart jumlah per kelas ──
ax2 = axes[1]
kelas_names  = list(results.keys())
kelas_counts = list(results.values())
bar_colors   = ['#e74c3c', '#f39c12', '#2ecc71', '#95a5a6']
bars = ax2.bar(kelas_names, kelas_counts, color=bar_colors, edgecolor='white', linewidth=1.2, width=0.55)
for bar, val in zip(bars, kelas_counts):
    if val > 0:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(val), ha='center', va='bottom', fontweight='bold', fontsize=12)
ax2.set_title("Jumlah Citra per Kelas", fontsize=13, fontweight='bold')
ax2.set_ylabel("Jumlah Citra", fontsize=11)
ax2.set_ylim(0, max(kelas_counts) * 1.2 + 2)
ax2.tick_params(axis='x', labelsize=10)
ax2.spines[['top', 'right']].set_visible(False)

# ── (c) Tabel hasil pengujian berlabel + akurasi ──
ax3 = axes[2]
ax3.axis('off')

if detail_rows:
    col_labels = ["File", "Ground Truth", "Prediksi", "Status"]
    cell_colors = []
    for row in detail_rows:
        c = '#d5f5e3' if row[3] == '✓' else '#fadbd8'
        cell_colors.append([c] * 4)

    tbl = ax3.table(
        cellText=detail_rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        cellColours=cell_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    # Header warna
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#2c3e50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

acc_text = f"Akurasi: {correct}/{total_labeled} = {accuracy:.2f}%" if accuracy is not None else "Tidak ada ground truth"
ax3.set_title(f"Pengujian Citra Berlabel\n{acc_text}", fontsize=13, fontweight='bold')

plt.tight_layout()
eval_path = os.path.join(OUTPUT_DIR, "evaluasi_akurasi.png")
plt.savefig(eval_path, dpi=130, bbox_inches='tight')
plt.close()
print(f"  ✓ Tersimpan: {eval_path}")

print("\n✅ Semua gambar berhasil disimpan di folder report_images/")
