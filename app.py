import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
import os

from core.image_processing import process_image_full
from core.evaluator import run_evaluation

# ─── Konstanta tampilan ───────────────────────────────────────────────────────
THUMB_W, THUMB_H = 240, 180      # ukuran tiap thumbnail pipeline
COLS = 4                         # jumlah kolom grid
ROWS = 2                         # jumlah baris grid
PAD  = 8                         # padding antar cell

LABEL_COLOR = {
    "Matang":                   "#e74c3c",   # merah
    "Mentah":                   "#27ae60",   # hijau
    "Setengah Matang":          "#e67e22",   # oranye
    "Bukan Tomat / Gagal Deteksi": "#7f8c8d",
}

BG_DARK   = "#1e1e2e"
BG_PANEL  = "#2a2a3e"
BG_CARD   = "#313149"
FG_MAIN   = "#ececec"
FG_SUB    = "#a0a0c0"
ACCENT    = "#7c5cbf"


def cv_to_pil(img_bgr, size=(THUMB_W, THUMB_H)):
    """Konversi citra OpenCV BGR → Pillow RGB dan resize."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize(size, Image.LANCZOS)
    return pil


def gray_to_pil(mask, size=(THUMB_W, THUMB_H)):
    """Konversi grayscale mask → Pillow dan resize."""
    pil = Image.fromarray(mask).resize(size, Image.LANCZOS)
    return pil


class PipelineCell(tk.Frame):
    """Frame satu sel pipeline: judul + canvas gambar."""

    def __init__(self, parent, title, **kwargs):
        super().__init__(parent, bg=BG_CARD, bd=0, **kwargs)
        self.title_label = tk.Label(
            self, text=title,
            bg=BG_CARD, fg=FG_SUB,
            font=("Segoe UI", 8, "bold"),
            pady=3,
        )
        self.title_label.pack()

        self.canvas = tk.Label(
            self, bg="#111122",
            width=THUMB_W, height=THUMB_H,
        )
        self.canvas.pack(padx=2, pady=(0, 4))
        self._photo = None

    def set_image(self, pil_img):
        self._photo = ImageTk.PhotoImage(pil_img)
        self.canvas.config(image=self._photo, width=THUMB_W, height=THUMB_H)

    def clear(self):
        self._photo = None
        self.canvas.config(image="")


class TomatoDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Deteksi Kematangan Tomat")
        self.root.configure(bg=BG_DARK)

        # Hitung ukuran window dari grid
        grid_w = COLS * (THUMB_W + PAD * 2) + PAD
        grid_h = ROWS * (THUMB_H + 30 + PAD * 2) + PAD
        win_w  = 220 + grid_w + 30
        win_h  = max(580, grid_h + 160)
        self.root.geometry(f"{win_w}x{win_h}")
        self.root.resizable(True, True)

        self._build_left_panel()
        self._build_right_panel()

    # ── PANEL KIRI (Kontrol) ──────────────────────────────────────────────────
    def _build_left_panel(self):
        panel = tk.Frame(self.root, bg=BG_PANEL, width=210)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0), pady=10)
        panel.pack_propagate(False)

        # Judul
        tk.Label(
            panel, text="🍅 Kontrol",
            bg=BG_PANEL, fg=FG_MAIN,
            font=("Segoe UI", 14, "bold"), pady=12,
        ).pack()

        ttk.Separator(panel, orient="horizontal").pack(fill=tk.X, padx=10)

        # Tombol Pilih Gambar
        self.btn_select = tk.Button(
            panel, text="📂  Pilih Gambar",
            command=self.load_image,
            bg=ACCENT, fg="white", activebackground="#9b77e0",
            font=("Segoe UI", 10, "bold"),
            relief="flat", cursor="hand2",
            pady=8, padx=10, width=18,
        )
        self.btn_select.pack(pady=(14, 6))

        # Tombol Uji Dataset
        self.btn_test = tk.Button(
            panel, text="📊  Uji Dataset",
            command=self.test_dataset,
            bg="#3a3a5e", fg=FG_MAIN, activebackground="#4a4a6e",
            font=("Segoe UI", 10, "bold"),
            relief="flat", cursor="hand2",
            pady=8, padx=10, width=18,
        )
        self.btn_test.pack(pady=6)

        ttk.Separator(panel, orient="horizontal").pack(fill=tk.X, padx=10, pady=(16, 0))

        # Label Hasil
        tk.Label(
            panel, text="Hasil Klasifikasi",
            bg=BG_PANEL, fg=FG_SUB,
            font=("Segoe UI", 9),
        ).pack(pady=(12, 2))

        self.lbl_result = tk.Label(
            panel, text="—",
            bg=BG_PANEL, fg=FG_SUB,
            font=("Segoe UI", 18, "bold"),
            wraplength=180, justify="center",
        )
        self.lbl_result.pack(pady=4)

        # Proporsi warna
        tk.Label(panel, text="Proporsi Warna Terdeteksi",
                 bg=BG_PANEL, fg=FG_SUB,
                 font=("Segoe UI", 8)).pack(pady=(20, 2))

        self.lbl_red    = tk.Label(panel, text="🔴 Merah : —",   bg=BG_PANEL, fg="#e74c3c", font=("Segoe UI", 9))
        self.lbl_green  = tk.Label(panel, text="🟢 Hijau  : —",  bg=BG_PANEL, fg="#27ae60", font=("Segoe UI", 9))
        self.lbl_yellow = tk.Label(panel, text="🟡 Kuning : —",  bg=BG_PANEL, fg="#f1c40f", font=("Segoe UI", 9))
        self.lbl_red.pack()
        self.lbl_green.pack()
        self.lbl_yellow.pack()

        self.current_image_path = None

    # ── PANEL KANAN (Pipeline Grid) ───────────────────────────────────────────
    def _build_right_panel(self):
        right = tk.Frame(self.root, bg=BG_DARK)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Judul pipeline
        self.title_bar = tk.Label(
            right,
            text="Tahapan Pipeline Deteksi Kematangan Tomat",
            bg=BG_DARK, fg=FG_MAIN,
            font=("Segoe UI", 11, "bold"),
            pady=8,
        )
        self.title_bar.pack()

        # Grid frame
        grid_frame = tk.Frame(right, bg=BG_DARK)
        grid_frame.pack(fill=tk.BOTH, expand=True)

        # Definisi 8 sel pipeline (baris, kolom, judul)
        self.pipeline_defs = [
            (0, 0, "1. Citra Asli\n(Input)"),
            (0, 1, "2. Gaussian Blur\n(Noise Reduction)"),
            (0, 2, "3. Konversi HSV\n(Color Space)"),
            (0, 3, "4. Mask Merah\n(Warna Matang)"),
            (1, 0, "5. Mask Hijau\n(Warna Mentah)"),
            (1, 1, "6. Mask Kuning\n(Setengah Matang)"),
            (1, 2, "7. Morfologi\n(Closing + Opening)"),
            (1, 3, "8. Hasil Segmentasi\n(Output)"),
        ]

        self.cells = []
        for row, col, title in self.pipeline_defs:
            cell = PipelineCell(grid_frame, title)
            cell.grid(row=row, column=col, padx=PAD, pady=PAD, sticky="nsew")
            self.cells.append(cell)

        for c in range(COLS):
            grid_frame.columnconfigure(c, weight=1)
        for r in range(ROWS):
            grid_frame.rowconfigure(r, weight=1)

        self._show_placeholder()

    def _show_placeholder(self):
        """Tampilkan placeholder abu-abu saat belum ada gambar."""
        placeholder = Image.new("RGB", (THUMB_W, THUMB_H), "#111122")
        draw = ImageDraw.Draw(placeholder)
        draw.text((THUMB_W//2 - 30, THUMB_H//2 - 8), "Pilih\nGambar",
                  fill="#555577", align="center")
        for cell in self.cells:
            cell.set_image(placeholder)

    # ── AKSI TOMBOL ───────────────────────────────────────────────────────────
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Pilih Gambar Tomat",
            filetypes=[("Image Files", "*.bmp *.jpg *.png *.jpeg")],
        )
        if file_path:
            self.current_image_path = file_path
            self.process_and_display()

    def process_and_display(self):
        data = process_image_full(self.current_image_path)

        if data["original"] is None:
            messagebox.showerror("Error", data["label"])
            return

        label  = data["label"]
        p_r    = data["p_red"]
        p_g    = data["p_green"]
        p_y    = data["p_yellow"]

        # ── Update label hasil ──
        color = LABEL_COLOR.get(label, FG_MAIN)
        self.lbl_result.config(text=label, fg=color)

        # ── Update proporsi warna ──
        self.lbl_red.config(   text=f"🔴 Merah  : {p_r*100:.1f}%")
        self.lbl_green.config( text=f"🟢 Hijau   : {p_g*100:.1f}%")
        self.lbl_yellow.config(text=f"🟡 Kuning  : {p_y*100:.1f}%")

        # ── Konversi semua intermediate step ke Pillow ──
        imgs = [
            cv_to_pil(data["original"]),
            cv_to_pil(data["blurred"]),
            cv_to_pil(data["hsv_display"]),
            gray_to_pil(data["mask_red"]),
            gray_to_pil(data["mask_green"]),
            gray_to_pil(data["mask_yellow"]),
            gray_to_pil(data["mask_final"]),
            cv_to_pil(data["segmented"]),
        ]

        # Beri border warna pada sel terakhir (hasil) sesuai kelas
        last_img = imgs[-1].copy()
        draw = ImageDraw.Draw(last_img)
        border_color = LABEL_COLOR.get(label, "#ffffff")
        for i in range(4):
            draw.rectangle([i, i, THUMB_W - 1 - i, THUMB_H - 1 - i],
                           outline=border_color)
        imgs[-1] = last_img

        for cell, pil_img in zip(self.cells, imgs):
            cell.set_image(pil_img)

        # Update judul bar
        self.title_bar.config(
            text=f"Pipeline Deteksi  —  {os.path.basename(self.current_image_path)}"
        )

    def test_dataset(self):
        dataset_dir = filedialog.askdirectory(title="Pilih Folder Dataset")
        if dataset_dir:
            success, report = run_evaluation(dataset_dir)
            if success:
                messagebox.showinfo("Hasil Pengujian Dataset", report)
            else:
                messagebox.showerror("Error", report)


if __name__ == "__main__":
    root = tk.Tk()
    app = TomatoDetectionApp(root)
    root.mainloop()
