import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os

from core.image_processing import process_image
from core.evaluator import run_evaluation

class TomatoDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Deteksi Kematangan Tomat")
        self.root.geometry("900x600")
        
        # Frame Kontrol (Kiri)
        self.control_frame = tk.Frame(self.root, width=200, bg="#f0f0f0")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        tk.Label(self.control_frame, text="Kontrol", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=10)
        
        self.btn_select = tk.Button(self.control_frame, text="Pilih Gambar", command=self.load_image, width=20)
        self.btn_select.pack(pady=5)
        
        self.btn_test = tk.Button(self.control_frame, text="Uji Dataset", command=self.test_dataset, width=20)
        self.btn_test.pack(pady=5)
        
        self.lbl_result_title = tk.Label(self.control_frame, text="Hasil Klasifikasi:", font=("Arial", 12), bg="#f0f0f0")
        self.lbl_result_title.pack(pady=(30, 5))
        
        self.lbl_result = tk.Label(self.control_frame, text="-", font=("Arial", 16, "bold"), fg="blue", bg="#f0f0f0")
        self.lbl_result.pack(pady=5)
        
        # Frame Tampilan Gambar (Kanan)
        self.display_frame = tk.Frame(self.root)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Gambar Asli
        self.lbl_orig_title = tk.Label(self.display_frame, text="Citra Asli")
        self.lbl_orig_title.grid(row=0, column=0, pady=5)
        self.lbl_orig = tk.Label(self.display_frame, bg="gray", width=40, height=15)
        self.lbl_orig.grid(row=1, column=0, padx=10)
        
        # Hasil Segmentasi
        self.lbl_seg_title = tk.Label(self.display_frame, text="Hasil Segmentasi")
        self.lbl_seg_title.grid(row=0, column=1, pady=5)
        self.lbl_seg = tk.Label(self.display_frame, bg="gray", width=40, height=15)
        self.lbl_seg.grid(row=1, column=1, padx=10)
        
        self.current_image_path = None
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Pilih Gambar Tomat",
            filetypes=[("Image Files", "*.bmp *.jpg *.png *.jpeg")]
        )
        if file_path:
            self.current_image_path = file_path
            self.process_and_display()
            
    def process_and_display(self):
        orig, segmented, mask, label = process_image(self.current_image_path)
        
        if orig is None:
            messagebox.showerror("Error", label)
            return
            
        # Update Label
        self.lbl_result.config(text=label)
        if label == "Matang":
            self.lbl_result.config(fg="red")
        elif label == "Mentah":
            self.lbl_result.config(fg="green")
        else:
            self.lbl_result.config(fg="orange")
            
        # Tampilkan Gambar (Perlu konversi OpenCV BGR ke RGB, lalu ke ImageTk)
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        seg_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
        
        # Resize untuk display GUI (misal 300x300)
        orig_pil = Image.fromarray(orig_rgb).resize((300, 300))
        seg_pil = Image.fromarray(seg_rgb).resize((300, 300))
        
        self.orig_tk = ImageTk.PhotoImage(orig_pil)
        self.seg_tk = ImageTk.PhotoImage(seg_pil)
        
        self.lbl_orig.config(image=self.orig_tk, width=300, height=300)
        self.lbl_seg.config(image=self.seg_tk, width=300, height=300)
        
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
