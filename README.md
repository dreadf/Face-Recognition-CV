# Face Recognition (OpenCV + LBPH)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Cara menjalankan

Jalankan urutan di bawah dari folder root project ini.

1. **Ambil data wajah** (ganti `person_id` di `face_create_dataset.py` dulu sebelum run):
   ```bash
   python3 face_create_dataset.py
   ```
2. **Latih model:**
   ```bash
   python3 face_training.py
   ```
3. **Jalankan pengenalan wajah real-time:**
   ```bash
   python3 face_recognition.py
   ```

Tekan `q` di jendela kamera untuk keluar dari program manapun.

## Evaluasi (opsional)

Cek akurasi model dengan train/test split:
```bash
python3 evaluate.py
```
