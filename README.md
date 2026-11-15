
# Sistem Deteksi Kantuk Pengemudi Berbasis CNN

Sistem deteksi kantuk pengemudi secara real-time menggunakan kamera laptop. Sistem mengklasifikasikan kondisi mata (`awake` / `sleepy`) dengan Convolutional Neural Network (CNN), lalu membunyikan alarm jika kondisi mengantuk terdeteksi terus-menerus.

Proyek ini saya kerjakan sebagai Penulisan Ilmiah (PI) di Universitas Gunadarma, dengan mengikuti tahapan CRISP-DM.

## Latar Belakang

Kantuk saat mengemudi menurunkan kewaspadaan dan meningkatkan risiko kecelakaan lalu lintas. Proyek ini adalah prototipe awal untuk membantu mencegah kecelakaan akibat kantuk.

## Cara Kerja

1. Kamera menangkap gambar pengemudi secara real-time (OpenCV).
2. Wajah dideteksi dengan Haar Cascade, lalu area mata di-crop dari wajah terbesar.
3. Citra area mata diubah ukurannya menjadi 64 × 64 piksel dan diklasifikasikan oleh model CNN menjadi `awake` atau `sleepy`.
4. Jika kondisi `sleepy` terdeteksi terus-menerus selama 3 detik, sistem membunyikan alarm.
5. Dashboard menampilkan status, probabilitas tiap kelas, dan durasi kondisi sleepy.

## Dataset

Dataset dikumpulkan sendiri menggunakan kamera laptop lewat `scripts/capture_dataset.py`. Isinya citra area mata berukuran 64 × 64 piksel dalam dua kelas, `awake` dan `sleepy`. Dataset ikut dibagikan di repo ini supaya hasilnya bisa direproduksi.

- `dataset_custom/`: citra hasil pengambilan kamera.
- `dataset_custom_split/`: dataset hasil pembagian yang dipakai untuk pelatihan dan evaluasi model.

| Split           | Awake | Sleepy | Total           |
| --------------- | ----- | ------ | --------------- |
| Train           | 450   | 455    | 905             |
| Validation      | 136   | 136    | 272             |
| Test            | 134   | 135    | 269             |
| **Total** | 720   | 726    | **1.446** |

Pembagian data 70% train, 15% validation, 15% test (`scripts/split_dataset_custom.py`).

## Arsitektur Model

CNN dengan input 64 × 64 × 3:

- Augmentasi data: flip horizontal, rotasi, zoom, contrast, brightness
- Rescaling (1/255)
- 3 blok Conv2D (32, 64, 128 filter, kernel 3 × 3) + MaxPooling
- Flatten, Dense 128 (ReLU), Dropout 0,4
- Dense 2 (Softmax)

Pelatihan: optimizer Adam, loss sparse categorical crossentropy, 20 epoch, batch size 32.

## Hasil

| Metrik                  | Nilai            |
| ----------------------- | ---------------- |
| Training accuracy       | 98,67%           |
| Validation accuracy     | 100%             |
| **Test accuracy** | **99,26%** |
| Test loss               | 0,0146           |

Classification report pada data test (269 citra):

| Kelas  | Precision | Recall | F1-score | Support |
| ------ | --------- | ------ | -------- | ------- |
| awake  | 1,0000    | 0,9851 | 0,9925   | 134     |
| sleepy | 0,9854    | 1,0000 | 0,9926   | 135     |

Pengujian black box menunjukkan seluruh fungsi sistem berjalan sesuai skenario.

## Struktur Folder

```
├── dataset_custom/          # citra hasil pengambilan kamera
├── dataset_custom_split/    # dataset hasil pembagian train / val / test
├── models/                  # model CNN terlatih (.keras)
├── notebooks/
│   └── training_cnn_custom.ipynb
├── results/                 # classification report & ringkasan pelatihan
├── scripts/
│   ├── capture_dataset.py       # pengambilan dataset dari kamera
│   ├── split_dataset_custom.py  # pembagian dataset
│   └── realtime_detection.py    # deteksi real-time + alarm + dashboard
└── requirements.txt
```

## Cara Menjalankan

Persyaratan: Python. Skrip deteksi real-time memakai `winsound` untuk alarm, jadi berjalan di Windows.

```bash
pip install -r requirements.txt
```

**1. Ambil dataset (opsional)**

```bash
python scripts/capture_dataset.py
```

Tombol: `A` mode awake, `S` mode sleepy, `P` pause, `Q` keluar.

**2. Bagi dataset (opsional)**

```bash
python scripts/split_dataset_custom.py
```

**3. Latih model (opsional)**

Jalankan `notebooks/training_cnn_custom.ipynb`.

**4. Jalankan deteksi real-time**

```bash
python scripts/realtime_detection.py
```

Skrip ini memuat model `models/drowsiness_model_custom_v1.keras`. Tekan `Q` untuk keluar.

## Teknologi

Python, TensorFlow / Keras, OpenCV, NumPy, Matplotlib, scikit-learn.

## Keterbatasan

- Dataset dikumpulkan dari satu orang dengan kamera laptop, sehingga performa pada orang lain, kondisi cahaya berbeda, atau kacamata belum diuji.
- Haar Cascade bisa gagal mendeteksi wajah jika posisi wajah miring atau sebagian tertutup.
- Alarm memakai `winsound`, sehingga hanya berjalan di Windows.

## Penulis

Eka Rezky Setiawarno, Informatika, Universitas Gunadarma.
