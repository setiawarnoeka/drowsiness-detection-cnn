import cv2
import os

# Folder dataset custom
BASE_DIR = "dataset_custom"
AWAKE_DIR = os.path.join(BASE_DIR, "awake")
SLEEPY_DIR = os.path.join(BASE_DIR, "sleepy")

os.makedirs(AWAKE_DIR, exist_ok=True)
os.makedirs(SLEEPY_DIR, exist_ok=True)

# Hitung jumlah file yang sudah ada
awake_count = len(os.listdir(AWAKE_DIR))
sleepy_count = len(os.listdir(SLEEPY_DIR))

# Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Buka webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Webcam tidak bisa dibuka")
    exit()

mode = None
frame_count = 0
save_interval = 5  # simpan tiap 5 frame 

print("Tekan A untuk mode AWAKE")
print("Tekan S untuk mode SLEEPY")
print("Tekan P untuk PAUSE")
print("Tekan Q untuk keluar")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame gagal dibaca")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(120, 120)
    )

    eye_region = None

    if len(faces) > 0:
        # Ambil wajah terbesar
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

        # Crop area mata dari wajah
        x1 = x + int(0.10 * w)
        x2 = x + int(0.90 * w)
        y1 = y + int(0.20 * h)
        y2 = y + int(0.55 * h)

        eye_region = frame[y1:y2, x1:x2]

        # Kotak wajah
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Kotak area mata
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Simpan gambar sesuai mode
        if mode is not None and frame_count % save_interval == 0:
            resized_eye = cv2.resize(eye_region, (64, 64))

            if mode == "awake":
                filename = os.path.join(AWAKE_DIR, f"awake_{awake_count}.jpg")
                cv2.imwrite(filename, resized_eye)
                awake_count += 1

            elif mode == "sleepy":
                filename = os.path.join(SLEEPY_DIR, f"sleepy_{sleepy_count}.jpg")
                cv2.imwrite(filename, resized_eye)
                sleepy_count += 1

    # Tampilkan status
    cv2.putText(
        frame,
        f"Mode: {mode if mode else 'PAUSE'}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Awake: {awake_count} | Sleepy: {sleepy_count}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("Capture Dataset", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("a"):
        mode = "awake"
        print("Mode AWAKE aktif")

    elif key == ord("s"):
        mode = "sleepy"
        print("Mode SLEEPY aktif")

    elif key == ord("p"):
        mode = None
        print("Pause")

    elif key == ord("q"):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()