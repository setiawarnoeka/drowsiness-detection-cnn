import cv2
import time
import os
import threading
import winsound
import numpy as np
from tensorflow.keras.models import load_model

# KONFIGURASI

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "drowsiness_model_custom_v1.keras")

SLEEPY_DURATION_THRESHOLD = 3
ALARM_INTERVAL = 0.3

WINDOW_NAME = "Sistem Deteksi Kantuk Pengemudi"

DASHBOARD_WIDTH = 1280
DASHBOARD_HEIGHT = 720

CAMERA_X = 30
CAMERA_Y = 80
CAMERA_WIDTH = 780
CAMERA_HEIGHT = 585

PANEL_X = 840
PANEL_Y = 80
PANEL_WIDTH = 410
PANEL_HEIGHT = 585

# LOAD MODEL

model = load_model(MODEL_PATH)
print("Model berhasil dimuat:", MODEL_PATH)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# FUNGSI ALARM

def play_alarm():
    winsound.Beep(1200, 300)

# FUNGSI UI

def put_text(img, text, pos, scale=0.55, color=(230, 230, 230), thickness=1):
    cv2.putText(
        img,
        text,
        pos,
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_bar(img, x, y, width, height, value, color):
    value = max(0.0, min(1.0, value))

    cv2.rectangle(img, (x, y), (x + width, y + height), (60, 60, 60), -1)

    fill_width = int(width * value)
    cv2.rectangle(img, (x, y), (x + fill_width, y + height), color, -1)

    cv2.rectangle(img, (x, y), (x + width, y + height), (180, 180, 180), 1)


def draw_panel(canvas, status, status_color, awake_prob, sleepy_prob, sleepy_duration):
    # Panel utama
    cv2.rectangle(
        canvas,
        (PANEL_X, PANEL_Y),
        (PANEL_X + PANEL_WIDTH, PANEL_Y + PANEL_HEIGHT),
        (35, 35, 35),
        -1
    )

    cv2.rectangle(
        canvas,
        (PANEL_X, PANEL_Y),
        (PANEL_X + PANEL_WIDTH, PANEL_Y + PANEL_HEIGHT),
        (130, 130, 130),
        1
    )

    put_text(canvas, "PANEL MONITORING", (PANEL_X + 25, PANEL_Y + 40), 0.7, (255, 255, 255), 1)
    cv2.line(
        canvas,
        (PANEL_X + 25, PANEL_Y + 55),
        (PANEL_X + PANEL_WIDTH - 25, PANEL_Y + 55),
        (120, 120, 120),
        1
    )

    # Status
    put_text(canvas, "Status Deteksi", (PANEL_X + 25, PANEL_Y + 95), 0.55, (200, 200, 200), 1)

    status_box_y = PANEL_Y + 115
    cv2.rectangle(
        canvas,
        (PANEL_X + 25, status_box_y),
        (PANEL_X + PANEL_WIDTH - 25, status_box_y + 55),
        status_color,
        -1
    )

    put_text(canvas, status, (PANEL_X + 40, status_box_y + 35), 0.65, (255, 255, 255), 1)

    # Probabilitas
    section_y = PANEL_Y + 220
    put_text(canvas, "Hasil Klasifikasi CNN", (PANEL_X + 25, section_y), 0.55, (200, 200, 200), 1)

    put_text(canvas, f"Awake  : {awake_prob * 100:.2f}%", (PANEL_X + 25, section_y + 40), 0.55)
    draw_bar(canvas, PANEL_X + 25, section_y + 55, 310, 18, awake_prob, (0, 150, 0))

    put_text(canvas, f"Sleepy : {sleepy_prob * 100:.2f}%", (PANEL_X + 25, section_y + 100), 0.55)
    draw_bar(canvas, PANEL_X + 25, section_y + 115, 310, 18, sleepy_prob, (0, 0, 180))

    # Timer
    timer_y = PANEL_Y + 410
    put_text(canvas, "Durasi Kondisi Sleepy", (PANEL_X + 25, timer_y), 0.55, (200, 200, 200), 1)

    put_text(
        canvas,
        f"{sleepy_duration:.1f} / {SLEEPY_DURATION_THRESHOLD} detik",
        (PANEL_X + 25, timer_y + 40),
        0.65,
        (255, 255, 255),
        1
    )

    timer_value = sleepy_duration / SLEEPY_DURATION_THRESHOLD
    draw_bar(canvas, PANEL_X + 25, timer_y + 60, 310, 20, timer_value, (0, 165, 255))


# WEBCAM

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Webcam tidak dapat dibuka")
    exit()

print("Webcam berhasil dijalankan")
print("Tekan Q untuk keluar")

sleepy_start_time = None
last_alarm_time = 0

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, DASHBOARD_WIDTH, DASHBOARD_HEIGHT)

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

    status = "WAJAH TIDAK TERDETEKSI"
    status_color = (90, 90, 90)

    awake_prob = 0.0
    sleepy_prob = 0.0
    sleepy_duration = 0.0

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

        # Crop area mata sesuai dataset custom
        x1 = x + int(0.10 * w)
        x2 = x + int(0.90 * w)
        y1 = y + int(0.20 * h)
        y2 = y + int(0.55 * h)

        eye_region = frame[y1:y2, x1:x2]

        if eye_region.size != 0:
            eye_region_rgb = cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB)
            resized_eye = cv2.resize(eye_region_rgb, (64, 64))

            # Tidak normalisasi manual karena model sudah punya Rescaling(1./255)
            input_data = resized_eye.reshape(1, 64, 64, 3)

            prediction = model.predict(input_data, verbose=0)

            awake_prob = float(prediction[0][0])
            sleepy_prob = float(prediction[0][1])

            if sleepy_prob > awake_prob:
                status = "TERINDIKASI MENGANTUK"
                status_color = (0, 140, 255)

                if sleepy_start_time is None:
                    sleepy_start_time = time.time()

                sleepy_duration = time.time() - sleepy_start_time

                if sleepy_duration >= SLEEPY_DURATION_THRESHOLD:
                    status = "PERINGATAN KANTUK"
                    status_color = (0, 0, 180)

                    current_time = time.time()
                    if current_time - last_alarm_time >= ALARM_INTERVAL:
                        threading.Thread(target=play_alarm, daemon=True).start()
                        last_alarm_time = current_time
            else:
                status = "NORMAL"
                status_color = (0, 150, 0)
                sleepy_start_time = None
                sleepy_duration = 0.0

            # Bounding box pada frame asli
            cv2.rectangle(frame, (x, y), (x + w, y + h), (180, 180, 180), 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
            put_text(frame, "Area Mata", (x1, y1 - 8), 0.45, status_color, 1)

    else:
        sleepy_start_time = None

    if sleepy_start_time is not None:
        sleepy_duration = time.time() - sleepy_start_time


    # DASHBOARD CANVAS


    canvas = np.zeros((DASHBOARD_HEIGHT, DASHBOARD_WIDTH, 3), dtype=np.uint8)
    canvas[:] = (45, 45, 45)

    # Header
    cv2.rectangle(canvas, (0, 0), (DASHBOARD_WIDTH, 60), (30, 30, 30), -1)
    put_text(canvas, "SISTEM DETEKSI KANTUK PENGEMUDI BERBASIS CNN", (30, 38), 0.75, (255, 255, 255), 1)

    # Resize kamera
    camera_view = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))

    # Border area kamera
    cv2.rectangle(
        canvas,
        (CAMERA_X - 1, CAMERA_Y - 1),
        (CAMERA_X + CAMERA_WIDTH + 1, CAMERA_Y + CAMERA_HEIGHT + 1),
        (130, 130, 130),
        1
    )

    canvas[
        CAMERA_Y:CAMERA_Y + CAMERA_HEIGHT,
        CAMERA_X:CAMERA_X + CAMERA_WIDTH
    ] = camera_view

    put_text(canvas, "Tampilan Kamera", (CAMERA_X, CAMERA_Y - 15), 0.55, (220, 220, 220), 1)

    # Panel kanan
    draw_panel(canvas, status, status_color, awake_prob, sleepy_prob, sleepy_duration)

    # Footer
    put_text(canvas, "Tekan Q untuk keluar", (30, DASHBOARD_HEIGHT - 25), 0.5, (220, 220, 220), 1)

    cv2.imshow(WINDOW_NAME, canvas)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()