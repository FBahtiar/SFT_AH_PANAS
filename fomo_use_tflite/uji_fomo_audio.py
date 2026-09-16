import cv2
import winsound
import numpy as np
import tensorflow as tf
import time
import math
import serial
import threading

MAGIC = bytes([0xFF, 0xFE, 0xFF, 0xFE])

# =====================================================================
# 1. HELPER AUDIO NON-BLOCKING (Agar Beep Tidak Membekukan Frame)
# =====================================================================
def play_beep_async(freq, duration=25):
    """Menjalankan Beep di background thread agar tidak bikin video lag."""
    threading.Thread(target=winsound.Beep, args=(freq, duration), daemon=True).start()

# =====================================================================
# 2. SERIAL HELPER FUNCTIONS WITH FLUSHING
# =====================================================================
def read_exact(ser, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            raise serial.SerialTimeoutException("Timeout serial")
        buf.extend(chunk)
    return bytes(buf)

def sync_to_magic(ser):
    """Mencari magic marker sambil membuang frame tua jika buffer menumpuk."""
    # FAKTOR UTAMA PENYEBAB DELAY:
    # Jika data di buffer serial sudah menumpuk terlalu banyak (> 5KB), 
    # buang semuanya agar kita hanya membaca frame TERBARU dari ESP32-S3.
    if ser.in_waiting > 5000:
        ser.reset_input_buffer()

    window = bytearray(4)
    while True:
        b = ser.read(1)
        if not b:
            continue
        window = window[1:] + bytearray(b)
        if bytes(window) == MAGIC:
            return

# Re-define custom loss
def fomo_loss(y_true, y_pred):
    y_pred_softmax = tf.nn.softmax(y_pred, axis=-1)
    return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred_softmax))

class SmoothValue:
    def __init__(self, alpha=0.30):
        self.alpha = alpha
        self.value = None

    def update(self, new_val):
        if self.value is None:
            self.value = new_val
        else:
            self.value = self.alpha * new_val + (1 - self.alpha) * self.value
        return self.value

# =====================================================================
# 3. MAIN INFERENCE LOOP
# =====================================================================
def run_trained_fomo_serial(model_path="fomo_pure_face_esp32.h5", port="COM6", baud_rate=460800, threshold=0.35):
    print(f"Memuat model dari '{model_path}'...")
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={'fomo_loss': fomo_loss})
    except Exception as e:
        print(f"Gagal memuat model: {e}")
        return

    # WARMUP & OPTIMASI TF INFERENCE (Menggantikan model.predict yang lambat)
    print("Mengoptimasi graph TensorFlow untuk respons cepat...")
    @tf.function(reduce_retracing=True)
    def predict_fast(x):
        return model(x, training=False)

    # Warmup kompilasi TF
    dummy_input = tf.zeros((1, 96, 96, 3), dtype=tf.float32)
    _ = predict_fast(dummy_input)

    # Inisialisasi Serial
    try:
        ser = serial.Serial()
        ser.port = port
        ser.baudrate = baud_rate
        ser.timeout = 1
        ser.dtr = False
        ser.rts = False
        ser.open()
        ser.reset_input_buffer()
        print(f"[INFO] Terhubung ke ESP32-S3 pada {port} @ {baud_rate} baud.")
    except Exception as e:
        print(f"[ERROR] Gagal membuka port serial {port}: {e}")
        return

    grid_h, grid_w = 12, 12
    smooth_x = SmoothValue(alpha=0.30)
    smooth_y = SmoothValue(alpha=0.30)
    
    last_beep_time = 0
    fps_start_time = time.time()
    frame_count = 0
    fps = 0

    print("\n--- Testing FOMO Real-Time (Tekan 'q' untuk keluar) ---")

    while True:
        try:
            # Sync & buang frame lama
            sync_to_magic(ser)

            # Baca ukuran & data JPEG
            len_bytes = read_exact(ser, 4)
            frame_len = int.from_bytes(len_bytes, byteorder="little")

            if frame_len <= 0 or frame_len > 300_000:
                continue

            jpeg_bytes = read_exact(ser, frame_len)
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

        except (serial.SerialException, serial.SerialTimeoutException):
            continue

        frame_h, frame_w, _ = frame.shape
        screen_center_x = int(frame_w / 2)
        screen_center_y = int(frame_h / 2)

        # Preprocessing gambar (96x96 RGB)
        resized_frame = cv2.resize(frame, (96, 96))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        input_tensor = tf.keras.applications.mobilenet_v2.preprocess_input(
            np.expand_dims(rgb_frame, axis=0).astype(np.float32)
        )

        # PREDIKSI CEPAT (Menggunakan predict_fast alih-alih model.predict)
        logits = predict_fast(input_tensor)
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
        
        object_probs = probabilities[:, :, 0]
        max_prob = np.max(object_probs)

        current_time = time.time()

        if max_prob > threshold:
            gy, gx = np.unravel_index(np.argmax(object_probs), object_probs.shape)

            raw_target_x = (gx + 0.5) * (frame_w / grid_w)
            raw_target_y = (gy + 0.5) * (frame_h / grid_h)

            target_x = int(smooth_x.update(raw_target_x))
            target_y = int(smooth_y.update(raw_target_y))

            box_w = int(frame_w / grid_w) * 2
            box_h = int(frame_h / grid_h) * 2
            bx1 = target_x - int(box_w / 2)
            by1 = target_y - int(box_h / 2)

            distance_px = math.sqrt((target_x - screen_center_x)**2 + (target_y - screen_center_y)**2)
            effective_max_dist = screen_center_x * 0.9 
            norm_distance = min(distance_px / effective_max_dist, 1.0)
            curved_distance = norm_distance ** 2.2

            min_delay, max_delay = 0.03, 0.45            
            max_delay = 0.40
            beep_delay = min_delay + (curved_distance * (max_delay - min_delay))

            is_centered = (bx1 <= screen_center_x <= bx1 + box_w) and (by1 <= screen_center_y <= by1 + box_h)
            if is_centered:
                freq = 2500  # Bullseye (Nada melengking)
            else:
                # 700 Hz (jauh) -> 1800 Hz (mendekati pusat)
                freq = int(1800 - (curved_distance * (1800 - 700)))

            # BEEP ASYNCHRONOUS (Bebas Freeze)
            if (current_time - last_beep_time) >= beep_delay:
                duration = 15 if is_centered else 25
                play_beep_async(freq, duration)
                last_beep_time = current_time

            # Visualisasi
            cv2.rectangle(frame, (bx1, by1), (bx1 + box_w, by1 + box_h), (0, 255, 0), 2)
            cv2.circle(frame, (target_x, target_y), 6, (0, 255, 0), -1)
            cv2.circle(frame, (screen_center_x, screen_center_y), 8, (0, 0, 255), 2)
            cv2.line(frame, (screen_center_x, screen_center_y), (target_x, target_y), (255, 255, 0), 2)

            status = "TERKUNCI / BULLSEYE!" if is_centered else "Mengarahkan..."
            cv2.putText(frame, f"Status: {status}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Conf: {max_prob:.2f} | Jarak: {int(distance_px)}px", 
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        else:
            cv2.circle(frame, (screen_center_x, screen_center_y), 8, (0, 0, 255), 2)
            cv2.putText(frame, "Mencari Target...", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Hitung & Tampilkan FPS
        frame_count += 1
        if (current_time - fps_start_time) >= 1.0:
            fps = frame_count
            frame_count = 0
            fps_start_time = current_time

        cv2.putText(frame, f"FPS: {fps}", (frame_w - 110, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("ESP32-S3 FOMO Fast Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    ser.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_trained_fomo_serial(
        model_path="fomo_pure_face_esp32.h5", 
        port="COM6", 
        baud_rate=460800, 
        threshold=0.35
    )