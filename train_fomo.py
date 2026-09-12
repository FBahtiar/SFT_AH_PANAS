import os
import time
import serial
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# =====================================================================
# GLOBAL VARIABLES FOR MOUSE CALLBACK
# =====================================================================
click_x, click_y = None, None
clicked = False

MAGIC = bytes([0xFF, 0xFE, 0xFF, 0xFE])

def mouse_callback(event, x, y, flags, param):
    global click_x, click_y, clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        click_x, click_y = x, y
        clicked = True

# =====================================================================
# SERIAL HELPER FUNCTIONS
# =====================================================================
def read_exact(ser, n):
    """Membaca tepat n byte dari stream serial."""
    buf = bytearray()
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            raise serial.SerialTimeoutException("Timeout saat membaca data serial")
        buf.extend(chunk)
    return bytes(buf)

def sync_to_magic(ser):
    """Mencari 4 byte magic marker (0xFF, 0xFE, 0xFF, 0xFE) di stream serial."""
    window = bytearray(4)
    while True:
        b = ser.read(1)
        if not b:
            continue
        window = window[1:] + bytearray(b)
        if bytes(window) == MAGIC:
            return

# =====================================================================
# 1. GENERATE FOMO HEATMAP LABEL
# =====================================================================
def create_fomo_heatmap(cx, cy, img_w, img_h, grid_size=(12, 12), num_classes=1):
    grid_h, grid_w = grid_size
    num_classes_with_bg = num_classes + 1
    
    label_map = np.zeros((grid_h, grid_w, num_classes_with_bg), dtype=np.float32)
    label_map[:, :, 1] = 1.0  # Index 1: Background

    if cx is not None and cy is not None:
        gx = int(cx / img_w * grid_w)
        gy = int(cy / img_h * grid_h)
        
        gx = min(max(gx, 0), grid_w - 1)
        gy = min(max(gy, 0), grid_h - 1)
        
        label_map[gy, gx, 0] = 1.0  # Index 0: Target
        label_map[gy, gx, 1] = 0.0  # Hapus label background

    return label_map

# =====================================================================
# 2. PREPROCESSING & DATA AUGMENTATION
# =====================================================================
def augment_fomo_sample(img_rgb, cx, cy, img_w, img_h):
    augmented_imgs = []
    augmented_labels = []

    # 1. Sampel Asli
    norm_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_rgb.copy().astype(np.float32))
    label = create_fomo_heatmap(cx, cy, img_w, img_h)
    augmented_imgs.append(norm_img)
    augmented_labels.append(label)

    # 2. Augmentasi Flip Horizontal
    flipped_img = cv2.flip(img_rgb, 1)
    flipped_cx = img_w - cx if cx is not None else None
    norm_flipped = tf.keras.applications.mobilenet_v2.preprocess_input(flipped_img.astype(np.float32))
    label_flipped = create_fomo_heatmap(flipped_cx, cy, img_w, img_h)
    augmented_imgs.append(norm_flipped)
    augmented_labels.append(label_flipped)

    # 3. Augmentasi Kecerahan Rendah
    dark_img = cv2.convertScaleAbs(img_rgb, alpha=0.7, beta=-10)
    norm_dark = tf.keras.applications.mobilenet_v2.preprocess_input(dark_img.astype(np.float32))
    augmented_imgs.append(norm_dark)
    augmented_labels.append(label)

    # 4. Augmentasi Kecerahan Tinggi
    bright_img = cv2.convertScaleAbs(img_rgb, alpha=1.2, beta=15)
    norm_bright = tf.keras.applications.mobilenet_v2.preprocess_input(bright_img.astype(np.float32))
    augmented_imgs.append(norm_bright)
    augmented_labels.append(label)

    return augmented_imgs, augmented_labels

# =====================================================================
# 3. DATA COLLECTION (VIA ESP32-S3 USB SERIAL)
# =====================================================================
def collect_data_pure_fomo_serial(num_samples=30, port="COM6", baud_rate=460800):
    global click_x, click_y, clicked
    
    # Reset variabel koordinat & status klik
    click_x, click_y = None, None
    clicked = False
    
    try:
        ser = serial.Serial()
        ser.port = port
        ser.baudrate = baud_rate
        ser.timeout = 1
        ser.dtr = False
        ser.rts = False
        ser.open()
        ser.reset_input_buffer()
        print(f"\n[INFO] Terhubung ke ESP32-S3 via Serial pada {port} @ {baud_rate} baud.")
    except Exception as e:
        print(f"\n[ERROR] Gagal membuka port serial {port}: {e}")
        return np.array([]), np.array([])

    cv2.namedWindow("ESP32-S3 FOMO Data Collection")
    cv2.setMouseCallback("ESP32-S3 FOMO Data Collection", mouse_callback)

    X_data = []
    Y_data = []
    count = 0

    print("=======================================================")
    print(f" TERHUBUNG KE ESP32-S3 SERIAL ({port})")
    print(" 1. Arahkan kamera ESP32-S3 ke target/wajah.")
    print(" 2. KLIK KIRI mouse tepat pada target untuk LANGSUNG menyimpan sampel.")
    print(" 3. Tekan 'q' untuk berhenti lebih awal.")
    print("=======================================================\n")

    while count < num_samples:
        try:
            sync_to_magic(ser)
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
            print("[WARNING] Terjadi kesalahan baca serial, mencoba sinkronisasi ulang...")
            continue

        frame_h, frame_w, _ = frame.shape
        display_frame = frame.copy()

        # LOGIKA OTOMATIS: Dijalankan langsung saat mouse diklik kiri
        if clicked:
            if click_x is not None and click_y is not None:
                resized_rgb = cv2.cvtColor(cv2.resize(frame, (96, 96)), cv2.COLOR_BGR2RGB)
                aug_imgs, aug_labels = augment_fomo_sample(resized_rgb, click_x, click_y, frame_w, frame_h)

                X_data.extend(aug_imgs)
                Y_data.extend(aug_labels)
                count += 1
                print(f"Sampel Utama [{count}/{num_samples}] + Augmentasi berhasil disimpan via Klik!")
            
            # Reset flag clicked agar tidak menyimpan terus-menerus di frame berikutnya
            clicked = False

        # Visualisasi titik klik terakhir di layar
        if click_x is not None and click_y is not None:
            cv2.circle(display_frame, (click_x, click_y), 8, (0, 255, 0), -1)
            cv2.putText(display_frame, f"Target Terakhir: ({click_x}, {click_y})", 
                        (20, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(display_frame, f"Sampel Terkumpul: {count}/{num_samples}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display_frame, "KLIK KIRI pada Target untuk Simpan", (20, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("ESP32-S3 FOMO Data Collection", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    ser.close()
    cv2.destroyAllWindows()
    return np.array(X_data), np.array(Y_data)

# =====================================================================
# 4. ARSITEKTUR FOMO & TRAINING
# =====================================================================
def build_fomo_model(input_shape=(96, 96, 3), num_classes=1):
    num_classes_with_bg = num_classes + 1
    
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=0.35,
        include_top=False,
        weights='imagenet'
    )
    
    for layer in backbone.layers:
        layer.trainable = False

    fomo_output = backbone.get_layer('block_6_expand_relu').output
    
    x = layers.Conv2D(32, kernel_size=(1, 1), padding='same', activation='relu')(fomo_output)
    logits = layers.Conv2D(num_classes_with_bg, kernel_size=(1, 1), padding='same', activation=None)(x)
    
    return tf.keras.Model(inputs=backbone.input, outputs=logits, name="FOMO_Pure_Model")

def fomo_loss(y_true, y_pred):
    y_pred_softmax = tf.nn.softmax(y_pred, axis=-1)
    return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred_softmax))

# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == '__main__':
    PORT = "COM6"         # Port ESP32-S3
    BAUD_RATE = 460800    # Baud rate serial

    # 1. Kumpulkan Sampel Data dari ESP32-S3 via Serial Stream
    X_train, Y_train = collect_data_pure_fomo_serial(num_samples=100, port=PORT, baud_rate=BAUD_RATE)
    
    if len(X_train) > 0:
        print(f"\nTotal Dataset Training (Termasuk Augmentasi): {X_train.shape[0]} Sampel")
        print("Membuat dan Melatih Model FOMO...")
        
        model = build_fomo_model(input_shape=(96, 96, 3), num_classes=1)
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss=fomo_loss,
            metrics=['accuracy']
        )

        model.fit(
            X_train, Y_train, 
            batch_size=8, 
            epochs=100, 
            shuffle=True,
            verbose=1
        )

        print("\n=======================================================")
        print(" PENYIMPANAN MODEL (.h5 & .tflite INT8)")
        print("=======================================================")

        # A. SIMPAN KE FORMAT .H5 (Keras Model)
        h5_path = "fomo_pure_face_esp32.h5"
        model.save(h5_path)
        print(f"[SUCCESS] Model H5 disimpan ke: {h5_path}")

        # B. KONVERSI & SIMPAN KE FORMAT .TFLITE (Full INT8 Quantization)
        def representative_data_gen():
            """ Generator data kalibrasi kuantisasi INT8 dari dataset training """
            for i in range(min(100, len(X_train))):
                sample = np.expand_dims(X_train[i], axis=0).astype(np.float32)
                yield [sample]

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        
        # Enforce full INT8 quantization untuk akselerasi ESP-NN di ESP32-S3
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        try:
            tflite_quant_model = converter.convert()
            tflite_path = "fomo_pure_face_esp32.tflite"
            
            with open(tflite_path, "wb") as f:
                f.write(tflite_quant_model)
                
            print(f"[SUCCESS] Model TFLite (INT8) disimpan ke: {tflite_path}")
        except Exception as e:
            print(f"[ERROR] Gagal mengonversi ke TFLite INT8: {e}")
            # Fallback ke Float32 TFLite jika kuantisasi INT8 gagal
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_float_model = converter.convert()
            tflite_path = "fomo_pure_face_esp32_float.tflite"
            with open(tflite_path, "wb") as f:
                f.write(tflite_float_model)
            print(f"[WARNING] Disimpan sebagai Float32 TFLite ke: {tflite_path}")