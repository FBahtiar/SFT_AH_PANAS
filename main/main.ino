#include "config.h"
#include "audiomodule.h"
#include "cameramodule.h"
// ❌ targetmodule.h DIHAPUS — logikanya sudah pindah ke cameramodule.cpp

unsigned long last_beep_time = 0;

void setup() {
    Serial.begin(115200);
    pinMode(ONBOARD_LED, OUTPUT);
    digitalWrite(ONBOARD_LED, LOW);

    // Inisialisasi Audio DAC PCM5102
    setup_i2s_pcm5102();

    while (!Serial);
    Serial.println("Edge Impulse Inferencing Demo - ESP32-S3 WROOM + PCM5102 Audio");

    if (!ei_camera_init()) {
        Serial.println("Failed to initialize Camera!");
    } else {
        Serial.println("Camera initialized");
    }

    // Nada panggil saat startup berhasil
    play_tone(1000, 100);
    delay(50);
    play_tone(1500, 150);

    Serial.println("\nStarting continuous inference in 2 seconds...");
    delay(2000);
}

void loop() {
    delay(5);

    DetectionResult det;
    if (!camera_run_detection(&det)) return;

    if (!det.found) {
        digitalWrite(ONBOARD_LED, LOW);
        Serial.printf("Tidak ada target (confidence < %.2f)\r\n", CONF_THRESHOLD);
        return;
    }

    digitalWrite(ONBOARD_LED, HIGH);

    float img_cx = det.img_w / 2.0f;
    float img_cy = det.img_h / 2.0f;
    float tgt_cx = det.x + det.width  / 2.0f;
    float tgt_cy = det.y + det.height / 2.0f;

    Serial.println("--------------------------------");
    Serial.printf("Target       : %s (%.2f)\r\n", det.label, det.confidence);
    Serial.printf("Pusat gambar : (%.1f, %.1f)\r\n", img_cx, img_cy);
    Serial.printf("Pusat target : (%.1f, %.1f)\r\n", tgt_cx, tgt_cy);
    Serial.printf("Offset       : dx=%.1f, dy=%.1f\r\n", det.dx, det.dy);
    Serial.printf("Jarak pusat  : %.2f piksel\r\n", det.dist);

    if (det.dx > 5)       Serial.println("Arah X : target di KANAN");
    else if (det.dx < -5) Serial.println("Arah X : target di KIRI");
    else                  Serial.println("Arah X : TENGAH");

    if (det.dy > 5)       Serial.println("Arah Y : target di BAWAH");
    else if (det.dy < -5) Serial.println("Arah Y : target di ATAS");
    else                  Serial.println("Arah Y : TENGAH");
    Serial.println("--------------------------------");

    // ---- Umpan balik suara dinamis ----
    unsigned long current_time = millis();
    float max_dx = det.img_w / 2.0f;
    float pan = det.dx / max_dx;

    float norm_dist = det.dist / 100.0f;
    if (norm_dist > 1.0f) norm_dist = 1.0f;
    float curved_dist = norm_dist * norm_dist;

    unsigned long beep_delay = (unsigned long)(10 + (curved_dist * (500 - 10)));
    int freq;
    int duration;

    if (det.dist <= 5.0f) {
        freq = 2800;
        duration = 10;
        beep_delay = 10;
        pan = 0.0f;
    } else {
        freq = (int)(2000.0f - (curved_dist * (2000.0f - 600.0f)));
        duration = 20;
    }

    if ((current_time - last_beep_time) >= beep_delay) {
        play_tone(freq, duration, pan);
        last_beep_time = current_time;
    }
}