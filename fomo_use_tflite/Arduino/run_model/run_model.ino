/*********************************************************************
 * ESP32-S3 + OV2640 (FOMO/TFLM) + PCM5102 (I2S) — VERSI NON-BLOCKING
 *
 * Perubahan dari versi lama:
 *  1. Beep via FreeRTOS task di core 0 -> loop() TIDAK pernah menunggu
 *  2. Tensor arena: coba SRAM internal dulu (cepat), fallback PSRAM
 *  3. Preprocessing RGB565->int8 pakai LUT (tanpa float per piksel)
 *  4. Tanpa delay(50), Serial print dibatasi 2x/detik
 *  5. Parsing output pakai selisih int8 (tanpa expf per sel)
 *
 * WIRING PCM5102 : BCK=GPIO1, LRCK=GPIO2, DIN=GPIO3, VIN=3V3, GND=GND
 * Pad PCM5102    : XSMT=H, SCK=L
 *********************************************************************/

#include <Arduino.h>
#include "driver/i2s.h"
#include "esp_camera.h"

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"   // array g_model, aligned(16)

// =====================================================================
// KONFIGURASI
// =====================================================================
// Setelah boot, lihat "Arena terpakai: XXX bytes" di Serial.
// Turunkan nilai ini mendekati angka tsb agar BISA muat di SRAM internal
// (SRAM internal hanya sanggup ~200-280 KB; sisanya otomatis ke PSRAM).
const int kTensorArenaSize = 160 * 1024;

// Saklar normalisasi input (sesuai training): 1 = [-1,1], 0 = [0,1]
#define INPUT_NORM_M1_1   1
// Saklar urutan byte RGB565 (1 kalau R/B tertukar)
#define RGB565_SWAP_BYTES 0

// Skema output model custom: kanal 0 = TARGET, kanal 1 = BACKGROUND
#define FOMO_CH_TARGET     0
#define FOMO_CH_BACKGROUND 1

#define THRESHOLD_HIGH  0.70f   // skor > ini -> beep tinggi (1800 Hz)
#define THRESHOLD_LOW   0.40f   // skor > ini -> beep rendah (1000 Hz)

// Ritme beep (pola berdenyut). Set BEEP_GAP_MS = 0 untuk nada kontinu.
#define BEEP_ON_MS   80
#define BEEP_GAP_MS  120

// Debug Serial maksimal 2x per detik (hemat waktu loop)
#define DEBUG_INTERVAL_MS 500

// =====================================================================
// I2S PCM5102A
// =====================================================================
#define I2S_NUM         I2S_NUM_0
#define I2S_BCK_PIN     1    // BCK  -> GPIO 1
#define I2S_LRCK_PIN    2    // LRCK -> GPIO 2
#define I2S_DATA_PIN    3    // DIN  -> GPIO 3
#define SAMPLE_RATE     16000

// =====================================================================
// STATE AUDIO (dibagi antar task, akses atomik)
// =====================================================================
volatile bool     beep_active = false;
volatile uint32_t beep_end_ms = 0;
volatile float    beep_freq   = 1000.0f;

// =====================================================================
// OBJEK TFLM
// =====================================================================
static uint8_t* tensor_arena = nullptr;
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor  = nullptr;
TfLiteTensor* output_tensor = nullptr;

// LUT preprocessing (dihitung sekali saat boot)
static int8_t lutR[32], lutG[64], lutB[32];

// Threshold versi "delta logits" (dihitung sekali saat boot)
static float t_high_delta = 0.0f;
static float t_low_delta  = 0.0f;

// =====================================================================
// HELPER
// =====================================================================
void fatal(const char* msg) {
    Serial.printf("[FATAL] %s -> sistem berhenti, tekan RESET\n", msg);
    while (true) delay(1000);
}

static inline int8_t quantize_val(float v, float scale, int zp) {
    int q = (int)roundf(v / scale) + zp;
    return (int8_t)constrain(q, -128, 127);
}

// =====================================================================
// AUDIO: I2S TX ke PCM5102 + TASK FREERTOS (core 0)
// =====================================================================
void setup_i2s_pcm5102() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,      // buffer lega = audio anti patah-patah
        .dma_buf_len = 256,
        .use_apll = false
    };
    i2s_pin_config_t pin_config = {
        .mck_io_num = I2S_PIN_NO_CHANGE,
        .bck_io_num = I2S_BCK_PIN,
        .ws_io_num = I2S_LRCK_PIN,
        .data_out_num = I2S_DATA_PIN,
        .data_in_num = I2S_PIN_NO_CHANGE
    };
    esp_err_t err = i2s_driver_install(I2S_NUM, &i2s_config, 0, NULL);
    if (err != ESP_OK) {
        Serial.printf("[FATAL] i2s_driver_install gagal: 0x%x\n", err);
        while (true) delay(1000);
    }
    err = i2s_set_pin(I2S_NUM, &pin_config);
    if (err != ESP_OK) {
        Serial.printf("[FATAL] i2s_set_pin gagal: 0x%x\n", err);
        while (true) delay(1000);
    }
    i2s_zero_dma_buffer(I2S_NUM);
}

// Task audio: berjalan PERMANEN di core 0, mengisi buffer I2S terus-menerus
void audioTask(void *param) {
    const int N = 256;                       // 16 ms per buffer @ 16 kHz
    static int16_t buf[N * 2];               // stereo 16-bit
    float phase = 0.0f;

    for (;;) {
        bool active = beep_active && (millis() < beep_end_ms);
        if (beep_active && !active) beep_active = false;   // beep selesai

        float step = (2.0f * PI * beep_freq) / (float)SAMPLE_RATE;
        for (int i = 0; i < N; i++) {
            int16_t s = 0;
            if (active) {
                s = (int16_t)(sinf(phase) * 10000.0f);
                phase += step;
                if (phase >= 2.0f * PI) phase -= 2.0f * PI;
            }
            buf[i * 2]     = s;   // kiri
            buf[i * 2 + 1] = s;   // kanan
        }
        size_t w;
        i2s_write(I2S_NUM, buf, sizeof(buf), &w, portMAX_DELAY);
    }
}

// "Tombol" beep: selesai dalam mikrodetik, TIDAK menunggu apa pun
void trigger_beep(int frequency_hz, uint32_t duration_ms) {
    beep_freq   = (float)frequency_hz;
    beep_end_ms = millis() + duration_ms;
    beep_active = true;
}

// =====================================================================
// KAMERA OV2640
// =====================================================================
#define PWDN_GPIO_NUM  -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  15
#define SIOD_GPIO_NUM   4
#define SIOC_GPIO_NUM   5
#define Y9_GPIO_NUM    16
#define Y8_GPIO_NUM    17
#define Y7_GPIO_NUM    18
#define Y6_GPIO_NUM    12
#define Y5_GPIO_NUM    10
#define Y4_GPIO_NUM     8
#define Y3_GPIO_NUM     9
#define Y2_GPIO_NUM    11
#define VSYNC_GPIO_NUM  6
#define HREF_GPIO_NUM   7
#define PCLK_GPIO_NUM  13

void setup_camera() {
    camera_config_t config = {};
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;  config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;  config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;  config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;  config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href  = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn  = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.frame_size   = FRAMESIZE_96X96;
    config.pixel_format = PIXFORMAT_RGB565;
    config.grab_mode    = CAMERA_GRAB_LATEST;
    config.fb_location  = CAMERA_FB_IN_PSRAM;
    config.fb_count     = 2;
    config.jpeg_quality = 12;

    if (esp_camera_init(&config) != ESP_OK) {
        fatal("Kamera gagal init (cek PSRAM setting & kabel flat)");
    }
    Serial.println("[OK] Kamera aktif (RGB565)");
}

// =====================================================================
// LUT PREPROCESSING: hitung semua kemungkinan nilai SEKALI saat boot
// =====================================================================
void build_luts(float scale, int zp) {
    for (int v = 0; v < 32; v++) {              // komponen 5-bit (R dan B)
        uint8_t x8 = (v << 3) | (v >> 2);       // ekspansi 5 -> 8 bit
#if INPUT_NORM_M1_1
        float n = x8 / 127.5f - 1.0f;
#else
        float n = x8 / 255.0f;
#endif
        lutR[v] = quantize_val(n, scale, zp);
        lutB[v] = quantize_val(n, scale, zp);
    }
    for (int v = 0; v < 64; v++) {              // komponen 6-bit (G)
        uint8_t g8 = (v << 2) | (v >> 4);       // ekspansi 6 -> 8 bit
#if INPUT_NORM_M1_1
        float n = g8 / 127.5f - 1.0f;
#else
        float n = g8 / 255.0f;
#endif
        lutG[v] = quantize_val(n, scale, zp);
    }
}

// =====================================================================
// SETUP
// =====================================================================
void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n=== ESP32-S3 FOMO + PCM5102 (non-blocking) ===");

    Serial.printf("Total PSRAM: %d bytes\n", ESP.getPsramSize());
    if (ESP.getPsramSize() == 0) fatal("PSRAM tidak terdeteksi!");

    // ---- A. AUDIO PALING AWAL + mulai task audio di core 0 ----
    setup_i2s_pcm5102();
    xTaskCreatePinnedToCore(audioTask, "audio", 4096, NULL, 5, NULL, 0);

    Serial.println("[TEST] Nada tes 3 detik...");
    trigger_beep(1000, 3000);
    delay(3000);                 // hanya untuk tes boot (bisa dihapus nanti)
    Serial.println("[OK] I2S siap");

    // ---- B. KAMERA ----
    setup_camera();

    // ---- C. Tensor arena: SRAM internal dulu (CEPAT), fallback PSRAM ----
    tensor_arena = (uint8_t*)heap_caps_aligned_alloc(16, kTensorArenaSize,
                                                     MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (tensor_arena) {
        Serial.println("[OK] Arena di SRAM internal (cepat)");
    } else {
        Serial.println("[WARN] SRAM tidak cukup, arena ke PSRAM (lambat)");
        Serial.println("       -> Turunkan kTensorArenaSize agar muat SRAM internal");
        tensor_arena = (uint8_t*)heap_caps_aligned_alloc(16, kTensorArenaSize,
                                                         MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!tensor_arena) fatal("Gagal alokasi arena (SRAM & PSRAM)");
    }

    // ---- D. Muat model ----
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("Schema model=%d, library=%d\n",
                      model->version(), TFLITE_SCHEMA_VERSION);
        fatal("Versi schema tidak cocok");
    }

    // ---- E. Interpreter ----
    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) fatal("AllocateTensors gagal");

    input_tensor  = interpreter->input(0);
    output_tensor = interpreter->output(0);

    Serial.print("Input  shape: ");
    for (int i = 0; i < input_tensor->dims->size; i++)
        Serial.printf("%d ", input_tensor->dims->data[i]);
    Serial.printf("| scale=%.5f zp=%d\n",
                  input_tensor->params.scale, input_tensor->params.zero_point);
    Serial.printf("Arena terpakai: %u / %d bytes\n",
                  (unsigned)interpreter->arena_used_bytes(), kTensorArenaSize);

    // ---- F. Siapkan LUT + threshold (semua perhitungan berat di sini) ----
    build_luts(input_tensor->params.scale, input_tensor->params.zero_point);

    // p(target) > 0.70  <=>  (l0 - l1) > ln(0.7/0.3) = 0.8473
    // p(target) > 0.40  <=>  (l0 - l1) > ln(0.4/0.6) = -0.4055
    float out_scale = output_tensor->params.scale;
    t_high_delta = 0.8473f / out_scale;
    t_low_delta  = -0.4055f / out_scale;

    // ---- G. Beep sukses booting (non-blocking) ----
    trigger_beep(1000, 100);
    delay(150);                  // jeda antar beep (audio tetap jalan di task)
    trigger_beep(2000, 150);
    Serial.println("=== Sistem siap ===");
}

// =====================================================================
// LOOP: kamera -> LUT -> inferensi -> parsing -> beep (semua non-blocking)
// =====================================================================
void loop() {
    // 1. Ambil frame
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        delay(20);
        return;
    }

    // 2. Validasi ukuran frame
    size_t npixels = fb->len / 2;   // RGB565 = 2 byte/piksel
    if (npixels * 3 != (size_t)input_tensor->bytes) {
        Serial.printf("[ERROR] Frame %u byte tidak cocok dengan input model %u\n",
                      (unsigned)fb->len, (unsigned)input_tensor->bytes);
        esp_camera_fb_return(fb);
        delay(100);
        return;
    }

    // 3. Preprocessing dengan LUT (sangat cepat, tanpa float per piksel)
    int8_t* in8 = input_tensor->data.int8;
    for (size_t p = 0; p < npixels; p++) {
#if RGB565_SWAP_BYTES
        uint16_t c = ((uint16_t)fb->buf[p*2+1] << 8) | fb->buf[p*2];
#else
        uint16_t c = ((uint16_t)fb->buf[p*2]   << 8) | fb->buf[p*2+1];
#endif
        in8[p*3+0] = lutR[(c >> 11) & 0x1F];   // R (5 bit)
        in8[p*3+1] = lutG[(c >> 5)  & 0x3F];   // G (6 bit)
        in8[p*3+2] = lutB[ c        & 0x1F];   // B (5 bit)
    }

    // 4. Inferensi
    unsigned long t0 = millis();
    TfLiteStatus status = interpreter->Invoke();
    unsigned long inf_ms = millis() - t0;

    if (status != kTfLiteOk) {
        esp_camera_fb_return(fb);
        Serial.println("[ERROR] Invoke gagal");
        return;
    }

    // 5. Parsing output: cari selisih (l0-l1) MAKSIMUM dalam satuan int8
    //    (zero point saling menghilangkan, jadi cukup selisih mentah)
    int per_cell = output_tensor->dims->data[output_tensor->dims->size - 1]; // 2
    int cells    = output_tensor->bytes / per_cell;                          // 144
    int grid_w   = output_tensor->dims->data[2];                             // 12
    float out_scale = output_tensor->params.scale;
    int8_t* out8 = output_tensor->data.int8;

    int max_d = -32768;
    int max_cell = -1;
    for (int i = 0; i < cells; i++) {
        int d = (int)out8[i * per_cell + FOMO_CH_TARGET]
              - (int)out8[i * per_cell + FOMO_CH_BACKGROUND];
        if (d > max_d) { max_d = d; max_cell = i; }
    }

    esp_camera_fb_return(fb);

    // Skor asli hanya untuk debug/threshold tampilan
    float delta_f  = (float)max_d * out_scale;
    float max_prob = 1.0f / (1.0f + expf(-delta_f));

    // 6. DebugSerial: maksimal 2x per detik
    static uint32_t last_dbg = 0;
    if (max_cell >= 0 && (millis() - last_dbg >= DEBUG_INTERVAL_MS)) {
        int gx = max_cell % grid_w;
        int gy = max_cell / grid_w;
        Serial.printf("Inf: %lu ms | skor: %.2f | sel: (%d,%d)\n",
                      inf_ms, max_prob, gx, gy);
        last_dbg = millis();
    }

    // 7. Feedback audio berdenyut, NON-BLOCKING (0 ms menunggu)
    static uint32_t next_beep_ms = 0;
    uint32_t now = millis();
    if (now >= next_beep_ms) {
        if (delta_f > t_high_delta) {
            trigger_beep(1800, BEEP_ON_MS);
            next_beep_ms = now + BEEP_ON_MS + BEEP_GAP_MS;
        } else if (delta_f > t_low_delta) {
            trigger_beep(1000, BEEP_ON_MS);
            next_beep_ms = now + BEEP_ON_MS + BEEP_GAP_MS;
        }
    }
    // Tidak ada delay() di sini — loop secepat mungkin
}