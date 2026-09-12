#include <Arduino.h>
#include "driver/i2s.h"
#include "esp_camera.h"

// ================= TFLM =================
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"   // WAJIB: array g_model dengan __attribute__((aligned(16)))

// =====================================================================
// KONFIGURASI
// =====================================================================
const int kTensorArenaSize = 400 * 1024;   // arena di PSRAM 8MB

// --- Saklar normalisasi input (sesuai cara model dilatih) ---
// 1 = (px/127.5 - 1) -> rentang [-1,1]  <- BENAR untuk modelmu
//     (MobileNetV2 preprocess_input, dibuktikan dari zp=95)
// 0 = px/255 -> rentang [0,1]
#define INPUT_NORM_M1_1   1

// --- Saklar urutan byte RGB565 (ganti ke 1 kalau R/B tertukar) ---
#define RGB565_SWAP_BYTES 0

// --- Skema output model custom (dari kode training Python) ---
// Kanal 0 = TARGET, kanal 1 = BACKGROUND, loss = softmax 2 kanal.
// p(target) per sel = sigmoid(l0 - l1)
// DO NOT change these unless you retrain with a different scheme.
#define FOMO_CH_TARGET     0
#define FOMO_CH_BACKGROUND 1

#define THRESHOLD_HIGH  0.70f   // skor > ini  -> beep tinggi
#define THRESHOLD_LOW   0.40f   // skor > ini  -> beep rendah

// =====================================================================
// I2S PCM5102A
// =====================================================================
#define I2S_NUM         I2S_NUM_0
#define I2S_BCK_PIN     1    // BCK  -> GPIO 1
#define I2S_LRCK_PIN    2    // LRCK -> GPIO 2
#define I2S_DATA_PIN    3    // DIN  -> GPIO 3
#define SAMPLE_RATE     16000

// =====================================================================
// OBJEK TFLITE
// =====================================================================
static uint8_t* tensor_arena = nullptr;
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

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
// AUDIO: I2S TX ke PCM5102A
// =====================================================================
void setup_i2s_pcm5102() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = 128,
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

void play_pcm_beep(int frequency_hz, int duration_ms) {
    static int16_t sample_buffer[256];
    size_t bytes_written;
    float phase_step = (2.0f * PI * frequency_hz) / SAMPLE_RATE;
    float phase = 0.0f;
    long remaining = ((long)SAMPLE_RATE * duration_ms) / 1000;

    while (remaining > 0) {
        int chunk = (remaining > 128) ? 128 : (int)remaining;
        for (int i = 0; i < chunk; i++) {
            int16_t s = (int16_t)(sinf(phase) * 10000.0f);
            sample_buffer[i * 2]     = s;
            sample_buffer[i * 2 + 1] = s;
            phase += phase_step;
            if (phase >= 2.0f * PI) phase -= 2.0f * PI;
        }
        i2s_write(I2S_NUM, sample_buffer, chunk * 2 * sizeof(int16_t),
                  &bytes_written, portMAX_DELAY);
        remaining -= chunk;
    }
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
    config.frame_size   = FRAMESIZE_96X96;    // sama dengan input model
    config.pixel_format = PIXFORMAT_RGB565;   // model butuh 3 channel
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
// SETUP
// =====================================================================
void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n=== ESP32-S3 FOMO (TFLM + model_data.h) ===");

    Serial.printf("Total PSRAM: %d bytes\n", ESP.getPsramSize());
    if (ESP.getPsramSize() == 0) fatal("PSRAM tidak terdeteksi!");

    // ---- A. AUDIO PALING AWAL (sebelum beep apa pun!) ----
    setup_i2s_pcm5102();

    // Nada tes panjang untuk pengukuran multimeter
    Serial.println("[TEST] Nada tes 3 detik... ukur pin BCK/LRCK SEKARANG");
    play_pcm_beep(1000, 3000);

    Serial.println("[OK] I2S siap");
    delay(100);

    // ---- B. KAMERA ----
    setup_camera();

    // ---- C. Tensor arena di PSRAM ----
    tensor_arena = (uint8_t*)heap_caps_aligned_alloc(16, kTensorArenaSize,
                                                     MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!tensor_arena) fatal("Gagal alokasi arena PSRAM");

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

    // ---- G. Beep sukses booting ----
    play_pcm_beep(1000, 100); delay(50);
    play_pcm_beep(2000, 150);
    Serial.println("=== Sistem siap ===");
}

// =====================================================================
// LOOP: kamera RGB565 -> preprocess -> inferensi -> parsing FOMO -> beep
// =====================================================================
void loop() {
    // 1. Ambil frame
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("[ERROR] Capture gagal");
        delay(100);
        return;
    }

    // 2. Validasi: RGB565 = 2 byte/piksel, model butuh npixels*3 byte
    size_t npixels = fb->len / 2;
    if (npixels * 3 != (size_t)input_tensor->bytes) {
        Serial.printf("[ERROR] Frame %u byte tidak cocok dengan input model %u\n",
                      (unsigned)fb->len, (unsigned)input_tensor->bytes);
        esp_camera_fb_return(fb);
        delay(500);
        return;
    }

    // 3. Preprocessing: RGB565 -> R,G,B -> normalisasi -> quantize int8
    float in_scale = input_tensor->params.scale;
    int   in_zp    = input_tensor->params.zero_point;
    int8_t* in8 = input_tensor->data.int8;

    for (size_t p = 0; p < npixels; p++) {
        uint16_t c;
#if RGB565_SWAP_BYTES
        c = ((uint16_t)fb->buf[p*2+1] << 8) | fb->buf[p*2];
#else
        c = ((uint16_t)fb->buf[p*2]   << 8) | fb->buf[p*2+1];
#endif
        uint8_t r = (c >> 8) & 0xF8;  r |= r >> 5;   // 5 -> 8 bit
        uint8_t g = (c >> 3) & 0xFC;  g |= g >> 6;   // 6 -> 8 bit
        uint8_t b = (c << 3) & 0xF8;  b |= b >> 5;

        float rf, gf, bf;
#if INPUT_NORM_M1_1
        rf = r / 127.5f - 1.0f;
        gf = g / 127.5f - 1.0f;
        bf = b / 127.5f - 1.0f;
#else
        rf = r / 255.0f;  gf = g / 255.0f;  bf = b / 255.0f;
#endif
        in8[p*3+0] = quantize_val(rf, in_scale, in_zp);
        in8[p*3+1] = quantize_val(gf, in_scale, in_zp);
        in8[p*3+2] = quantize_val(bf, in_scale, in_zp);
    }

    // 4. Inferensi
    unsigned long t0 = millis();
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("[ERROR] Invoke gagal");
        esp_camera_fb_return(fb);
        return;
    }
    unsigned long inf_ms = millis() - t0;

    // 5. Parsing output FOMO custom (skema dari kode training):
    //    (1, grid, grid, 2) -> kanal 0 = TARGET, kanal 1 = BACKGROUND.
    //    Loss softmax 2 kanal: p(target) = sigmoid(l0 - l1)
    int per_cell = output_tensor->dims->data[output_tensor->dims->size - 1]; // 2
    int cells    = output_tensor->bytes / per_cell;                          // 144
    int grid_w   = output_tensor->dims->data[2];                             // 12
    float out_scale = output_tensor->params.scale;
    int   out_zp    = output_tensor->params.zero_point;
    int8_t* out8 = output_tensor->data.int8;

    float max_prob = 0.0f;
    int   max_cell = -1;
    for (int i = 0; i < cells; i++) {
        float l0 = (out8[i * per_cell + FOMO_CH_TARGET]     - out_zp) * out_scale;
        float l1 = (out8[i * per_cell + FOMO_CH_BACKGROUND] - out_zp) * out_scale;
        float p_target = 1.0f / (1.0f + expf(l1 - l0));   // softmax 2 kelas
        if (p_target > max_prob) { max_prob = p_target; max_cell = i; }
    }

    esp_camera_fb_return(fb);

    // 6. Debug: skor + posisi sel grid terbaik (12x12)
    if (max_cell >= 0) {
        int gx = max_cell % grid_w;
        int gy = max_cell / grid_w;
        Serial.printf("Inferensi: %lu ms | skor: %.2f | sel: (%d,%d) | l0=%.1f l1=%.1f\n",
                      inf_ms, max_prob, gx, gy,
                      (out8[max_cell*per_cell + FOMO_CH_TARGET]     - out_zp) * out_scale,
                      (out8[max_cell*per_cell + FOMO_CH_BACKGROUND] - out_zp) * out_scale);
    }

    // 7. Feedback audio berdasarkan skor
    if (max_prob > THRESHOLD_HIGH)      play_pcm_beep(1800, 80);
    else if (max_prob > THRESHOLD_LOW)  play_pcm_beep(1000, 50);

    delay(50);
}