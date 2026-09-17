#include "cameramodule.h"
#include "config.h"

// ================================================================
//  SATU-SATUNYA file yang boleh meng-include Edge Impulse!
// ================================================================
#include <SFT_inferencing.h>
#include "edge-impulse-sdk/dsp/image/image.hpp"
// ================================================================

uint8_t *snapshot_buf = nullptr;
static bool is_initialised = false;

static camera_config_t camera_config = {
    .pin_pwdn = PWDN_GPIO_NUM,
    .pin_reset = RESET_GPIO_NUM,
    .pin_xclk = XCLK_GPIO_NUM,
    .pin_sscb_sda = SIOD_GPIO_NUM,
    .pin_sscb_scl = SIOC_GPIO_NUM,
    .pin_d7 = Y9_GPIO_NUM,
    .pin_d6 = Y8_GPIO_NUM,
    .pin_d5 = Y7_GPIO_NUM,
    .pin_d4 = Y6_GPIO_NUM,
    .pin_d3 = Y5_GPIO_NUM,
    .pin_d2 = Y4_GPIO_NUM,
    .pin_d1 = Y3_GPIO_NUM,
    .pin_d0 = Y2_GPIO_NUM,
    .pin_vsync = VSYNC_GPIO_NUM,
    .pin_href = HREF_GPIO_NUM,
    .pin_pclk = PCLK_GPIO_NUM,
    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,
    .pixel_format = PIXFORMAT_JPEG,
    .frame_size = FRAMESIZE_QVGA,
    .jpeg_quality = 12,
    .fb_count = 1,
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
};

bool ei_camera_init(void) {
    if (is_initialised) return true;

    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed with error 0x%x\n", err);
        return false;
    }

    sensor_t * s = esp_camera_sensor_get();
    if (s->id.PID == OV2640_PID) {
        s->set_vflip(s, 1);
        s->set_hmirror(s, 0);
    }

    is_initialised = true;
    return true;
}

void ei_camera_deinit(void) {
    esp_err_t err = esp_camera_deinit();
    if (err != ESP_OK) {
        Serial.printf("Camera deinit failed\n");
        return;
    }
    is_initialised = false;
}

bool ei_camera_capture(uint32_t img_width, uint32_t img_height, uint8_t *out_buf) {
    bool do_resize = false;

    if (!is_initialised) {
        ei_printf("ERR: Camera is not initialized\r\n");
        return false;
    }

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        ei_printf("Camera capture failed\n");
        return false;
    }

    bool converted = fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, snapshot_buf);
    esp_camera_fb_return(fb);

    if (!converted) {
        ei_printf("Conversion failed\n");
        return false;
    }

    if ((img_width != EI_CAMERA_RAW_FRAME_BUFFER_COLS) || (img_height != EI_CAMERA_RAW_FRAME_BUFFER_ROWS)) {
        do_resize = true;
    }

    if (do_resize) {
        ei::image::processing::crop_and_interpolate_rgb888(
            out_buf,
            EI_CAMERA_RAW_FRAME_BUFFER_COLS,
            EI_CAMERA_RAW_FRAME_BUFFER_ROWS,
            out_buf,
            img_width,
            img_height);
    }

    return true;
}

int ei_camera_get_data(size_t offset, size_t length, float *out_ptr) {
    size_t pixel_ix = offset * 3;
    size_t pixels_left = length;
    size_t out_ptr_ix = 0;

    while (pixels_left != 0) {
        out_ptr[out_ptr_ix] = (snapshot_buf[pixel_ix + 2] << 16) + (snapshot_buf[pixel_ix + 1] << 8) + snapshot_buf[pixel_ix];
        out_ptr_ix++;
        pixel_ix += 3;
        pixels_left--;
    }
    return 0;
}

// Internal: cari bounding box dengan confidence tertinggi >= CONF_THRESHOLD
static void find_best_target(ei_impulse_result_t *result, DetectionResult *out) {
    out->found = false;
    out->confidence = 0.0f;
    out->x = 0;  out->y = 0;
    out->width = 0;  out->height = 0;

    for (uint32_t i = 0; i < result->bounding_boxes_count; i++) {
        ei_impulse_result_bounding_box_t bb = result->bounding_boxes[i];
        if (bb.value == 0 || bb.value < CONF_THRESHOLD) continue;

        ei_printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                  bb.label, bb.value, bb.x, bb.y, bb.width, bb.height);

        if (bb.value > out->confidence) {
            out->found      = true;
            out->label      = bb.label;
            out->confidence = bb.value;
            out->x = bb.x;      out->y = bb.y;
            out->width = bb.width;  out->height = bb.height;
        }
    }
}

bool camera_run_detection(DetectionResult *out) {
    snapshot_buf = (uint8_t*)malloc(EI_CAMERA_RAW_FRAME_BUFFER_COLS *
                                    EI_CAMERA_RAW_FRAME_BUFFER_ROWS *
                                    EI_CAMERA_FRAME_BYTE_SIZE);
    if (snapshot_buf == nullptr) {
        ei_printf("ERR: Failed to allocate snapshot buffer!\n");
        return false;
    }

    ei::signal_t signal;
    signal.total_length = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT;
    signal.get_data = &ei_camera_get_data;

    if (!ei_camera_capture((size_t)EI_CLASSIFIER_INPUT_WIDTH,
                           (size_t)EI_CLASSIFIER_INPUT_HEIGHT, snapshot_buf)) {
        ei_printf("Failed to capture image\r\n");
        free(snapshot_buf);
        snapshot_buf = nullptr;
        return false;
    }

    ei_impulse_result_t result = { 0 };
    EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);

    free(snapshot_buf);
    snapshot_buf = nullptr;

    if (err != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", err);
        return false;
    }

    ei_printf("Predictions (DSP: %d ms., Classification: %d ms.): \n",
              result.timing.dsp, result.timing.classification);

    find_best_target(&result, out);

    out->img_w = EI_CLASSIFIER_INPUT_WIDTH;
    out->img_h = EI_CLASSIFIER_INPUT_HEIGHT;

    if (out->found) {
        float tgt_cx = out->x + out->width  / 2.0f;
        float tgt_cy = out->y + out->height / 2.0f;
        out->dx   = tgt_cx - (out->img_w / 2.0f);
        out->dy   = tgt_cy - (out->img_h / 2.0f);
        out->dist = sqrtf(out->dx * out->dx + out->dy * out->dy);
    }
    return true;
}