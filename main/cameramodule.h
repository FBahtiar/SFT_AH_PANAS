#ifndef CAMERA_MODULE_H
#define CAMERA_MODULE_H

#include <Arduino.h>
#include "esp_camera.h"

// ❌ TIDAK ADA include SFT_inferencing.h di sini!

struct DetectionResult {
    bool        found;
    const char* label;
    float       confidence;
    int         x, y, width, height;  // bounding box
    float       dx, dy;               // offset dari pusat gambar
    float       dist;                 // jarak dari pusat (piksel)
    int         img_w, img_h;
};

bool ei_camera_init(void);
void ei_camera_deinit(void);
bool camera_run_detection(DetectionResult *out);

#endif