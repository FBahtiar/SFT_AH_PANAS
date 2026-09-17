#ifndef CONFIG_H
#define CONFIG_H

#include <Arduino.h>

// ================== PIN FREENOVE ESP32-S3 WROOM CAM (OV2640) ==================
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     15
#define SIOD_GPIO_NUM     4
#define SIOC_GPIO_NUM     5
#define Y9_GPIO_NUM       16
#define Y8_GPIO_NUM       17
#define Y7_GPIO_NUM       18
#define Y6_GPIO_NUM       12
#define Y5_GPIO_NUM       10
#define Y4_GPIO_NUM       8
#define Y3_GPIO_NUM       9
#define Y2_GPIO_NUM       11
#define VSYNC_GPIO_NUM    6
#define HREF_GPIO_NUM     7
#define PCLK_GPIO_NUM     13

#define ONBOARD_LED       2      // LED bawaan Freenove WROOM CAM
#define CONF_THRESHOLD    0.60f  // Confidence minimum target valid

// ================== PIN PCM5102 I2S AUDIO DAC ==================
#define I2S_BCLK_PIN      42     // Hubungkan ke BCK (PCM5102)
#define I2S_LRCK_PIN      41     // Hubungkan ke LCK (PCM5102)
#define I2S_DOUT_PIN      21     // Hubungkan ke DIN (PCM5102)
#define I2S_NUM           I2S_NUM_0
#define SAMPLE_RATE       22050

// ================== KONSTANTA CAMERA & EI ==================
#define EI_CAMERA_RAW_FRAME_BUFFER_COLS   320
#define EI_CAMERA_RAW_FRAME_BUFFER_ROWS   240
#define EI_CAMERA_FRAME_BYTE_SIZE         3

#endif