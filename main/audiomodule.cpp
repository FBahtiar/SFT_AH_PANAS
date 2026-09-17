#include "audiomodule.h"
#include "config.h"
#include <driver/i2s.h>
#include <math.h>

void setup_i2s_pcm5102() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 64,
        .use_apll = false,
        .tx_desc_auto_clear = true
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_BCLK_PIN,
        .ws_io_num = I2S_LRCK_PIN,
        .data_out_num = I2S_DOUT_PIN,
        .data_in_num = I2S_PIN_NO_CHANGE
    };

    i2s_driver_install(I2S_NUM, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_NUM, &pin_config);
    i2s_zero_dma_buffer(I2S_NUM);
}

void play_tone(int frequency_hz, int duration_ms, float pan) {
    int num_samples = (SAMPLE_RATE * duration_ms) / 1000;
    static int16_t buffer[512]; 
    int samples_per_chunk = sizeof(buffer) / (2 * sizeof(int16_t));

    int remaining_samples = num_samples;
    int sample_index = 0;

    if (pan < -1.0f) pan = -1.0f;
    if (pan > 1.0f) pan = 1.0f;

    float vol_left  = 8000.0f * (0.5f * (1.0f - pan));
    float vol_right = 8000.0f * (0.5f * (1.0f + pan));

    while (remaining_samples > 0) {
        int chunk_size = (remaining_samples < samples_per_chunk) ? remaining_samples : samples_per_chunk;

        for (int i = 0; i < chunk_size; i++) {
            float angle = (float)(sample_index + i) / SAMPLE_RATE * frequency_hz * 2.0f * 3.14159265f;
            float raw_sine = sinf(angle);

            buffer[2 * i]     = (int16_t)(raw_sine * vol_left);  
            buffer[2 * i + 1] = (int16_t)(raw_sine * vol_right); 
        }

        size_t bytes_written;
        i2s_write(I2S_NUM, buffer, chunk_size * 2 * sizeof(int16_t), &bytes_written, portMAX_DELAY);

        sample_index += chunk_size;
        remaining_samples -= chunk_size;
    }
}