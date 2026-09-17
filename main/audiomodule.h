#ifndef AUDIO_MODULE_H
#define AUDIO_MODULE_H

#include <Arduino.h>

void setup_i2s_pcm5102();

void play_tone(int frequency_hz, int duration_ms, float pan = 0.0f);

#endif