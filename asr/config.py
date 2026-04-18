"""Shared configuration for ASR pipeline."""

FEATURE_LEN = 128
SAMPLE_RATE = 22050
FFT_LEN = 1024
WINDOW_LEN = FFT_LEN
WINDOW = "hann"
HOP_LEN = 200
DEVICE = "cpu"
