"""Audio feature extraction and IO helpers."""

import base64
import tempfile

import librosa
import torch
from sklearn import preprocessing

from asr.config import FFT_LEN, FEATURE_LEN, HOP_LEN, SAMPLE_RATE


def melfuture(wav_path, augument=False):
    """提取音频mel频谱特征。"""
    try:
        import soundfile as sf

        audio, s_r = sf.read(wav_path)
        if s_r != SAMPLE_RATE:
            import resampy

            audio = resampy.resample(audio, s_r, SAMPLE_RATE)
            s_r = SAMPLE_RATE
    except Exception as e:
        print(f"sf加载失败: {str(e)}，用librosa试试")
        audio, s_r = librosa.load(path=wav_path, sr=SAMPLE_RATE, res_type="kaiser_best")

    if len(audio.shape) > 1 and audio.shape[1] > 1:
        audio = audio.mean(axis=1)

    audio = preprocessing.minmax_scale(audio, axis=0)
    audio = librosa.effects.preemphasis(audio)

    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=s_r,
        n_fft=FFT_LEN,
        hop_length=HOP_LEN,
        n_mels=FEATURE_LEN,
        fmax=8000,
    )
    spec = librosa.power_to_db(spec)
    spec = (spec - spec.mean()) / spec.std()
    return torch.FloatTensor(spec)


def save_base64_audio(audio_base64):
    """保存base64音频到临时文件。"""
    try:
        if "," in audio_base64:
            audio_base64 = audio_base64.split(",")[1]

        audio_data = base64.b64decode(audio_base64)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_path = temp_file.name
            temp_file.write(audio_data)

        return temp_path
    except Exception as e:
        raise ValueError(f"base64处理出错: {str(e)}")
