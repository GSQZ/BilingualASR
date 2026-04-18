"""Whisper model bootstrap helpers."""

import os

from faster_whisper import WhisperModel


def init_whisper_model():
    """初始化并返回Whisper模型，优先使用本地缓存。"""
    try:
        print("加载中文语音识别模型中...")
        whisper_cache_dir = os.path.join(os.getcwd(), "whisper-cache")
        os.makedirs(whisper_cache_dir, exist_ok=True)

        local_model_dir = os.path.join(os.getcwd(), "whisper-small")
        if os.path.exists(local_model_dir) and os.path.isdir(local_model_dir):
            files = os.listdir(local_model_dir)
            if len(files) > 0:
                try:
                    print(f"使用本地模型: {local_model_dir}")
                    return WhisperModel(
                        local_model_dir,
                        device="cpu",
                        compute_type="int8",
                        download_root=whisper_cache_dir,
                    )
                except Exception as e:
                    print(f"本地模型加载失败: {str(e)}")

        print("从缓存或Hugging Face下载whisper模型...")
        return WhisperModel("small", device="cpu", compute_type="int8", download_root=whisper_cache_dir)
    except Exception as e:
        print(f"中文模型加载失败: {str(e)}")
        print("只能使用维吾尔语功能了...")
        return None
