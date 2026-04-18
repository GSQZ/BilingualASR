#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""双语音频识别API。

重构说明：将模型/音频处理拆分到 asr/ 目录中，main.py 仅保留 API 路由与应用启动。
"""

import os

from flask import Flask, jsonify, request
from umsc import UgMultiScriptConverter

from asr.audio import save_base64_audio
from asr.config import DEVICE, FEATURE_LEN
from asr.uyghur_model import UModel, latin_to_arabic
from asr.whisper_service import init_whisper_model

app = Flask(__name__)

# 初始化维吾尔语识别模型
uyghur_model = UModel(FEATURE_LEN)
uyghur_model.to(DEVICE)

# 初始化中文识别模型
whisper_model = init_whisper_model()
if whisper_model is not None:
    print("中文模型加载完成！")


@app.route("/recognize", methods=["POST"])
def recognize_speech():
    """语音识别API接口。"""
    temp_path = None

    try:
        if not request.is_json:
            return jsonify({"error": "只接受JSON格式请求"}), 400

        data = request.get_json()
        if "audio_base64" not in data:
            return jsonify({"error": "缺少audio_base64字段"}), 400

        language = data.get("lang", "ug").lower()
        temp_path = save_base64_audio(data["audio_base64"])

        if language == "ug":
            latin_text = uyghur_model.predict(temp_path, DEVICE)
            arabic_text = latin_to_arabic(latin_text)
            cyrillic_text = UgMultiScriptConverter("ULS", "UCS")(latin_text)
            result = {
                "lang": "ug",
                "latin": latin_text,
                "arabic": arabic_text,
                "cyrillic": cyrillic_text,
            }

        elif language == "zh":
            if whisper_model is None:
                return jsonify({"error": "中文模型未加载，无法识别中文"}), 500

            segments, _ = whisper_model.transcribe(
                temp_path,
                language="zh",
                beam_size=5,
                condition_on_previous_text=False,
            )
            transcription = " ".join([segment.text for segment in segments])

            if "by索兰娅" in transcription or "字幕" in transcription or len(transcription.strip()) < 2:
                transcription = ""

            result = {"lang": "zh", "text": transcription}

        else:
            return jsonify({"error": f"不支持的语言: {language}，目前只支持: ug(维吾尔语), zh(中文)"}), 400

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"处理失败: {str(e)}"}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    print("双语识别API已启动，访问 http://localhost:5000/recognize")
    app.run(host="0.0.0.0", port=5000, debug=False)
