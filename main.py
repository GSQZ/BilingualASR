#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双语音频识别API - 单文件实现
支持维吾尔语和中文识别
接收base64编码的音频，返回JSON格式的识别结果

TODO: 未来考虑分割成多个模块文件，现在先这样用着
"""

import os
import re
import torch
import torch.nn as nn
import librosa
from flask import Flask, request, jsonify
import tempfile
import base64
from sklearn import preprocessing
import numpy as np
from faster_whisper import WhisperModel
import json

# ---------- 基础配置 ----------
# 音频处理参数 - 这些参数是经过反复实验调整的，别随便改
featurelen = 128  # melspec特征数，调整过几次，这个值效果最好
sample_rate = 22050
fft_len = 1024
window_len = fft_len  # 窗口长度和FFT长度保持一致
window = "hann"  # 别用别的窗口，效果会变差
hop_len = 200  # 帧移动长度

# ---------- 维吾尔语字符处理 ----------
class Uyghur():
    def __init__(self):
        # 维吾尔拉丁字母表，记得保留这些特殊字符
        self.uyghur_latin = "abcdefghijklmnopqrstuvwxyz éöü'" 
        self._vocab_list = [self.pad_char, self.sos_char, self.eos_char] + list(self.uyghur_latin)
        self._vocab2idx = {v: idx for idx, v in enumerate(self._vocab_list)}

    def encode(self, s):
        # 简单的文本预处理，把常见标点符号替换成空格
        s = s.replace("-", ' ').replace(",", ' ').replace(".", ' ').replace("!", ' ').replace("?", ' ').replace("'","'")
        s = re.sub(r'\s+',' ',s).strip().lower()
        seq = [self.vocab_to_idx(v) for v in s if v in self.uyghur_latin]
        return seq

    def decode(self, seq):
        # 解码序列到文本
        vocabs = []
        for idx in seq:
            v = self.idx_to_vocab(idx)
            if idx == self.pad_idx or idx == self.eos_idx:
                break
            elif idx == self.sos_idx:
                pass
            else:
                vocabs.append(v)
        s = re.sub(r'\s+',' ',"".join(vocabs)).strip()
        return s

    def vocab_to_idx(self, vocab):
        return self._vocab2idx[vocab]

    def idx_to_vocab(self, idx):
        return self._vocab_list[idx]

    def vocab_list(self):
        return self._vocab_list

    @property
    def vocab_size(self):
        return len(self._vocab_list)

    @property
    def pad_idx(self):
        return self.vocab_to_idx(self.pad_char)

    @property
    def sos_idx(self):
        return self.vocab_to_idx(self.sos_char)

    @property
    def eos_idx(self):
        return self.vocab_to_idx(self.eos_char)

    @property
    def pad_char(self):
        return "<pad>"

    @property
    def sos_char(self):
        return "<sos>"

    @property
    def eos_char(self):
        return "<eos>"

# 初始化Uyghur处理器
uyghur_latin = Uyghur()

# ---------- 音频特征提取 ----------
def melfuture(wav_path, augument=False):
    """提取音频mel频谱特征
    
    说明: 这个函数试过很多种实现方式，这个是最稳定的
    augument参数留着以后做数据增强用，现在还没实现
    """
    try:
        # 先用soundfile试试，速度会快很多
        import soundfile as sf
        audio, s_r = sf.read(wav_path)
        if s_r != sample_rate:
            # 重采样，这里可能会慢一点
            import resampy
            audio = resampy.resample(audio, s_r, sample_rate)
            s_r = sample_rate
    except Exception as e:
        # sf不行就用librosa，慢但稳定
        print(f"sf加载失败: {str(e)}，用librosa试试")
        audio, s_r = librosa.load(path=wav_path, sr=sample_rate, res_type='kaiser_best')
    
    # 处理多通道，简单取平均就行
    if len(audio.shape) > 1 and audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    
    # 归一化和预加重，这一步挺重要的
    audio = preprocessing.minmax_scale(audio, axis=0)
    audio = librosa.effects.preemphasis(audio)

    # 提取梅尔频谱特征 - 参数都是调出来的，别乱改
    spec = librosa.feature.melspectrogram(y=audio, sr=s_r, n_fft=fft_len, hop_length=hop_len, n_mels=featurelen, fmax=8000)  
    spec = librosa.power_to_db(spec)

    # 简单的归一化，用标准分数法
    spec = (spec - spec.mean()) / spec.std()
    spec = torch.FloatTensor(spec)
    
    return spec

# ---------- 模型定义 ----------
class ResB(nn.Module):
    def __init__(self, num_filters, kernel, pad, d=0.4):
        super().__init__()
        # 简单的残差块，没什么特别的
        self.conv = nn.Sequential(
            nn.Conv1d(num_filters, num_filters, kernel_size=kernel, stride=1, padding=pad, bias=False),
            nn.BatchNorm1d(num_filters)
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(num_filters)
        self.drop = nn.Dropout(d)

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out += identity
        out = self.bn(out)
        out = self.relu(out)
        out = self.drop(out)
        return out

class UModel(nn.Module):
    def __init__(self, num_features_input, load_best=False):
        super(UModel, self).__init__()

        # 多尺度卷积输入层，效果比单一卷积好不少
        self.in1 = nn.Conv1d(128, 256, 11, 2, 5*1, dilation=1, bias=False)
        self.in2 = nn.Conv1d(128, 256, 15, 2, 7*2, dilation=2, bias=False)
        self.in3 = nn.Conv1d(128, 256, 19, 2, 9*3, dilation=3, bias=False)
        self.concat = nn.Conv1d(256*3, 256, 1, 1, bias=True)
        self.relu = nn.ReLU()

        # 第一层CNN，提取局部特征
        self.cnn1 = nn.Sequential(
            nn.Conv1d(256, 256, 11, 1, 5, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            ResB(256, 11, 5, 0.2),
            ResB(256, 11, 5, 0.2),
            ResB(256, 11, 5, 0.2),
            ResB(256, 11, 5, 0.2)
        )
        # GRU捕获长期依赖，这里只用了一层，省内存...
        self.rnn = nn.GRU(256, 384, num_layers=1, batch_first=True, bidirectional=True)
        # 第二层CNN，融合特征
        self.cnn2 = nn.Sequential(
            ResB(384, 13, 6, 0.2),
            ResB(384, 13, 6, 0.2),
            ResB(384, 13, 6, 0.2),
            nn.Conv1d(384, 512, 17, 1, 8, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            ResB(512, 17, 8, 0.3),
            ResB(512, 17, 8, 0.3),
            nn.Conv1d(512, 1024, 1, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            ResB(1024, 1, 0, 0.0),
        )
        self.outlayer = nn.Conv1d(1024, uyghur_latin.vocab_size, 1, 1)
        self.softMax = nn.LogSoftmax(dim=1)

        self.checkpoint = 'results/UModel'
        self._load(load_best)
        print(f'维吾尔语模型已加载，参数量: {self.parameters_count(self):,}')

    def forward(self, x, input_lengths):
        inp = torch.cat([self.in1(x), self.in2(x), self.in3(x)], dim=1)
        inp = self.concat(inp)
        inp = self.relu(inp)
        out = self.cnn1(inp)

        out_lens = input_lengths//2
        out = out.permute(0, 2, 1)

        out, _ = self.rnn(out)
        out = (out[:, :, :self.rnn.hidden_size] + out[:, :, self.rnn.hidden_size:]).contiguous()

        out = self.cnn2(out.permute(0, 2, 1))
        out = self.outlayer(out)
        out = self.softMax(out) 
        return out, out_lens

    def parameters_count(self, model):
        sum_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return sum_par

    def _load(self, load_best=False):
        path = None
        self.trained_epochs = 0
        self.best_cer = 1.0
        # 优先加载最佳模型，不存在就用最新的
        if load_best == True and os.path.exists(self.checkpoint + '_best.pth'):
            path = self.checkpoint + '_best.pth'
        elif os.path.exists(self.checkpoint + '_last.pth'):
            path = self.checkpoint + '_last.pth'
        
        # 如果本地没有模型文件，尝试从网络下载
        if path is None:
            try:
                import requests
                import urllib.request
                from tqdm import tqdm
                
                # 确保目录存在
                os.makedirs(os.path.dirname(self.checkpoint), exist_ok=True)
                
                model_url = "https://pan.sayqz.com/d/%E5%85%AC%E5%85%B1%E8%B5%84%E6%BA%90/%E4%BC%AACDN/assets/pt/UModel_best.pth"
                download_path = self.checkpoint + '_best.pth'
                
                print(f"模型文件不存在，正在从{model_url}下载...")
                
                # 使用带进度条的下载
                try:
                    with requests.get(model_url, stream=True) as r:
                        r.raise_for_status()
                        total_size = int(r.headers.get('content-length', 0))
                        
                        with open(download_path, 'wb') as f, tqdm(
                                desc="下载模型",
                                total=total_size,
                                unit='iB',
                                unit_scale=True,
                                unit_divisor=1024,
                        ) as bar:
                            for chunk in r.iter_content(chunk_size=8192):
                                size = f.write(chunk)
                                bar.update(size)
                    
                    print(f"模型下载完成: {download_path}")
                    path = download_path
                except Exception as e:
                    print(f"使用requests下载失败: {str(e)}，尝试使用urllib...")
                    urllib.request.urlretrieve(model_url, download_path)
                    print(f"模型下载完成: {download_path}")
                    path = download_path
                
            except Exception as e:
                print(f"模型下载失败: {str(e)}")
                print("请手动下载模型文件到 results/UModel_best.pth")
                print("下载地址: https://pan.sayqz.com/%E5%85%AC%E5%85%B1%E8%B5%84%E6%BA%90/%E4%BC%AACDN/assets/pt/UModel_best.pth")
        
        if path is not None:
            pack = torch.load(path, map_location='cpu')
            self.load_state_dict(pack['st_dict'])
            self.trained_epochs = pack['epoch']
            self.best_cer = pack.get('BCER', 1.0)
            print(f'找到模型: {path}')
            print(f'CER: {self.best_cer:.2%} (字错率，越低越好)')
            print(f'训练轮次: {self.trained_epochs}')

    def predict(self, path, device):
        """处理一个音频文件，返回识别文本"""
        self.eval()
        spect = melfuture(path).to(device)    
        spect.unsqueeze_(0)
        xn = [spect.size(2)]
        xn = torch.IntTensor(xn)
        out, xn = self.forward(spect, xn)
        text = self.greedydecode(out, xn)
        self.train()
        return text[0]

    def greedydecode(self, yps, yps_lens):
        """CTC贪婪解码，简单粗暴但有效
        
        FIXME: 以后可以考虑beam search提高精度
        """
        _, max_yps = torch.max(yps, 1)
        preds = []
        for x in range(len(max_yps)):
            pred = []
            last = None
            for i in range(yps_lens[x]):
                char = int(max_yps[x][i].item())
                if char != uyghur_latin.pad_idx:
                    if char != last:
                        pred.append(char)
                last = char
            preds.append(pred)

        predstrs = [uyghur_latin.decode(pred) for pred in preds]
        return predstrs

# ---------- 拉丁字母到阿拉伯字母转换 ----------
class UgMultiScriptConverter:
    """维吾尔语转换工具 - 拉丁转阿拉伯
    
    这个转换表很简单，实际上有更复杂的转换规则
    不过对于API演示已经够用了
    """
    def __init__(self, source_script, target_script):
        self.source_script = source_script
        self.target_script = target_script
        
        # 基本转换表 - 如果有错误请告诉我
        self.latin_to_arabic = {
            'a': 'ئا',
            'b': 'ب',
            'c': 'چ',
            'd': 'د',
            'e': 'ە',
            'f': 'ف',
            'g': 'گ',
            'h': 'ھ',
            'i': 'ئى',
            'j': 'ج',
            'k': 'ك',
            'l': 'ل',
            'm': 'م',
            'n': 'ن',
            'o': 'ئو',
            'p': 'پ',
            'q': 'ق',
            'r': 'ر',
            's': 'س',
            't': 'ت',
            'u': 'ئۇ',
            'v': 'ۋ',
            'w': 'ۋ',
            'x': 'خ',
            'y': 'ي',
            'z': 'ز',
            'é': 'ې',
            'ö': 'ۆ',
            'ü': 'ۈ',
            "'": 'ئ',
            ' ': ' '
        }

    def __call__(self, text):
        """转换函数，接收拉丁字母文本返回阿拉伯字母文本"""
        if self.source_script == 'ULS' and self.target_script == 'UAS':
            # 拉丁转阿拉伯
            result = ""
            for char in text.lower():
                if char in self.latin_to_arabic:
                    result += self.latin_to_arabic[char]
                else:
                    result += char  # 不认识的字符保留原样
            return result
        else:
            # 其他转换方向还没实现
            print(f"不支持从 {self.source_script} 到 {self.target_script} 的转换")
            return text

def latin_to_arabic(text):
    """转换接口，让调用更简单"""
    source_script = 'ULS'  # Uyghur Latin Script
    target_script = 'UAS'  # Uyghur Arabic Script
    converter = UgMultiScriptConverter(source_script, target_script)
    return converter(text)

# ---------- API实现 ----------
app = Flask(__name__)

# 初始化维吾尔语识别模型
device = 'cpu'  # 暂时只用CPU，GPU版本以后再说
uyghur_model = UModel(featurelen)
uyghur_model.to(device)

# 初始化中文识别模型
try:
    print("加载中文语音识别模型中...")
    # 先看看有没有本地模型
    if os.path.exists('./whisper-small'):
        whisper_model = WhisperModel("./whisper-small", device="cpu", compute_type="int8")
    else:
        # 没有就下载 - 网络不好的话这一步会很慢
        whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
    print("中文模型加载完成！")
except Exception as e:
    print(f"中文模型加载失败: {str(e)}")
    print("只能使用维吾尔语功能了...")
    whisper_model = None

def save_base64_audio(audio_base64):
    """保存base64音频到临时文件
    
    谁能想到这个简单功能我调试了两小时...
    """
    try:
        # 去掉可能的data URI前缀
        if ',' in audio_base64:
            audio_base64 = audio_base64.split(',')[1]
        
        # 解码base64
        audio_data = base64.b64decode(audio_base64)
        
        # 用临时文件保存，让系统自己处理清理
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_path = temp_file.name
            temp_file.write(audio_data)
        
        return temp_path
    except Exception as e:
        raise ValueError(f"base64处理出错: {str(e)}")

@app.route('/recognize', methods=['POST'])
def recognize_speech():
    """语音识别API接口
    
    这个接口处理POST请求，接收base64音频，返回识别结果
    支持维吾尔语和中文两种语言
    """
    temp_path = None
    
    try:
        # 检查请求格式
        if not request.is_json:
            return jsonify({'error': '只接受JSON格式请求'}), 400
            
        data = request.get_json()
        
        # 检查必要参数
        if 'audio_base64' not in data:
            return jsonify({'error': '缺少audio_base64字段'}), 400
            
        # 获取语言选择，默认是维吾尔语
        language = data.get('lang', 'ug').lower()
        
        # 保存音频到临时文件
        audio_base64 = data['audio_base64']
        temp_path = save_base64_audio(audio_base64)
        
        # 根据语言选择模型
        if language == 'ug':
            # 维吾尔语识别
            latin_text = uyghur_model.predict(temp_path, device)
            arabic_text = latin_to_arabic(latin_text)
            
            result = {
                'lang': 'ug',
                'latin': latin_text,
                'arabic': arabic_text
            }
            
        elif language == 'zh':
            # 中文识别
            if whisper_model is None:
                return jsonify({'error': '中文模型未加载，无法识别中文'}), 500
                
            segments, _ = whisper_model.transcribe(temp_path, language="zh", beam_size=5, condition_on_previous_text=False)
            transcription = " ".join([segment.text for segment in segments])
            
            # 过滤掉whisper常出现的无关内容
            if "by索兰娅" in transcription or "字幕" in transcription or len(transcription.strip()) < 2:
                transcription = ""
            
            result = {
                'lang': 'zh',
                'text': transcription
            }
            
        else:
            return jsonify({'error': f'不支持的语言: {language}，目前只支持: ug(维吾尔语), zh(中文)'}), 400
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'处理失败: {str(e)}'}), 500
    
    finally:
        # 清理临时文件
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    print("双语识别API已启动，访问 http://localhost:5000/recognize")
    app.run(host='0.0.0.0', port=5000, debug=False) 
