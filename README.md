# BilingualASR (双语语音识别API)

这是一个支持维吾尔语和中文的语音识别API，单文件实现，部署简单快捷。
![01](https://github.com/GSQZ/BilingualASR/blob/main/img/01.png)
## 从GitHub获取项目

```bash
# 克隆仓库
git clone https://github.com/GSQZ/BilingualASR.git

# 进入项目目录
cd BilingualASR

# 安装依赖
pip install -r requirements.txt

# 运行服务
python main.py
```

## 主要特点

- 维吾尔语识别（支持拉丁字母和阿拉伯字母输出）
- 中文语音识别（基于Whisper模型）
- 简单RESTful API接口，便于集成到各种项目
- 轻量级设计，单文件实现便于维护

## 快速开始

### 环境准备

需要Python 3.8以上环境，建议使用虚拟环境：

```bash
# 创建虚拟环境
python -m venv venv

# 激活环境（Windows）
venv\Scripts\activate

# 激活环境（Linux/Mac）
source venv/bin/activate
```

### 安装依赖

直接用pip一键安装所有依赖：

```bash
# 使用提供的requirements.txt安装所有依赖
pip install -r requirements.txt
```

如果安装过程中遇到问题，可以尝试逐个安装核心组件：

```bash
# 先安装PyTorch
pip install torch torchvision torchaudio

# 再安装其他依赖
pip install flask librosa scikit-learn numpy faster-whisper
```

### 模型文件

程序会在首次运行时自动下载所需的模型文件：

1. **维吾尔语模型**: 将自动从 [这里](https://pan.sayqz.com/%E5%85%AC%E5%85%B1%E8%B5%84%E6%BA%90/%E4%BC%AACDN/assets/pt/UModel_best.pth) 下载到 `results/UModel_best.pth`

2. **中文Whisper模型**: 首次运行时会自动从Hugging Face下载

如果自动下载失败，您可以手动下载这些文件：

```bash
# 创建模型目录
mkdir -p results
mkdir -p whisper-small

# 手动下载维吾尔语模型
# 下载地址: https://pan.sayqz.com/%E5%85%AC%E5%85%B1%E8%B5%84%E6%BA%90/%E4%BC%AACDN/assets/pt/UModel_best.pth
# 下载后放入 results/UModel_best.pth
```

## 使用方法

### 启动服务

```bash
# 直接运行主程序
python main.py
```

服务默认在 http://localhost:5000 上运行，如果需要修改端口可以编辑main.py最后几行。

### 接口调用示例

简单的Python调用示例：

```python
import requests
import base64

# 读取音频文件转base64
with open("test.wav", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode('utf-8')

# 识别维吾尔语
resp = requests.post(
    "http://localhost:5000/recognize",
    json={"audio_base64": audio_base64, "lang": "ug"}
)
print(resp.json())
# 输出: {'lang': 'ug', 'latin': '...', 'arabic': '...'}

# 识别中文
resp = requests.post(
    "http://localhost:5000/recognize",
    json={"audio_base64": audio_base64, "lang": "zh"}
)
print(resp.json())
# 输出: {'lang': 'zh', 'text': '...'}
```

## API说明

就一个接口，简单好用：

### POST /recognize

**参数:**
- `audio_base64`: 音频文件的base64编码（必填）
- `lang`: 语言代码 - `ug`(维吾尔语) 或 `zh`(中文)，默认是`ug`

**返回:**
- 维吾尔语: `{'lang': 'ug', 'latin': '拉丁字母文本', 'arabic': '阿拉伯字母文本'}`
- 中文: `{'lang': 'zh', 'text': '识别结果'}`

## 硬件要求

能跑起来就行，但推荐配置：

- CPU: 4核以上，性能越好速度越快
- 内存: 至少8GB（16GB更好）
- 存储: 至少1GB空间（主要是模型文件）
- GPU: 可选，有的话会大幅提升性能（需要修改代码以支持GPU）

## 常见问题

1. **模型加载失败？**
   - 检查网络连接，模型需要从网上下载
   - 确保`results`目录存在并有写入权限
   - 尝试手动下载模型文件：https://pan.sayqz.com/%E5%85%AC%E5%85%B1%E8%B5%84%E6%BA%90/%E4%BC%AACDN/assets/pt/UModel_best.pth

2. **识别准确率不够高？**
   - 确保音频质量良好，避免背景噪音
   - 维吾尔语识别需要清晰发音
   - 对于中文，可以尝试调整beam_size参数（代码中搜索相关参数）

3. **部署到生产环境？**

推荐使用Gunicorn和Nginx:

```bash
# 安装gunicorn
pip install gunicorn

# 启动服务
gunicorn -w 4 -b 127.0.0.1:5000 main:app
```

然后配置Nginx反向代理到这个地址即可。

## 许可证

MIT License

Copyright (c) 2025 [GSQZ](https://github.com/GSQZ)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
