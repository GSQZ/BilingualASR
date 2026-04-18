"""Uyghur ASR model and script conversion utilities."""

import os
import re

import torch
import torch.nn as nn
from umsc import UgMultiScriptConverter

from asr.audio import melfuture
from asr.config import FEATURE_LEN


class Uyghur:
    def __init__(self):
        self.uyghur_latin = "abcdefghijklmnopqrstuvwxyz éöü'"
        self._vocab_list = [self.pad_char, self.sos_char, self.eos_char] + list(self.uyghur_latin)
        self._vocab2idx = {v: idx for idx, v in enumerate(self._vocab_list)}

    def encode(self, s):
        s = (
            s.replace("-", " ")
            .replace(",", " ")
            .replace(".", " ")
            .replace("!", " ")
            .replace("?", " ")
            .replace("'", "'")
        )
        s = re.sub(r"\s+", " ", s).strip().lower()
        return [self.vocab_to_idx(v) for v in s if v in self.uyghur_latin]

    def decode(self, seq):
        vocabs = []
        for idx in seq:
            v = self.idx_to_vocab(idx)
            if idx == self.pad_idx or idx == self.eos_idx:
                break
            if idx != self.sos_idx:
                vocabs.append(v)
        return re.sub(r"\s+", " ", "".join(vocabs)).strip()

    def vocab_to_idx(self, vocab):
        return self._vocab2idx[vocab]

    def idx_to_vocab(self, idx):
        return self._vocab_list[idx]

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


uyghur_latin = Uyghur()


class ResB(nn.Module):
    def __init__(self, num_filters, kernel, pad, d=0.4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(num_filters, num_filters, kernel_size=kernel, stride=1, padding=pad, bias=False),
            nn.BatchNorm1d(num_filters),
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
    def __init__(self, num_features_input=FEATURE_LEN, load_best=False):
        super().__init__()

        self.in1 = nn.Conv1d(128, 256, 11, 2, 5 * 1, dilation=1, bias=False)
        self.in2 = nn.Conv1d(128, 256, 15, 2, 7 * 2, dilation=2, bias=False)
        self.in3 = nn.Conv1d(128, 256, 19, 2, 9 * 3, dilation=3, bias=False)
        self.concat = nn.Conv1d(256 * 3, 256, 1, 1, bias=True)
        self.relu = nn.ReLU()

        self.cnn1 = nn.Sequential(
            nn.Conv1d(256, 256, 11, 1, 5, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            ResB(256, 11, 5, 0.2),
            ResB(256, 11, 5, 0.2),
            ResB(256, 11, 5, 0.2),
            ResB(256, 11, 5, 0.2),
        )
        self.rnn = nn.GRU(256, 384, num_layers=1, batch_first=True, bidirectional=True)
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

        self.checkpoint = os.path.abspath(os.path.join(os.getcwd(), "results", "UModel"))
        self.model_loaded = self._load(load_best)

    def forward(self, x, input_lengths):
        inp = torch.cat([self.in1(x), self.in2(x), self.in3(x)], dim=1)
        inp = self.concat(inp)
        inp = self.relu(inp)
        out = self.cnn1(inp)

        out_lens = input_lengths // 2
        out = out.permute(0, 2, 1)

        out, _ = self.rnn(out)
        out = (out[:, :, : self.rnn.hidden_size] + out[:, :, self.rnn.hidden_size :]).contiguous()

        out = self.cnn2(out.permute(0, 2, 1))
        out = self.outlayer(out)
        out = self.softMax(out)
        return out, out_lens

    def _load(self, load_best=False):
        path = None
        self.trained_epochs = 0
        self.best_cer = 1.0

        best_path = self.checkpoint + "_best.pth"
        last_path = self.checkpoint + "_last.pth"

        if load_best and os.path.exists(best_path):
            path = best_path
        elif os.path.exists(last_path):
            path = last_path
        elif os.path.exists(best_path):
            path = best_path

        if path is not None:
            try:
                pack = torch.load(path, map_location="cpu")
                self.load_state_dict(pack["st_dict"])
                self.trained_epochs = pack["epoch"]
                self.best_cer = pack.get("BCER", 1.0)
                return True
            except Exception:
                path = None

        if path is None:
            try:
                import requests
                import urllib.request
                from tqdm import tqdm

                os.makedirs(os.path.dirname(self.checkpoint), exist_ok=True)
                model_url = "https://pan.sayqz.com/d/%E5%85%AC%E5%85%B1%E8%B5%84%E6%BA%90/%E4%BC%AACDN/assets/pt/UModel_best.pth"
                download_path = self.checkpoint + "_best.pth"

                try:
                    with requests.get(model_url, stream=True) as r:
                        r.raise_for_status()
                        total_size = int(r.headers.get("content-length", 0))
                        with open(download_path, "wb") as f, tqdm(
                            desc="下载模型",
                            total=total_size,
                            unit="iB",
                            unit_scale=True,
                            unit_divisor=1024,
                        ) as bar:
                            for chunk in r.iter_content(chunk_size=8192):
                                size = f.write(chunk)
                                bar.update(size)
                    path = download_path
                except Exception:
                    urllib.request.urlretrieve(model_url, download_path)
                    path = download_path

                if path is not None:
                    try:
                        pack = torch.load(path, map_location="cpu")
                        self.load_state_dict(pack["st_dict"])
                        self.trained_epochs = pack["epoch"]
                        self.best_cer = pack.get("BCER", 1.0)
                        return True
                    except Exception:
                        pass
            except Exception:
                pass

        return False

    def predict(self, path, device):
        self.eval()
        spect = melfuture(path).to(device)
        spect.unsqueeze_(0)
        xn = torch.IntTensor([spect.size(2)])
        out, xn = self.forward(spect, xn)
        text = self.greedydecode(out, xn)
        self.train()
        return text[0]

    def greedydecode(self, yps, yps_lens):
        _, max_yps = torch.max(yps, 1)
        preds = []
        for x in range(len(max_yps)):
            pred = []
            last = None
            for i in range(yps_lens[x]):
                char = int(max_yps[x][i].item())
                if char != uyghur_latin.pad_idx and char != last:
                    pred.append(char)
                last = char
            preds.append(pred)

        return [uyghur_latin.decode(pred) for pred in preds]


def latin_to_arabic(text):
    converter = UgMultiScriptConverter("ULS", "UAS")
    return converter(text)
