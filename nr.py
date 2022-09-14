#!/usr/bin/env python
# coding=utf-8
import librosa
from utils import logmmse
import numpy as np
import soundfile as sf

sample_rate = 16000
def load_preprocess_wav(fpath):
    wav = librosa.load(path=str(fpath), sr=sample_rate)[0]

    # denoise
    if len(wav) > sample_rate*(0.3+0.1):
        noise_wav = np.concatenate([wav[:int(sample_rate*0.15)],
                                    wav[-int(sample_rate*0.15):]])
        profile = logmmse.profile_noise(noise_wav, sample_rate)
        wav = logmmse.denoise(wav, profile)
    return wav

path = "/Users/xuefeng/Downloads/test.wav"
denoised = load_preprocess_wav(path)
sf.write("out.wav", denoised, 16000)
