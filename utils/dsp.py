#!/usr/bin/env python
# coding=utf-8
#from ctypes import *
import ctypes
import numpy as np
import librosa
import soundfile as sf
import sys
import gc
gc.disable()

sys.settrace

class DSP():

    def init(self, sample_rate, nr_type, level):
        self.so = ctypes.CDLL("./libnr.dylib")

        self.so.nr_init.argtypes=[ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.so.nr_process.restype=ctypes.POINTER(ctypes.c_short)
        self.so.nr_process.argtypes=[ctypes.POINTER(ctypes.c_short)]
        self.so.nr_init(16000, 2, 4)
    
    def release(self):
        self.so.nr_release()

    def denoise(self, data):
        arr_c = (ctypes.c_short * len(data))(*data)
        tmp = self.so.nr_process(arr_c)
        return np.ctypeslib.as_array(tmp, shape=(len(data),)).tolist()

    def denoise_wav(self, wav):
        frame_size = 160;
        output=[wav[i:i + frame_size] for i in range(0, len(wav), frame_size)]
        d = []
        for x in output:
            x = (x * 32767).astype(np.int16)
            d += self.denoise(x)
        d = np.array(d)
        d = (d / 32767).astype(np.float32)
        return d;

if __name__ == '__main__':
    dsp = DSP();
    data, fs = sf.read('./samples/xfding_en.wav')
    dsp.init(fs, 1, 4)
    d = dsp.denoise_wav(data)
    dsp.release()
    sf.write("./denoised.wav", d, fs)

