#!/usr/bin/env python
# coding=utf-8
#from ctypes import *
import ctypes
import numpy as np
import librosa
import soundfile as sf

class DSP():

    def init(self):
        #so = cdll.LoadLibrary("./libvwmdsp.so")
        self.so = ctypes.CDLL("./libvwmdsp.so")

        self.so.vwm_uplink_init.restype=ctypes.c_void_p
        self.so.vwm_uplink_process_ctl.argtypes=[ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.so.vwm_uplink_process.argtypes=[ctypes.c_void_p, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int]
        self.instance = self.so.vwm_uplink_init(10, 16000, 1, 16000, 1, 1, 1)
        self.so.vwm_uplink_process_ctl(self.instance, 0, "./")

    def denoise(self, wav, samplerate):
        frame_size = 160;
        output=[wav[i:i + frame_size] for i in range(0, len(wav), frame_size)]
        d = []
        for x in output:
            x = (x * 32767).astype(np.int16)
            arr_c = (ctypes.c_short * len(x))(*x)
            
            tmp =self.so.vwm_uplink_process(self.instance, arr_c, len(x), 1, 0, arr_c, 1)
            d += np.ctypeslib.as_array(arr_c).tolist()
        #print(len(output[0]), len(output))
        d = np.array(d)
        d = (d / 32767).astype(np.float32)
        return d;

if __name__ == '__main__':
    dsp = DSP();
    dsp.init()
    data, fs = sf.read('/data/xfding/xfding_en.wav')
    print(len(data), fs)
    d = dsp.denoise(data, fs)
    sf.write("../denoised.wav", d, fs)

