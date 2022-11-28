import os
import struct

import numpy as np
import librosa
from scipy.io import wavfile as wav


#This function extrcts MFCCs from a single file.
def extract_MFCC(audio):
    max_padding = 175
    try:
        mfccs = librosa.feature.mfcc(y=audio, n_mfcc=40)
        pad_width = max_padding - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    except Exception as e:
        print(e)
        return None
     
    return mfccs


def get_info(self,file):

    wave = open(file,"rb")
    riff = wave.read(12)
    fmt = wave.read(36)
        
    num_channels_string = fmt[10:12]
    num_channels = struct.unpack('<H', num_channels_string)[0]

    sample_rate_string = fmt[12:16]
    sample_rate = struct.unpack("<I",sample_rate_string)[0]
        
    bit_depth_string = fmt[22:24]
    bit_depth = struct.unpack("<H",bit_depth_string)[0]

    return (num_channels, sample_rate, bit_depth)


def preprocess_file(file):
    #sample rate conversion
    libro_audio, libro_sr = librosa.load(file)
    scipy_sr,scipy_audio = wav.read(file)
    print('Original sample rate:',scipy_sr)
    print('Resampled rate:',libro_sr)
    
    #bit depth converstion
    print('Original file min/max range: ',np.min(scipy_audio),' to ',np.max(scipy_audio))
    print('Resampled file min/max range: ',np.min(libro_audio),' to ',np.max(libro_audio))
    
    #Combining audio channels
    plt.figure(figsize=(12, 4))
    plt.plot(scipy_audio)
    print('')
    plt.figure(figsize=(12, 4))
    plt.plot(libro_audio)
    plt.show()
    
    return libro_audio,libro_sr,scipy_audio,scipy_sr
