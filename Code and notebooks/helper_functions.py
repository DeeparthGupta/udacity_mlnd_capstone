import os
import librosa
import numpy as np

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

import struct
import librosa

class WavInfo:
    
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