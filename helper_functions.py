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
        print(e,file)
        return None
     
    return mfccs