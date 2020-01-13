import numpy as np
import keras

class SpectralDataGenerator(keras.utils.Sequence):
    #Generates augmented audio data and return their MFCCs in batches
    def __init__(self, audio, labels, batch_size=32, dim=(32,32), channels=1, n_classes=10, shuffle=True):
        # init
        self.dim = dim
        self.batch_size = batch_size
        self.lables = labels
        self.audio = audio
        self.channels = channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.ID_list = list(range(0,(len(self.audio)-1)))
        self.on_epoch_end()

    def on_epoch_end(self):
        # Update indices after each epoch
        self.indexes = np.arange(len(self.ID_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        #Denotes number of batches per epoch
        return int(np.floor(len(self.ID_list) / (self.batch_size)))
    
    def __datagen(self,temp_ID_list):
        # Generate batch_size samples
        # init
        X = np.empty((self.batch_size, *self.dim, self.channels))
        y = np.empty((self.batch_size))

        #data generation
        for i, ID in enumerate(temp_ID_list):
            #convert audio to MFCC and store
            X[i] = extract_normalized_mfcc(self.audio[ID])
            
            #store labels
            y[i] = self.lables[ID]

        return X,y

    def extract_normalized_mfcc(audio):
        #Extracts MFCCs from a single audio sample.
        max_padding = 175
        try:
            mfccs = librosa.feature.mfcc(y=audio, n_mfcc=40)
            pad_width = max_padding - mfccs.shape[1]
            mean = np.mean(mfccs)
            std = np.std(mfccs)
            mfccs = (mfccs-mean)/std #normalize the MFCCs before padding
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        except Exception as e:
            print(e,file)
            return None
     
        return mfccs
    
    def __getitem__(self, index):
        #Generates a single batch of data
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size] #generates batch indices

        #Find list of IDs
        temp_ID_list = [self.ID_list[k] for k in indexes]

        #Generate data
        X,y = self.__datagen(temp_ID_list)

        return X,y