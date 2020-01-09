import numpy as np
import keras

class SpectralDataGenerator(keras.utils.Sequence):
    #generates augmented audio data and return their MFCCs in batches
    def __init__(self, ID_list, labels, batch_size=32, dim=(64,64,1), channels=1, n_classes=10, shuffle=True):
        # init
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.ID_list = ID_list
        self.channels = channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        # Update indices after each epoch
        self.indexes = np.arange(len(self.ID_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __datagen(self,temp_ID_list):
        # generate batch_size samples
        # init
        X = np.empty((self.batch_size, *self.dim, self.channels))
        y = np.empty((self.batch_size), dtype=int)

        #data generation
        # for i, ID in enumerate(temp_ID_list):
        #     break
        # pass

    def __len__(self):
        #denotes number of batches per epoch
        return int(np.floor(len(self.ID_list) / (self.batch_size)))